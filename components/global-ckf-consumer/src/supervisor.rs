// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_llm::kv_dc_relay::protocol::{
    KvPoolCatalogSnapshot, KvPoolCatalogUpdate, KvPoolDescriptor, KvPoolLoadUpdate,
    RELAY_CONTRACT_MARKER, RelayIdentity, ServingReadinessUpdate, SubscribeKvPoolLoadRequest,
    SubscribeServingReadinessRequest, WatchKvPoolCatalogRequest, validate_pool_descriptor,
    validate_producer_identity, validate_protocol_envelope, validate_topology_entry,
};
use tokio::sync::mpsc;

use crate::config::{Config, RelayConfig};
use crate::relay::{VerifiedRelay, connect_and_verify, validate_stream_identity};

const MIN_RECONNECT_DELAY: Duration = Duration::from_secs(1);
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub enum RelayEvent {
    Connected(VerifiedRelay),
    Catalog {
        relay: VerifiedRelay,
        revision: u64,
        snapshot: KvPoolCatalogSnapshot,
    },
    Readiness {
        relay: VerifiedRelay,
        revision: u64,
        entries: Vec<dynamo_llm::kv_dc_relay::protocol::TopologyEntry>,
    },
    Load {
        relay: VerifiedRelay,
        update: KvPoolLoadUpdate,
    },
    Disconnected {
        name: String,
        error: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct RelayState {
    pub identity: Option<RelayIdentity>,
    pub catalog_revision: Option<u64>,
    pub catalog: Vec<KvPoolDescriptor>,
    pub readiness_revision: Option<u64>,
    pub readiness: Vec<dynamo_llm::kv_dc_relay::protocol::TopologyEntry>,
    pub readiness_updated: Option<Instant>,
    pub load: Option<KvPoolLoadUpdate>,
    pub load_updated: Option<Instant>,
    pub connected: bool,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct FleetState {
    pub relays: BTreeMap<String, RelayState>,
}

impl FleetState {
    pub fn apply(&mut self, event: RelayEvent) {
        match event {
            RelayEvent::Connected(relay) => {
                let state = self.relays.entry(relay.name).or_default();
                if state.identity.as_ref() != Some(&relay.identity) {
                    *state = RelayState::default();
                    state.identity = Some(relay.identity);
                }
                state.connected = true;
                state.last_error = None;
            }
            RelayEvent::Catalog {
                relay,
                revision,
                snapshot,
            } => {
                let state = self.relays.entry(relay.name).or_default();
                state.identity = Some(relay.identity);
                state.catalog_revision = Some(revision);
                state.catalog = snapshot.pools;
            }
            RelayEvent::Readiness {
                relay,
                revision,
                entries,
            } => {
                let state = self.relays.entry(relay.name).or_default();
                state.identity = Some(relay.identity);
                state.readiness_revision = Some(revision);
                state.readiness = entries;
                state.readiness_updated = Some(Instant::now());
            }
            RelayEvent::Load { relay, update } => {
                let state = self.relays.entry(relay.name).or_default();
                state.identity = Some(relay.identity);
                state.load = Some(update);
                state.load_updated = Some(Instant::now());
            }
            RelayEvent::Disconnected { name, error } => {
                let state = self.relays.entry(name).or_default();
                state.connected = false;
                state.last_error = Some(error);
            }
        }
    }
}

pub fn spawn_relay_supervisors(config: &Config) -> mpsc::Receiver<RelayEvent> {
    let capacity = (config.relays.len() * 16).max(16);
    let (tx, rx) = mpsc::channel(capacity);
    for relay in config.relays.iter().cloned() {
        let config = config.clone();
        let tx = tx.clone();
        tokio::spawn(async move { supervise_relay(relay, config, tx).await });
    }
    drop(tx);
    rx
}

async fn supervise_relay(relay: RelayConfig, config: Config, tx: mpsc::Sender<RelayEvent>) {
    let mut attempt = 0u32;
    loop {
        let result = run_generation(&relay, &config, &tx).await;
        let error = result
            .err()
            .map(|error| format!("{error:#}"))
            .unwrap_or_else(|| "relay streams ended".to_string());
        if tx
            .send(RelayEvent::Disconnected {
                name: relay.name.clone(),
                error,
            })
            .await
            .is_err()
        {
            return;
        }
        tokio::time::sleep(reconnect_delay(&relay.name, attempt)).await;
        attempt = attempt.saturating_add(1);
    }
}

async fn run_generation(
    relay_config: &RelayConfig,
    config: &Config,
    tx: &mpsc::Sender<RelayEvent>,
) -> Result<()> {
    let (mut client, relay) = connect_and_verify(relay_config, config).await?;
    tx.send(RelayEvent::Connected(relay.clone())).await?;

    let subscriber_id = config.subscriber_id.clone();
    let mut catalog = client
        .watch_kv_pool_catalog(WatchKvPoolCatalogRequest {
            subscriber_id: subscriber_id.clone(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("open catalog stream")?
        .into_inner();
    let mut readiness = client
        .subscribe_serving_readiness(SubscribeServingReadinessRequest {
            subscriber_id: subscriber_id.clone(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("open readiness stream")?
        .into_inner();
    let mut load = client
        .subscribe_kv_pool_load(SubscribeKvPoolLoadRequest {
            subscriber_id,
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("open load stream")?
        .into_inner();

    let mut catalog_revision = None;
    let mut readiness_revision = None;
    let mut load_sequence = None;
    loop {
        let event = tokio::select! {
            message = catalog.message() => {
                let update = message.context("read catalog stream")?.context("catalog stream closed")?;
                validate_catalog(&relay, &update, catalog_revision)?;
                catalog_revision = Some(update.revision);
                RelayEvent::Catalog {
                    relay: relay.clone(),
                    revision: update.revision,
                    snapshot: update.snapshot.context("catalog update omitted its snapshot")?,
                }
            }
            message = readiness.message() => {
                let update = message.context("read readiness stream")?.context("readiness stream closed")?;
                validate_readiness(&relay, &update, readiness_revision)?;
                readiness_revision = Some(update.revision);
                RelayEvent::Readiness {
                    relay: relay.clone(),
                    revision: update.revision,
                    entries: update.entries,
                }
            }
            message = load.message() => {
                let update = message.context("read load stream")?.context("load stream closed")?;
                validate_load(&relay, &update, load_sequence)?;
                load_sequence = Some(update.window_sequence);
                RelayEvent::Load { relay: relay.clone(), update }
            }
        };
        tx.send(event).await.context("publish relay update")?;
    }
}

fn validate_catalog(
    relay: &VerifiedRelay,
    update: &KvPoolCatalogUpdate,
    previous_revision: Option<u64>,
) -> Result<()> {
    validate_envelope_and_identity(
        relay,
        update.protocol_version,
        update.contract_marker,
        update.relay.as_ref(),
    )?;
    require_increasing("catalog revision", previous_revision, update.revision)?;
    let snapshot = update
        .snapshot
        .as_ref()
        .context("catalog update omitted its snapshot")?;
    for descriptor in &snapshot.pools {
        validate_pool_descriptor(descriptor)?;
        let dc_id = descriptor
            .producer
            .as_ref()
            .and_then(|producer| producer.pool_id.as_ref())
            .map(|pool| pool.dc_id);
        if dc_id != Some(relay.expected_dc_id) {
            bail!(
                "relay {} published pool for DC {:?}, expected {}",
                relay.name,
                dc_id,
                relay.expected_dc_id
            );
        }
    }
    Ok(())
}

fn validate_readiness(
    relay: &VerifiedRelay,
    update: &ServingReadinessUpdate,
    previous_revision: Option<u64>,
) -> Result<()> {
    validate_envelope_and_identity(
        relay,
        update.protocol_version,
        update.contract_marker,
        update.relay.as_ref(),
    )?;
    if let Some(previous) = previous_revision {
        if update.revision < previous {
            bail!(
                "readiness revision regressed from {previous} to {}",
                update.revision
            );
        }
    }
    for entry in &update.entries {
        validate_topology_entry(entry)?;
    }
    Ok(())
}

fn validate_load(
    relay: &VerifiedRelay,
    update: &KvPoolLoadUpdate,
    previous_sequence: Option<u64>,
) -> Result<()> {
    validate_envelope_and_identity(
        relay,
        update.protocol_version,
        update.contract_marker,
        update.relay.as_ref(),
    )?;
    require_increasing(
        "load window sequence",
        previous_sequence,
        update.window_sequence,
    )?;
    for entry in &update.pools {
        let producer = entry
            .producer
            .as_ref()
            .context("load entry omitted producer identity")?;
        validate_producer_identity(producer)?;
        let dc_id = producer.pool_id.as_ref().map(|pool| pool.dc_id);
        if dc_id != Some(relay.expected_dc_id) {
            bail!(
                "relay {} published load for DC {:?}, expected {}",
                relay.name,
                dc_id,
                relay.expected_dc_id
            );
        }
    }
    Ok(())
}

fn validate_envelope_and_identity(
    relay: &VerifiedRelay,
    protocol_version: u32,
    contract_marker: u32,
    actual: Option<&RelayIdentity>,
) -> Result<()> {
    validate_protocol_envelope(protocol_version, contract_marker)?;
    validate_stream_identity(&relay.identity, actual)
}

fn require_increasing(label: &str, previous: Option<u64>, current: u64) -> Result<()> {
    if previous.is_some_and(|previous| current <= previous) {
        bail!("{label} did not increase: previous {previous:?}, current {current}");
    }
    Ok(())
}

fn reconnect_delay(name: &str, attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let base_ms = MIN_RECONNECT_DELAY.as_millis() as u64 * (1u64 << exponent);
    let capped_ms = base_ms.min(MAX_RECONNECT_DELAY.as_millis() as u64);
    let hash = name.bytes().fold(0u64, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(byte as u64)
    });
    let jitter_percent = 80 + ((hash + attempt as u64 * 17) % 41);
    Duration::from_millis(capped_ms * jitter_percent / 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relay(name: &str, incarnation: u64) -> VerifiedRelay {
        VerifiedRelay {
            name: name.into(),
            expected_dc_id: 7,
            identity: RelayIdentity {
                drt_instance_id: 5,
                relay_incarnation: incarnation,
            },
        }
    }

    #[test]
    fn disconnect_is_scoped_to_one_relay() {
        let mut state = FleetState::default();
        state.apply(RelayEvent::Connected(relay("a", 1)));
        state.apply(RelayEvent::Connected(relay("b", 1)));
        state.apply(RelayEvent::Disconnected {
            name: "a".into(),
            error: "lost".into(),
        });
        assert!(!state.relays["a"].connected);
        assert!(state.relays["b"].connected);
    }

    #[test]
    fn new_relay_generation_clears_only_its_stale_snapshots() {
        let mut state = FleetState::default();
        state.apply(RelayEvent::Connected(relay("a", 1)));
        state.relays.get_mut("a").unwrap().catalog_revision = Some(9);
        state.apply(RelayEvent::Connected(relay("a", 2)));
        assert_eq!(
            state.relays["a"]
                .identity
                .as_ref()
                .unwrap()
                .relay_incarnation,
            2
        );
        assert_eq!(state.relays["a"].catalog_revision, None);
    }

    #[test]
    fn sequence_validation_rejects_duplicates_and_regressions() {
        assert!(require_increasing("sequence", None, 0).is_ok());
        assert!(require_increasing("sequence", Some(4), 5).is_ok());
        assert!(require_increasing("sequence", Some(4), 4).is_err());
        assert!(require_increasing("sequence", Some(4), 3).is_err());
    }

    #[test]
    fn reconnect_backoff_is_bounded_and_jittered_by_relay() {
        let a = reconnect_delay("a", 0);
        let b = reconnect_delay("b", 0);
        assert_ne!(a, b);
        assert!(a >= Duration::from_millis(800));
        assert!(reconnect_delay("a", 20) <= Duration::from_secs(36));
    }
}
