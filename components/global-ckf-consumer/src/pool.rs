// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity as NativeProducerIdentity;
use dynamo_llm::kv_dc_relay::protocol::{
    FilterUpdate, ProducerIdentity, RELAY_CONTRACT_MARKER, SubscribeKvPoolRequest,
    validate_protocol_envelope,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::config::{Config, RelayConfig};
use crate::lane::{LaneSet, LaneUnavailableReason};
use crate::relay::{connect_and_verify, validate_stream_identity};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolStreamState {
    Connecting,
    Rebuilding,
    Streaming,
    Disconnected(String),
}

pub struct PoolSupervisor {
    pub state: watch::Receiver<PoolStreamState>,
    task: JoinHandle<()>,
}

impl Drop for PoolSupervisor {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub fn spawn_pool_supervisor(
    relay: RelayConfig,
    config: Config,
    wire_producer: ProducerIdentity,
    native_producer: NativeProducerIdentity,
    lanes: Arc<LaneSet>,
) -> PoolSupervisor {
    let (state_tx, state_rx) = watch::channel(PoolStreamState::Connecting);
    let task = tokio::spawn(async move {
        let mut attempt = 0u32;
        loop {
            let _ = state_tx.send(PoolStreamState::Connecting);
            let result = run_generation(
                &relay,
                &config,
                &wire_producer,
                native_producer,
                &lanes,
                &state_tx,
            )
            .await;
            let error = result
                .err()
                .map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "pool stream ended".into());
            let _ = state_tx.send(PoolStreamState::Disconnected(error));
            attempt = attempt.saturating_add(1);
            tokio::time::sleep(reconnect_delay(attempt)).await;
        }
    });
    PoolSupervisor {
        state: state_rx,
        task,
    }
}

async fn run_generation(
    relay_config: &RelayConfig,
    config: &Config,
    wire_producer: &ProducerIdentity,
    native_producer: NativeProducerIdentity,
    lanes: &Arc<LaneSet>,
    state_tx: &watch::Sender<PoolStreamState>,
) -> Result<()> {
    let (mut client, relay) = connect_and_verify(relay_config, config).await?;
    let lease = lanes.assign(native_producer, Instant::now())?;
    let _guard = LaneRetirementGuard::new(Arc::clone(lanes), lease);
    let _ = state_tx.send(PoolStreamState::Rebuilding);
    let mut stream = client
        .subscribe_kv_pool(SubscribeKvPoolRequest {
            subscriber_id: config.subscriber_id.clone(),
            expected_producer: Some(wire_producer.clone()),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("open pool stream")?
        .into_inner();
    while let Some(update) = stream.message().await.context("read pool stream")? {
        validate_filter_update(&relay.identity, wire_producer, &update)?;
        let outcome = lanes.ingest(lease, &update, Instant::now())?;
        if matches!(
            outcome,
            crate::lane::IngestedUpdate::SnapshotInstalled { .. }
        ) {
            let _ = state_tx.send(PoolStreamState::Streaming);
        }
    }
    bail!("pool stream closed")
}

fn validate_filter_update(
    relay: &dynamo_llm::kv_dc_relay::protocol::RelayIdentity,
    producer: &ProducerIdentity,
    update: &FilterUpdate,
) -> Result<()> {
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    validate_stream_identity(relay, update.relay.as_ref())?;
    if update.producer.as_ref() != Some(producer) {
        bail!("pool stream producer identity changed");
    }
    Ok(())
}

struct LaneRetirementGuard {
    lanes: Arc<LaneSet>,
    lease: dynamo_kv_router::indexer::cuckoo::LaneLease,
}

impl LaneRetirementGuard {
    fn new(lanes: Arc<LaneSet>, lease: dynamo_kv_router::indexer::cuckoo::LaneLease) -> Self {
        Self { lanes, lease }
    }
}

impl Drop for LaneRetirementGuard {
    fn drop(&mut self) {
        if let Err(error) = self
            .lanes
            .retire(self.lease, LaneUnavailableReason::Transport)
        {
            tracing::warn!(error = %error, "failed to retire disconnected pool lane");
        }
    }
}

fn reconnect_delay(attempt: u32) -> Duration {
    Duration::from_secs(1u64 << attempt.min(5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::kv_dc_relay::protocol::{RELAY_PROTOCOL_VERSION, RelayIdentity};

    #[test]
    fn filter_update_requires_exact_relay_and_producer_generation() {
        let relay = RelayIdentity {
            drt_instance_id: 1,
            relay_incarnation: 2,
        };
        let producer = ProducerIdentity::default();
        let update = FilterUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay.clone()),
            producer: Some(producer.clone()),
            contract_marker: RELAY_CONTRACT_MARKER,
            ..Default::default()
        };
        assert!(validate_filter_update(&relay, &producer, &update).is_ok());
        let mut wrong = update;
        wrong.relay.as_mut().unwrap().relay_incarnation += 1;
        assert!(validate_filter_update(&relay, &producer, &wrong).is_err());
    }

    #[test]
    fn reconnect_delay_is_bounded() {
        assert_eq!(reconnect_delay(0), Duration::from_secs(1));
        assert_eq!(reconnect_delay(20), Duration::from_secs(32));
    }
}
