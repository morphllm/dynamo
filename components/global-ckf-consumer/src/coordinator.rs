// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dynamo_kv_router::identity::{IndexerDomainId, PoolId};
use dynamo_kv_router::indexer::cuckoo::{
    ConsumerInstanceId, DcCkfFormatIdentity, GlobalCkfIngestionPoolConfig,
    ProducerIdentity as NativeProducerIdentity,
};
use dynamo_llm::kv_dc_relay::protocol::{
    KvPoolDescriptor, ProducerIdentity, pool_id_from_wire, producer_identity_from_wire,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::api::{AppState, PoolMetadata, PoolSignals, PublishedGroup, ReadinessSignal};
use crate::config::{Config, RelayConfig};
use crate::lane::{LaneAvailability, LaneSet};
use crate::pool::{PoolStreamState, PoolSupervisor, spawn_pool_supervisor};
use crate::supervisor::{FleetState, RelayEvent};

static NEXT_CONSUMER: AtomicU64 = AtomicU64::new(1);

struct DesiredPool {
    relay: RelayConfig,
    wire: ProducerIdentity,
    native: NativeProducerIdentity,
    metadata: PoolMetadata,
}

struct DesiredGroup {
    domain: IndexerDomainId,
    format: DcCkfFormatIdentity,
    pools: Vec<DesiredPool>,
}

struct Candidate {
    groups: Vec<PublishedGroup>,
    supervisors: Vec<PoolSupervisor>,
}

pub async fn run(
    config: Config,
    app: AppState,
    mut events: mpsc::Receiver<RelayEvent>,
) -> Result<()> {
    let relays: HashMap<_, _> = config
        .relays
        .iter()
        .cloned()
        .map(|relay| (relay.name.clone(), relay))
        .collect();
    let mut fleet = FleetState::default();
    let mut desired_generation = 0u64;
    let mut candidate: Option<JoinHandle<Result<(u64, Candidate)>>> = None;
    let mut active_supervisors = Vec::new();
    let mut maintenance = tokio::time::interval(Duration::from_secs(1));
    maintenance.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = maintenance.tick() => {
                let _active_pool_streams = active_supervisors.len();
                app.maintain(Instant::now());
            },
            event = events.recv() => {
                let Some(event) = event else { return Ok(()) };
                let catalog_changed = matches!(event, RelayEvent::Catalog { .. } | RelayEvent::Disconnected { .. } | RelayEvent::Connected(_));
                retire_withdrawn(&app, &fleet, &event);
                fleet.apply(event);
                app.replace_signals(collect_signals(&fleet));
                if catalog_changed {
                    desired_generation = desired_generation.saturating_add(1);
                    if let Some(task) = candidate.take() {
                        task.abort();
                    }
                    let desired = desired_groups(&fleet, &relays)?;
                    let generation = desired_generation;
                    let config = config.clone();
                    candidate = Some(tokio::spawn(async move {
                        let candidate = build_candidate(config, desired).await?;
                        Ok((generation, candidate))
                    }));
                }
            }
            completed = async { candidate.as_mut().expect("guarded candidate").await }, if candidate.is_some() => {
                candidate = None;
                match completed {
                    Ok(Ok((generation, ready))) if generation == desired_generation => {
                        let Candidate { groups, supervisors } = ready;
                        app.publish(groups).map_err(anyhow::Error::msg)?;
                        active_supervisors = supervisors;
                        tracing::info!(generation, pools = active_supervisors.len(), "published complete global CKF catalog generation");
                    }
                    Ok(Ok((_generation, _ready))) => {}
                    Ok(Err(error)) => tracing::warn!(error = %error, "global CKF catalog candidate failed"),
                    Err(error) if error.is_cancelled() => {}
                    Err(error) => tracing::warn!(error = %error, "global CKF catalog candidate task failed"),
                }
            }
        }
    }
}

fn collect_signals(fleet: &FleetState) -> HashMap<PoolId, PoolSignals> {
    let mut signals = HashMap::new();
    for state in fleet.relays.values().filter(|state| state.connected) {
        for entry in &state.readiness {
            for member in &entry.members {
                let Some(pool) = member.pool_id.as_ref() else {
                    continue;
                };
                let Ok(pool) = pool_id_from_wire(pool) else {
                    continue;
                };
                let signal = signals.entry(pool).or_insert_with(PoolSignals::default);
                signal.readiness_updated = state.readiness_updated;
                signal.readiness.push(ReadinessSignal {
                    revision: state.readiness_revision.unwrap_or_default(),
                    canonical_model_id: entry.canonical_model_id.clone(),
                    state: entry.state,
                    present_roles: entry.present_roles.clone(),
                    missing_roles: entry.missing_roles.clone(),
                });
            }
        }
        let Some(load) = &state.load else { continue };
        for entry in &load.pools {
            let Some(producer) = entry.producer.as_ref() else {
                continue;
            };
            let Ok(producer) = producer_identity_from_wire(producer) else {
                continue;
            };
            let signal = signals
                .entry(producer.pool_id())
                .or_insert_with(PoolSignals::default);
            signal.load_window_sequence = Some(load.window_sequence);
            signal.load_observed_ms = Some(load.observed_ms);
            signal.load_window_ms = Some(load.window_ms);
            signal.load_updated = state.load_updated;
            signal.kv_used_blocks = Some(entry.kv_used_blocks);
            signal.total_kv_blocks = Some(entry.total_kv_blocks);
            signal.active_decode_blocks = Some(entry.active_decode_blocks);
            signal.active_prefill_tokens = Some(entry.active_prefill_tokens);
            signal.prefill_token_capacity = Some(entry.prefill_token_capacity);
        }
    }
    signals
}

fn retire_withdrawn(app: &AppState, fleet: &FleetState, event: &RelayEvent) {
    match event {
        RelayEvent::Disconnected { name, .. } => app.retire_relay_except(name, &HashSet::new()),
        RelayEvent::Catalog {
            relay, snapshot, ..
        } => {
            let retained = snapshot
                .pools
                .iter()
                .filter_map(|descriptor| descriptor.producer.as_ref())
                .filter_map(|producer| producer_identity_from_wire(producer).ok())
                .map(|producer| producer.pool_id())
                .collect();
            app.retire_relay_except(&relay.name, &retained);
        }
        RelayEvent::Connected(relay) => {
            if fleet
                .relays
                .get(&relay.name)
                .is_some_and(|state| state.identity.as_ref() != Some(&relay.identity))
            {
                app.retire_relay_except(&relay.name, &HashSet::new());
            }
        }
        RelayEvent::Readiness { .. } | RelayEvent::Load { .. } => {}
    }
}

fn desired_groups(
    fleet: &FleetState,
    relays: &HashMap<String, RelayConfig>,
) -> Result<Vec<DesiredGroup>> {
    let mut groups: Vec<DesiredGroup> = Vec::new();
    for (relay_name, state) in &fleet.relays {
        if !state.connected {
            continue;
        }
        let relay = relays
            .get(relay_name)
            .with_context(|| format!("missing configuration for relay {relay_name}"))?;
        for descriptor in &state.catalog {
            let wire = descriptor
                .producer
                .clone()
                .context("catalog descriptor omitted producer")?;
            let native = producer_identity_from_wire(&wire)
                .context("convert catalog producer to native CKF identity")?;
            let metadata = metadata(relay_name, descriptor, native.pool_id());
            let group = groups.iter_mut().find(|group| {
                group.domain == native.indexer_domain() && group.format == native.format()
            });
            if let Some(group) = group {
                group.pools.push(DesiredPool {
                    relay: relay.clone(),
                    wire,
                    native,
                    metadata,
                });
            } else {
                groups.push(DesiredGroup {
                    domain: native.indexer_domain(),
                    format: native.format(),
                    pools: vec![DesiredPool {
                        relay: relay.clone(),
                        wire,
                        native,
                        metadata,
                    }],
                });
            }
        }
    }
    Ok(groups)
}

fn metadata(relay: &str, descriptor: &KvPoolDescriptor, pool_id: PoolId) -> PoolMetadata {
    let mut models = HashSet::new();
    for registration in &descriptor.registrations {
        models.insert(registration.canonical_model_id.clone());
        models.extend(registration.aliases.iter().cloned());
    }
    PoolMetadata {
        pool_id,
        relay: relay.to_owned(),
        models,
        roles: descriptor.pool_roles.iter().copied().collect(),
    }
}

async fn build_candidate(config: Config, groups: Vec<DesiredGroup>) -> Result<Candidate> {
    let mut published = Vec::with_capacity(groups.len());
    let mut supervisors = Vec::new();
    for desired in groups {
        let lanes = Arc::new(LaneSet::new(
            ConsumerInstanceId::new(NEXT_CONSUMER.fetch_add(1, Ordering::Relaxed)),
            desired.pools.iter().map(|pool| pool.native),
            Duration::from_secs(config.freshness_timeout_seconds),
            GlobalCkfIngestionPoolConfig::default(),
        )?);
        let mut metadata = Vec::with_capacity(desired.pools.len());
        for pool in desired.pools {
            metadata.push(pool.metadata);
            supervisors.push(spawn_pool_supervisor(
                pool.relay,
                config.clone(),
                pool.wire,
                pool.native,
                Arc::clone(&lanes),
            ));
        }
        published.push(PublishedGroup {
            lanes,
            pools: metadata,
        });
    }

    wait_until_ready(&published, &mut supervisors).await?;
    Ok(Candidate {
        groups: published,
        supervisors,
    })
}

async fn wait_until_ready(
    groups: &[PublishedGroup],
    supervisors: &mut [PoolSupervisor],
) -> Result<()> {
    if supervisors.is_empty() {
        return Ok(());
    }
    let mut maintenance = tokio::time::interval(Duration::from_millis(250));
    loop {
        let all_ready = supervisors
            .iter()
            .all(|supervisor| matches!(*supervisor.state.borrow(), PoolStreamState::Streaming));
        let all_lanes_ready = groups.iter().all(|group| {
            group.lanes.statuses(Instant::now()).is_ok_and(|statuses| {
                !statuses.is_empty()
                    && statuses
                        .iter()
                        .all(|status| status.availability == LaneAvailability::Available)
            })
        });
        if all_ready && all_lanes_ready {
            return Ok(());
        }
        maintenance.tick().await;
        for group in groups {
            group.lanes.retire_stale(Instant::now())?;
            for (pool, fault) in group.lanes.drain_faults()? {
                tracing::warn!(pool = %pool, ?fault, "native CKF ingestion fault while rebuilding catalog");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::kv_dc_relay::protocol::{CkfFormat, KvPoolId};

    #[test]
    fn metadata_keeps_canonical_models_aliases_and_roles() {
        let descriptor = KvPoolDescriptor {
            registrations: vec![dynamo_llm::kv_dc_relay::protocol::ModelRegistration {
                canonical_model_id: "model".into(),
                aliases: vec!["alias".into()],
                ..Default::default()
            }],
            pool_roles: vec![1, 2],
            ..Default::default()
        };
        let domain = dynamo_kv_router::identity::IndexerDomainId::new(
            dynamo_kv_router::identity::CacheSemanticsId::new(
                [1; 16],
                dynamo_kv_router::identity::IdentitySource::Explicit,
            ),
            dynamo_kv_router::identity::RoutingScopeId::new(
                [2; 16],
                dynamo_kv_router::identity::IdentitySource::Explicit,
            ),
        );
        let result = metadata(
            "east",
            &descriptor,
            PoolId::new(domain, dynamo_kv_router::identity::DcId::new(1)),
        );
        assert_eq!(
            result.models,
            HashSet::from(["model".into(), "alias".into()])
        );
        assert_eq!(result.roles, HashSet::from([1, 2]));
    }

    #[test]
    fn exact_format_is_part_of_group_identity() {
        let a = CkfFormat {
            seed: 1,
            bucket_count: 8,
            ..Default::default()
        };
        let b = CkfFormat {
            seed: 2,
            bucket_count: 8,
            ..Default::default()
        };
        assert_ne!(a, b);
        let _ = KvPoolId::default();
    }
}
