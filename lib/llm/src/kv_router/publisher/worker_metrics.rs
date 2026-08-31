// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;

use dynamo_kv_router::protocols::{ActiveLoad, DpRank};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::kv_router::KV_METRICS_SUBJECT;

#[derive(Debug, Clone, Default, PartialEq)]
struct WorkerMetrics {
    dp_rank: DpRank,
    active_decode_blocks: Option<u64>,
    kv_used_blocks: Option<u64>,
}

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<BTreeMap<DpRank, WorkerMetrics>>,
    rx: tokio::sync::watch::Receiver<BTreeMap<DpRank, WorkerMetrics>>,
}

const PUBLISH_DEBOUNCE: Duration = Duration::from_millis(1);
const IDLE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(BTreeMap::new());
        Ok(Self { tx, rx })
    }

    pub fn publish(
        &self,
        dp_rank: Option<DpRank>,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
    ) -> Result<()> {
        if active_decode_blocks.is_none() && kv_used_blocks.is_none() {
            anyhow::bail!("worker metrics publish requires at least one load metric");
        }

        let metrics = WorkerMetrics {
            dp_rank: dp_rank.unwrap_or(0),
            active_decode_blocks,
            kv_used_blocks,
        };
        tracing::trace!(
            "Publish metrics: dp_rank={}, active_decode_blocks={:?}, kv_used_blocks={:?}",
            metrics.dp_rank,
            metrics.active_decode_blocks,
            metrics.kv_used_blocks
        );
        self.tx.send_if_modified(|latest_by_rank| {
            if latest_by_rank.get(&metrics.dp_rank) == Some(&metrics) {
                return false;
            }
            latest_by_rank.insert(metrics.dp_rank, metrics);
            true
        });
        Ok(())
    }

    pub async fn create_endpoint(&self, endpoint: Endpoint) -> Result<()> {
        let worker_id = endpoint.drt().connection_id();
        let event_publisher = EventPublisher::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await?;
        self.start_metrics_publishing(event_publisher, worker_id);
        Ok(())
    }

    pub(super) fn start_metrics_publishing(&self, event_publisher: EventPublisher, worker_id: u64) {
        let mut metrics_rx = self.rx.clone();

        tokio::spawn(async move {
            let mut latest_by_rank = BTreeMap::<DpRank, WorkerMetrics>::new();
            let mut pending_by_rank = BTreeMap::<DpRank, WorkerMetrics>::new();
            let publish_timer = tokio::time::sleep(tokio::time::Duration::ZERO);
            tokio::pin!(publish_timer);
            let first_heartbeat = tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL;
            let mut heartbeat = tokio::time::interval_at(first_heartbeat, IDLE_HEARTBEAT_INTERVAL);
            heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            loop {
                tokio::select! {
                    result = metrics_rx.changed() => {
                        if result.is_err() {
                            tracing::debug!(
                                "Metrics publisher sender dropped, stopping event-plane background task"
                            );
                            break;
                        }

                        let received = metrics_rx.borrow_and_update().clone();
                        for (&rank, metrics) in &received {
                            if latest_by_rank.get(&rank) != Some(metrics) {
                                pending_by_rank.insert(rank, metrics.clone());
                            }
                        }
                        latest_by_rank = received;
                        if !pending_by_rank.is_empty() {
                            publish_timer.as_mut().reset(
                                tokio::time::Instant::now() + PUBLISH_DEBOUNCE
                            );
                        }
                    }
                    _ = &mut publish_timer, if !pending_by_rank.is_empty() => {
                        for (_, metrics) in std::mem::take(&mut pending_by_rank) {
                            publish_metrics(&event_publisher, worker_id, &metrics).await;
                        }
                    }
                    _ = heartbeat.tick(), if !latest_by_rank.is_empty() => {
                        for metrics in latest_by_rank.values() {
                            publish_metrics(&event_publisher, worker_id, metrics).await;
                        }
                    }
                }
            }
        });
    }
}

async fn publish_metrics(
    event_publisher: &EventPublisher,
    worker_id: u64,
    metrics: &WorkerMetrics,
) {
    let active_load = ActiveLoad {
        worker_id,
        dp_rank: metrics.dp_rank,
        active_decode_blocks: metrics.active_decode_blocks,
        active_prefill_tokens: None,
        kv_used_blocks: metrics.kv_used_blocks,
    };

    if let Err(error) = event_publisher.publish(&active_load).await {
        tracing::warn!(%error, dp_rank = metrics.dp_rank, "Failed to publish metrics");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_latest_metrics_for_every_rank() {
        let publisher = WorkerMetricsPublisher::new().unwrap();

        publisher.publish(Some(0), None, Some(0)).unwrap();
        publisher.publish(Some(1), None, Some(7)).unwrap();
        publisher.publish(Some(0), None, Some(3)).unwrap();

        let latest = publisher.rx.borrow();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[&0].kv_used_blocks, Some(3));
        assert_eq!(latest[&1].kv_used_blocks, Some(7));
    }
}
