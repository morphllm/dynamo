// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{
    CKF_LANE_COUNT, ConsumerInstanceId, GlobalCkfBucketImage, GlobalCkfDelta, GlobalCkfIndexer,
    GlobalCkfIngestOutcome, GlobalCkfIngestionError, GlobalCkfIngestionFault,
    GlobalCkfIngestionPool, GlobalCkfIngestionPoolBuildError, GlobalCkfIngestionPoolConfig,
    GlobalCkfManifest, GlobalCkfSnapshot, LaneLease, PrefixSearchConfig, ProducerIdentity,
};
use dynamo_llm::kv_dc_relay::protocol::wire::images::{
    FilterFormat, FormatError, ImagesFrame, ImagesWireError, SnapshotAssembly, decode,
};
use dynamo_llm::kv_dc_relay::protocol::{FilterUpdate, FilterUpdateKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneUnavailableReason {
    Absent,
    Rebuilding,
    Stale,
    Transport,
    Ingestion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneAvailability {
    Available,
    Unavailable(LaneUnavailableReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneStatus {
    pub pool_id: PoolId,
    pub physical_lane: u8,
    pub lease: Option<LaneLease>,
    pub producer: Option<ProducerIdentity>,
    pub availability: LaneAvailability,
    pub last_sequence: Option<u64>,
    pub last_update_age: Option<Duration>,
}

#[derive(Debug, thiserror::Error)]
pub enum LaneError {
    #[error("global CKF lane set must contain at least one producer")]
    Empty,
    #[error("producer for pool {0} is not configured in this lane set")]
    UnknownPool(PoolId),
    #[error("lane {lane} assignment epoch exhausted")]
    AssignmentEpochExhausted { lane: u8 },
    #[error("lease no longer owns its lane")]
    StaleLease,
    #[error("unsupported filter update kind {0}")]
    UnsupportedUpdateKind(i32),
    #[error("snapshot frame sequence {frame} differs from envelope sequence {envelope}")]
    SnapshotSequenceMismatch { frame: u64, envelope: u64 },
    #[error(
        "delta frame sequence ({base}, {sequence}) differs from envelope ({envelope_base}, {envelope_sequence})"
    )]
    DeltaSequenceMismatch {
        base: u64,
        sequence: u64,
        envelope_base: u64,
        envelope_sequence: u64,
    },
    #[error("frame DC {frame} differs from producer DC {producer}")]
    DcMismatch { frame: u64, producer: u64 },
    #[error(transparent)]
    Wire(#[from] ImagesWireError),
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Build(#[from] GlobalCkfIngestionPoolBuildError),
    #[error(transparent)]
    Ingestion(#[from] GlobalCkfIngestionError),
    #[error("global CKF manifest is invalid: {0}")]
    Manifest(String),
    #[error("lane state mutex is poisoned")]
    StatePoisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestedUpdate {
    SnapshotChunk,
    SnapshotInstalled { sequence: u64 },
    DeltaSubmitted { sequence: u64 },
    Heartbeat { sequence: u64 },
}

struct LaneState {
    pool_id: PoolId,
    producer: Option<ProducerIdentity>,
    lease: Option<LaneLease>,
    next_assignment_epoch: u64,
    availability: LaneAvailability,
    last_sequence: Option<u64>,
    last_update: Option<Instant>,
    snapshot: Option<SnapshotAssembly>,
}

/// One immutable indexer domain backed by Dynamo's bounded ingestion pool.
///
/// The relay coordinator owns the stream tasks. It calls `assign`, carries the returned lease
/// with that exact stream generation, passes validated filter updates to `ingest`, and calls
/// `retire` when the stream or catalog entry disappears.
pub struct LaneSet {
    consumer: ConsumerInstanceId,
    format: FilterFormat,
    freshness_timeout: Duration,
    lanes_by_pool: HashMap<PoolId, usize>,
    lanes: Mutex<Vec<LaneState>>,
    ingestion: Arc<GlobalCkfIngestionPool>,
}

impl LaneSet {
    pub fn new(
        consumer: ConsumerInstanceId,
        producers: impl IntoIterator<Item = ProducerIdentity>,
        freshness_timeout: Duration,
        ingestion_config: GlobalCkfIngestionPoolConfig,
    ) -> Result<Self, LaneError> {
        let mut producers: Vec<_> = producers.into_iter().collect();
        if producers.is_empty() {
            return Err(LaneError::Empty);
        }
        producers.sort_by_key(|producer| producer.pool_id());
        let first = producers[0];
        let format = FilterFormat::new(first.format().seed(), first.format().bucket_count())?;
        let mut manifest_lanes = [None; CKF_LANE_COUNT];
        let mut lanes_by_pool = HashMap::with_capacity(producers.len());
        let mut lanes = Vec::with_capacity(producers.len());
        for (lane, producer) in producers.into_iter().enumerate() {
            if lane == CKF_LANE_COUNT {
                return Err(LaneError::Manifest(format!(
                    "more than {CKF_LANE_COUNT} pools were supplied"
                )));
            }
            manifest_lanes[lane] = Some(producer.pool_id());
            lanes_by_pool.insert(producer.pool_id(), lane);
            lanes.push(LaneState {
                pool_id: producer.pool_id(),
                producer: None,
                lease: None,
                next_assignment_epoch: 0,
                availability: LaneAvailability::Unavailable(LaneUnavailableReason::Absent),
                last_sequence: None,
                last_update: None,
                snapshot: None,
            });
        }
        let manifest = GlobalCkfManifest::new(
            consumer,
            first.indexer_domain(),
            first.format(),
            manifest_lanes,
        )
        .map_err(|error| LaneError::Manifest(error.to_string()))?;
        let indexer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default())
            .map_err(|error| LaneError::Manifest(error.to_string()))?;
        let ingestion = Arc::new(GlobalCkfIngestionPool::new(indexer, ingestion_config)?);
        Ok(Self {
            consumer,
            format,
            freshness_timeout,
            lanes_by_pool,
            lanes: Mutex::new(lanes),
            ingestion,
        })
    }

    pub fn indexer(&self) -> &GlobalCkfIndexer {
        self.ingestion.indexer()
    }

    pub fn ingestion_pool(&self) -> &Arc<GlobalCkfIngestionPool> {
        &self.ingestion
    }

    pub fn assign(&self, producer: ProducerIdentity, now: Instant) -> Result<LaneLease, LaneError> {
        let lane = self.lane_for_pool(producer.pool_id())?;
        let mut lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
        let state = &mut lanes[lane];
        let epoch = state
            .next_assignment_epoch
            .checked_add(1)
            .ok_or(LaneError::AssignmentEpochExhausted { lane: lane as u8 })?;
        let lease = LaneLease::new(self.consumer, lane as u8, epoch);
        self.ingestion.assign(producer, lease)?;
        state.producer = Some(producer);
        state.lease = Some(lease);
        state.next_assignment_epoch = epoch;
        state.availability = LaneAvailability::Unavailable(LaneUnavailableReason::Rebuilding);
        state.last_sequence = None;
        state.last_update = Some(now);
        state.snapshot = Some(SnapshotAssembly::new(self.format));
        Ok(lease)
    }

    pub fn ingest(
        &self,
        lease: LaneLease,
        update: &FilterUpdate,
        now: Instant,
    ) -> Result<IngestedUpdate, LaneError> {
        let result = self.ingest_inner(lease, update, now);
        if result.is_err() {
            let _ = self.retire(lease, LaneUnavailableReason::Ingestion);
        }
        result
    }

    fn ingest_inner(
        &self,
        lease: LaneLease,
        update: &FilterUpdate,
        now: Instant,
    ) -> Result<IngestedUpdate, LaneError> {
        let lane = usize::from(lease.physical_lane());
        let mut lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
        let state = lanes.get_mut(lane).ok_or(LaneError::StaleLease)?;
        if state.lease != Some(lease) {
            return Err(LaneError::StaleLease);
        }
        let producer = state.producer.ok_or(LaneError::StaleLease)?;
        let outcome = match FilterUpdateKind::try_from(update.kind) {
            Ok(FilterUpdateKind::SnapshotChunk) => {
                let frame = decode(self.format, &update.payload)?;
                validate_dc(producer, frame.header().dc_id)?;
                let assembly = state.snapshot.as_mut().ok_or(LaneError::StaleLease)?;
                match assembly.absorb(&frame)? {
                    None => IngestedUpdate::SnapshotChunk,
                    Some((sequence, images)) => {
                        if sequence != update.sequence {
                            return Err(LaneError::SnapshotSequenceMismatch {
                                frame: sequence,
                                envelope: update.sequence,
                            });
                        }
                        let mut buckets = vec![0; self.format.bucket_count];
                        for image in images {
                            buckets[image.bucket as usize] = image.value;
                        }
                        let snapshot = GlobalCkfSnapshot::new(
                            producer,
                            lease,
                            sequence,
                            buckets.into_boxed_slice(),
                        );
                        match self.ingestion.install_snapshot(snapshot)? {
                            GlobalCkfIngestOutcome::SnapshotInstalled { sequence } => {
                                state.availability = LaneAvailability::Available;
                                state.last_sequence = Some(sequence);
                                IngestedUpdate::SnapshotInstalled { sequence }
                            }
                            GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate { .. } => {
                                IngestedUpdate::SnapshotChunk
                            }
                            _ => {
                                state.availability =
                                    LaneAvailability::Unavailable(LaneUnavailableReason::Ingestion);
                                return Err(LaneError::Ingestion(
                                    GlobalCkfIngestionError::LaneRetired { lane },
                                ));
                            }
                        }
                    }
                }
            }
            Ok(FilterUpdateKind::Delta) => {
                let frame = decode(self.format, &update.payload)?;
                validate_dc(producer, frame.header().dc_id)?;
                let ImagesFrame::Delta {
                    header,
                    base_epoch,
                    images,
                } = frame
                else {
                    return Err(LaneError::UnsupportedUpdateKind(update.kind));
                };
                if base_epoch != update.base_sequence || header.epoch != update.sequence {
                    return Err(LaneError::DeltaSequenceMismatch {
                        base: base_epoch,
                        sequence: header.epoch,
                        envelope_base: update.base_sequence,
                        envelope_sequence: update.sequence,
                    });
                }
                self.ingestion.submit_delta(GlobalCkfDelta::new(
                    producer,
                    lease,
                    base_epoch,
                    header.epoch,
                    images
                        .into_iter()
                        .map(|image| GlobalCkfBucketImage::new(image.bucket as usize, image.value))
                        .collect(),
                ))?;
                state.last_sequence = Some(header.epoch);
                IngestedUpdate::DeltaSubmitted {
                    sequence: header.epoch,
                }
            }
            Ok(FilterUpdateKind::Heartbeat) => IngestedUpdate::Heartbeat {
                sequence: update.sequence,
            },
            Ok(FilterUpdateKind::Unspecified) | Err(_) => {
                return Err(LaneError::UnsupportedUpdateKind(update.kind));
            }
        };
        state.last_update = Some(now);
        Ok(outcome)
    }

    pub fn retire(
        &self,
        lease: LaneLease,
        reason: LaneUnavailableReason,
    ) -> Result<bool, LaneError> {
        let lane = usize::from(lease.physical_lane());
        let mut lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
        let state = lanes.get_mut(lane).ok_or(LaneError::StaleLease)?;
        if state.lease != Some(lease) {
            return Ok(false);
        }
        match self.ingestion.retire_lane(lease) {
            Ok(()) | Err(GlobalCkfIngestionError::LaneRetired { .. }) => {}
            Err(error) => return Err(error.into()),
        }
        state.lease = None;
        state.producer = None;
        state.availability = LaneAvailability::Unavailable(reason);
        state.last_sequence = None;
        state.last_update = None;
        state.snapshot = None;
        Ok(true)
    }

    pub fn retire_stale(&self, now: Instant) -> Result<Vec<PoolId>, LaneError> {
        let expired: Vec<_> = {
            let lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
            lanes
                .iter()
                .filter_map(|state| {
                    let lease = state.lease?;
                    let updated = state.last_update?;
                    (now.saturating_duration_since(updated) >= self.freshness_timeout)
                        .then_some((state.pool_id, lease))
                })
                .collect()
        };
        let mut retired = Vec::with_capacity(expired.len());
        for (pool_id, lease) in expired {
            if self.retire(lease, LaneUnavailableReason::Stale)? {
                retired.push(pool_id);
            }
        }
        Ok(retired)
    }

    pub fn drain_faults(&self) -> Result<Vec<(PoolId, GlobalCkfIngestionFault)>, LaneError> {
        let mut faults = Vec::new();
        while let Some(fault) = self.ingestion.try_recv_fault() {
            match fault {
                GlobalCkfIngestionFault::WorkerFailed { worker } => {
                    let affected: Vec<_> = self
                        .lanes_by_pool
                        .iter()
                        .filter_map(|(pool, lane)| {
                            (self.ingestion.worker_for_lane(*lane) == Some(worker))
                                .then_some((*pool, *lane))
                        })
                        .collect();
                    for (pool, lane) in affected {
                        self.mark_native_fault(lane)?;
                        faults.push((pool, fault));
                    }
                }
                _ => {
                    let lane = fault_lane(fault);
                    let pool = {
                        let lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
                        lanes.get(lane).map(|state| state.pool_id)
                    };
                    if let Some(pool) = pool {
                        self.mark_native_fault(lane)?;
                        faults.push((pool, fault));
                    }
                }
            }
        }
        Ok(faults)
    }

    pub fn statuses(&self, now: Instant) -> Result<Vec<LaneStatus>, LaneError> {
        let lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
        Ok(lanes
            .iter()
            .enumerate()
            .map(|(lane, state)| LaneStatus {
                pool_id: state.pool_id,
                physical_lane: lane as u8,
                lease: state.lease,
                producer: state.producer,
                availability: state.availability,
                last_sequence: state.last_sequence,
                last_update_age: state
                    .last_update
                    .map(|updated| now.saturating_duration_since(updated)),
            })
            .collect())
    }

    fn lane_for_pool(&self, pool_id: PoolId) -> Result<usize, LaneError> {
        self.lanes_by_pool
            .get(&pool_id)
            .copied()
            .ok_or(LaneError::UnknownPool(pool_id))
    }

    fn mark_native_fault(&self, lane: usize) -> Result<(), LaneError> {
        let mut lanes = self.lanes.lock().map_err(|_| LaneError::StatePoisoned)?;
        if let Some(state) = lanes.get_mut(lane) {
            state.lease = None;
            state.producer = None;
            state.availability = LaneAvailability::Unavailable(LaneUnavailableReason::Ingestion);
            state.last_sequence = None;
            state.last_update = None;
            state.snapshot = None;
        }
        Ok(())
    }
}

fn validate_dc(producer: ProducerIdentity, frame_dc: u64) -> Result<(), LaneError> {
    let producer_dc = producer.dc_id().get();
    if frame_dc != producer_dc {
        return Err(LaneError::DcMismatch {
            frame: frame_dc,
            producer: producer_dc,
        });
    }
    Ok(())
}

fn fault_lane(fault: GlobalCkfIngestionFault) -> usize {
    match fault {
        GlobalCkfIngestionFault::Saturated { lane, .. }
        | GlobalCkfIngestionFault::OutstandingImageLimit { lane, .. }
        | GlobalCkfIngestionFault::DirtyToAppliedAgeExceeded { lane, .. }
        | GlobalCkfIngestionFault::LaneDeactivated { lane, .. } => lane,
        GlobalCkfIngestionFault::WorkerFailed { .. } => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::DcCkfFormatIdentity;
    use dynamo_llm::kv_dc_relay::protocol::wire::images::{
        BucketImage, encode_delta, encode_snapshot_chunks,
    };

    use super::*;

    const BUCKETS: usize = 8;
    const FRESHNESS: Duration = Duration::from_secs(5);

    fn domain() -> IndexerDomainId {
        IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([2; 16], IdentitySource::Explicit),
        )
    }

    fn producer(dc: u64, generation: u64) -> ProducerIdentity {
        ProducerIdentity::new(
            PoolId::new(domain(), DcId::new(dc)),
            generation,
            generation,
            DcCkfFormatIdentity::try_new(17, BUCKETS).expect("valid native format"),
        )
    }

    fn lanes(producers: impl IntoIterator<Item = ProducerIdentity>) -> LaneSet {
        LaneSet::new(
            ConsumerInstanceId::new(9),
            producers,
            FRESHNESS,
            GlobalCkfIngestionPoolConfig {
                worker_count: 2,
                queue_capacity: 8,
                control_timeout: Duration::from_secs(1),
                max_outstanding_images_per_lane: None,
                max_dirty_to_applied_age: Duration::from_secs(1),
            },
        )
        .expect("lane set")
    }

    fn update(kind: FilterUpdateKind, base: u64, sequence: u64, payload: Vec<u8>) -> FilterUpdate {
        FilterUpdate {
            base_sequence: base,
            sequence,
            kind: kind as i32,
            payload: payload.into(),
            ..Default::default()
        }
    }

    fn install(lanes: &LaneSet, identity: ProducerIdentity, lease: LaneLease, sequence: u64) {
        let format = FilterFormat::new(identity.format().seed(), BUCKETS).unwrap();
        let mut words = vec![0; BUCKETS];
        words[3] = 42;
        let frames = encode_snapshot_chunks(format, identity.dc_id().get(), sequence, &words)
            .expect("snapshot encoding");
        for frame in frames {
            lanes
                .ingest(
                    lease,
                    &update(FilterUpdateKind::SnapshotChunk, 0, sequence, frame),
                    Instant::now(),
                )
                .expect("snapshot ingestion");
        }
    }

    #[test]
    fn sparse_snapshot_is_expanded_and_makes_lane_ready() {
        let identity = producer(1, 1);
        let lanes = lanes([identity]);
        let lease = lanes.assign(identity, Instant::now()).unwrap();
        install(&lanes, identity, lease, 4);
        assert_eq!(lanes.indexer().ready_lanes(), 1);
        assert_eq!(
            lanes.statuses(Instant::now()).unwrap()[0].availability,
            LaneAvailability::Available
        );
    }

    #[test]
    fn retirement_is_independent_per_lane() {
        let east = producer(1, 1);
        let west = producer(2, 1);
        let lanes = lanes([west, east]);
        let east_lease = lanes.assign(east, Instant::now()).unwrap();
        let west_lease = lanes.assign(west, Instant::now()).unwrap();
        install(&lanes, east, east_lease, 4);
        install(&lanes, west, west_lease, 4);
        assert_eq!(lanes.indexer().ready_lanes(), 0b11);
        lanes
            .retire(east_lease, LaneUnavailableReason::Transport)
            .unwrap();
        assert_eq!(lanes.indexer().ready_lanes(), 0b10);
    }

    #[test]
    fn stale_lease_cannot_retire_ready_replacement() {
        let first = producer(1, 1);
        let lanes = lanes([first]);
        let old = lanes.assign(first, Instant::now()).unwrap();
        lanes.retire(old, LaneUnavailableReason::Transport).unwrap();
        let replacement = producer(1, 2);
        let current = lanes.assign(replacement, Instant::now()).unwrap();
        install(&lanes, replacement, current, 5);
        assert!(!lanes.retire(old, LaneUnavailableReason::Transport).unwrap());
        assert_eq!(lanes.indexer().ready_lanes(), 1);
    }

    #[test]
    fn freshness_timeout_retires_only_expired_lane() {
        let identity = producer(1, 1);
        let lanes = lanes([identity]);
        let assigned = Instant::now();
        lanes.assign(identity, assigned).unwrap();
        assert_eq!(
            lanes.retire_stale(assigned + FRESHNESS).unwrap(),
            vec![identity.pool_id()]
        );
        assert_eq!(
            lanes.statuses(assigned + FRESHNESS).unwrap()[0].availability,
            LaneAvailability::Unavailable(LaneUnavailableReason::Stale)
        );
    }

    #[test]
    fn malformed_frame_fails_closed() {
        let identity = producer(1, 1);
        let lanes = lanes([identity]);
        let lease = lanes.assign(identity, Instant::now()).unwrap();
        install(&lanes, identity, lease, 4);
        let malformed = update(FilterUpdateKind::Delta, 4, 5, vec![1, 2, 3]);
        assert!(matches!(
            lanes.ingest(lease, &malformed, Instant::now()),
            Err(LaneError::Wire(_))
        ));
        assert_eq!(lanes.indexer().ready_lanes(), 0);
    }

    #[test]
    fn native_gap_fault_deactivates_lane() {
        let identity = producer(1, 1);
        let lanes = lanes([identity]);
        let lease = lanes.assign(identity, Instant::now()).unwrap();
        install(&lanes, identity, lease, 4);
        let format = FilterFormat::new(identity.format().seed(), BUCKETS).unwrap();
        let payload = encode_delta(
            format,
            identity.dc_id().get(),
            5,
            6,
            &[BucketImage {
                bucket: 3,
                value: 99,
            }],
        )
        .unwrap();
        lanes
            .ingest(
                lease,
                &update(FilterUpdateKind::Delta, 5, 6, payload),
                Instant::now(),
            )
            .expect("bounded submission succeeds before worker detects gap");

        let deadline = Instant::now() + Duration::from_secs(1);
        let faults = loop {
            let faults = lanes.drain_faults().unwrap();
            if !faults.is_empty() {
                break faults;
            }
            assert!(
                Instant::now() < deadline,
                "native gap fault was not reported"
            );
            thread::yield_now();
        };
        assert_eq!(faults[0].0, identity.pool_id());
        assert_eq!(lanes.indexer().ready_lanes(), 0);
    }
}
