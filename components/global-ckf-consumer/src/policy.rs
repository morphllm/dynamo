// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_kv_router::identity::{DcId, IdentitySource, PoolId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Freshness {
    pub age: Duration,
    pub maximum_age: Duration,
}

impl Freshness {
    pub const fn is_fresh(self) -> bool {
        self.age.as_nanos() <= self.maximum_age.as_nanos()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadinessFact {
    pub ready: bool,
    pub freshness: Freshness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyFact {
    pub used_blocks: u64,
    pub total_blocks: u64,
    pub observed_ranks: u32,
    pub expected_ranks: u32,
    pub freshness: Freshness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneFact {
    Absent,
    Unavailable,
    Available,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolFacts {
    pub pool_id: PoolId,
    pub lane: LaneFact,
    pub matched_prefix_blocks: u64,
    pub readiness: Option<ReadinessFact>,
    pub occupancy: Option<OccupancyFact>,
    pub prefill_tps_per_rank: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyInput {
    pub local_dc: DcId,
    pub query_token_count: u64,
    pub native_block_size_tokens: u64,
    pub stable_tie_key: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IneligibleReason {
    LaneAbsent,
    LaneUnavailable,
    ReadinessMissing,
    ReadinessStale,
    NotReady,
    OccupancyMissing,
    OccupancyStale,
    OccupancyIncomplete,
    PrefixExceedsQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EligiblePool {
    pub pool_id: PoolId,
    pub matched_prefix_blocks: u64,
    pub uncached_prefill_tokens: u64,
    pub used_blocks: u64,
    pub total_blocks: u64,
    pub prefill_capacity_tps: u64,
    pub local: bool,
    pub stable_rank: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IneligiblePool {
    pub pool_id: PoolId,
    pub reason: IneligibleReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecision {
    pub selected: Option<EligiblePool>,
    pub eligible: Vec<EligiblePool>,
    pub ineligible: Vec<IneligiblePool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PolicyError {
    #[error("native block size must be greater than zero")]
    ZeroBlockSize,
    #[error("prefill token arithmetic overflowed for pool {pool_id}")]
    TokenArithmeticOverflow { pool_id: PoolId },
}

/// Select an exact global pool without changing the facts API or forwarding a request.
///
/// Eligibility is fail closed. Eligible pools are ordered by projected uncached prefill time,
/// then occupancy, local DC, and a stable request keyed rank. Input order is never a tie breaker.
pub fn select_pool(
    input: PolicyInput,
    pools: impl IntoIterator<Item = PoolFacts>,
) -> Result<PolicyDecision, PolicyError> {
    if input.native_block_size_tokens == 0 {
        return Err(PolicyError::ZeroBlockSize);
    }

    let mut eligible = Vec::new();
    let mut ineligible = Vec::new();
    for facts in pools {
        match evaluate_pool(input, facts)? {
            Ok(candidate) => eligible.push(candidate),
            Err(reason) => ineligible.push(IneligiblePool {
                pool_id: facts.pool_id,
                reason,
            }),
        }
    }
    eligible.sort_by(candidate_order);
    ineligible.sort_by_key(|pool| (pool.reason, pool.pool_id));
    Ok(PolicyDecision {
        selected: eligible.first().copied(),
        eligible,
        ineligible,
    })
}

fn evaluate_pool(
    input: PolicyInput,
    facts: PoolFacts,
) -> Result<Result<EligiblePool, IneligibleReason>, PolicyError> {
    let reason = match facts.lane {
        LaneFact::Absent => Some(IneligibleReason::LaneAbsent),
        LaneFact::Unavailable => Some(IneligibleReason::LaneUnavailable),
        LaneFact::Available => match facts.readiness {
            None => Some(IneligibleReason::ReadinessMissing),
            Some(readiness) if !readiness.freshness.is_fresh() => {
                Some(IneligibleReason::ReadinessStale)
            }
            Some(readiness) if !readiness.ready => Some(IneligibleReason::NotReady),
            Some(_) => match facts.occupancy {
                None => Some(IneligibleReason::OccupancyMissing),
                Some(occupancy) if !occupancy.freshness.is_fresh() => {
                    Some(IneligibleReason::OccupancyStale)
                }
                Some(occupancy)
                    if occupancy.total_blocks == 0
                        || occupancy.expected_ranks == 0
                        || occupancy.observed_ranks != occupancy.expected_ranks =>
                {
                    Some(IneligibleReason::OccupancyIncomplete)
                }
                Some(_)
                    if facts.matched_prefix_blocks
                        > input.query_token_count / input.native_block_size_tokens =>
                {
                    Some(IneligibleReason::PrefixExceedsQuery)
                }
                Some(_) => None,
            },
        },
    };
    if let Some(reason) = reason {
        return Ok(Err(reason));
    }

    let occupancy = facts.occupancy.expect("eligible pool has occupancy");
    let prefill_capacity_tps = facts
        .prefill_tps_per_rank
        .checked_mul(u64::from(occupancy.expected_ranks))
        .ok_or(PolicyError::TokenArithmeticOverflow {
            pool_id: facts.pool_id,
        })?;
    let cached_tokens = facts
        .matched_prefix_blocks
        .checked_mul(input.native_block_size_tokens)
        .ok_or(PolicyError::TokenArithmeticOverflow {
            pool_id: facts.pool_id,
        })?;
    let uncached_prefill_tokens = input.query_token_count - cached_tokens;
    Ok(Ok(EligiblePool {
        pool_id: facts.pool_id,
        matched_prefix_blocks: facts.matched_prefix_blocks,
        uncached_prefill_tokens,
        used_blocks: occupancy.used_blocks,
        total_blocks: occupancy.total_blocks,
        prefill_capacity_tps,
        local: facts.pool_id.dc_id() == input.local_dc,
        stable_rank: stable_pool_rank(input.stable_tie_key, facts.pool_id),
    }))
}

fn candidate_order(left: &EligiblePool, right: &EligiblePool) -> std::cmp::Ordering {
    (u128::from(left.uncached_prefill_tokens) * u128::from(right.prefill_capacity_tps))
        .cmp(&(u128::from(right.uncached_prefill_tokens) * u128::from(left.prefill_capacity_tps)))
        .then_with(|| {
            (u128::from(left.used_blocks) * u128::from(right.total_blocks))
                .cmp(&(u128::from(right.used_blocks) * u128::from(left.total_blocks)))
        })
        .then_with(|| (!left.local).cmp(&(!right.local)))
        .then_with(|| left.stable_rank.cmp(&right.stable_rank))
        .then_with(|| left.pool_id.cmp(&right.pool_id))
}

fn stable_pool_rank(key: u64, pool_id: PoolId) -> u64 {
    let domain = pool_id.indexer_domain();
    let mut hash = 0xcbf2_9ce4_8422_2325 ^ key;
    for byte in domain
        .cache_semantics()
        .digest()
        .into_iter()
        .chain([identity_source(domain.cache_semantics().source())])
        .chain(domain.routing_scope().digest())
        .chain([identity_source(domain.routing_scope().source())])
        .chain(pool_id.dc_id().get().to_be_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

const fn identity_source(source: IdentitySource) -> u8 {
    match source {
        IdentitySource::DefaultDerived => 0,
        IdentitySource::Explicit => 1,
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{CacheSemanticsId, IndexerDomainId, RoutingScopeId};

    use super::*;

    const FRESH: Freshness = Freshness {
        age: Duration::from_secs(1),
        maximum_age: Duration::from_secs(5),
    };
    const STALE: Freshness = Freshness {
        age: Duration::from_secs(6),
        maximum_age: Duration::from_secs(5),
    };

    fn pool(dc: u64) -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(dc),
        )
    }

    fn input() -> PolicyInput {
        PolicyInput {
            local_dc: DcId::new(1),
            query_token_count: 2_600,
            native_block_size_tokens: 256,
            stable_tie_key: 42,
        }
    }

    fn facts(dc: u64, prefix: u64, used: u64) -> PoolFacts {
        PoolFacts {
            pool_id: pool(dc),
            lane: LaneFact::Available,
            matched_prefix_blocks: prefix,
            readiness: Some(ReadinessFact {
                ready: true,
                freshness: FRESH,
            }),
            occupancy: Some(OccupancyFact {
                used_blocks: used,
                total_blocks: 100,
                observed_ranks: 1,
                expected_ranks: 1,
                freshness: FRESH,
            }),
            prefill_tps_per_rank: 10_000,
        }
    }

    fn selected(pools: impl IntoIterator<Item = PoolFacts>) -> PoolId {
        select_pool(input(), pools)
            .unwrap()
            .selected
            .expect("eligible pool")
            .pool_id
    }

    #[test]
    fn missing_and_stale_occupancy_fail_closed() {
        let mut missing = facts(1, 10, 0);
        missing.occupancy = None;
        let mut stale = facts(2, 10, 0);
        stale.occupancy.as_mut().unwrap().freshness = STALE;
        let decision = select_pool(input(), [stale, missing]).unwrap();
        assert!(decision.selected.is_none());
        assert_eq!(
            decision
                .ineligible
                .iter()
                .map(|pool| pool.reason)
                .collect::<Vec<_>>(),
            vec![
                IneligibleReason::OccupancyMissing,
                IneligibleReason::OccupancyStale,
            ]
        );
    }

    #[test]
    fn absent_lane_is_never_zero_overlap_fallback() {
        let mut absent = facts(1, 0, 0);
        absent.lane = LaneFact::Absent;
        assert_eq!(selected([absent, facts(2, 0, 0)]), pool(2));
    }

    #[test]
    fn exact_prefix_wins_before_occupancy() {
        assert_eq!(selected([facts(1, 9, 0), facts(2, 10, 99)]), pool(2));
    }

    #[test]
    fn faster_pool_wins_on_projected_prefill_time() {
        let slow = facts(1, 10, 0);
        let mut fast = facts(2, 9, 0);
        fast.prefill_tps_per_rank = 100_000;
        assert_eq!(selected([slow, fast]), pool(2));
    }

    #[test]
    fn capacity_scales_with_live_rank_count() {
        let one_rank = facts(1, 8, 0);
        let mut eight_ranks = facts(2, 0, 0);
        eight_ranks.occupancy.as_mut().unwrap().observed_ranks = 8;
        eight_ranks.occupancy.as_mut().unwrap().expected_ranks = 8;
        assert_eq!(selected([one_rank, eight_ranks]), pool(2));
    }

    #[test]
    fn occupancy_breaks_an_equal_prefix_tie() {
        assert_eq!(selected([facts(1, 10, 60), facts(2, 10, 50)]), pool(2));
    }

    #[test]
    fn local_pool_wins_an_exact_cost_tie() {
        assert_eq!(selected([facts(2, 10, 50), facts(1, 10, 50)]), pool(1));
    }

    #[test]
    fn partial_block_tokens_are_counted_exactly() {
        let mut short = input();
        short.query_token_count = 40;
        let selected = select_pool(short, [facts(1, 0, 0)])
            .unwrap()
            .selected
            .unwrap();
        assert_eq!(selected.matched_prefix_blocks, 0);
        assert_eq!(selected.uncached_prefill_tokens, 40);

        let selected = select_pool(input(), [facts(1, 10, 0)])
            .unwrap()
            .selected
            .unwrap();
        assert_eq!(selected.uncached_prefill_tokens, 40);
    }

    #[test]
    fn stable_remote_tie_is_deterministic_and_input_order_independent() {
        let east = facts(2, 10, 0);
        let west = facts(3, 10, 0);
        let first = selected([east, west]);
        let second = selected([west, east]);
        assert_eq!(first, second);
        let expected = [pool(2), pool(3)]
            .into_iter()
            .min_by_key(|pool| stable_pool_rank(input().stable_tie_key, *pool))
            .unwrap();
        assert_eq!(first, expected);
    }

    #[test]
    fn readiness_states_are_lexicographically_fail_closed() {
        let mut missing = facts(1, 10, 0);
        missing.readiness = None;
        let mut stale = facts(2, 10, 0);
        stale.readiness.as_mut().unwrap().freshness = STALE;
        let mut not_ready = facts(3, 10, 0);
        not_ready.readiness.as_mut().unwrap().ready = false;
        let decision = select_pool(input(), [not_ready, stale, missing]).unwrap();
        assert_eq!(
            decision
                .ineligible
                .iter()
                .map(|pool| pool.reason)
                .collect::<Vec<_>>(),
            vec![
                IneligibleReason::ReadinessMissing,
                IneligibleReason::ReadinessStale,
                IneligibleReason::NotReady,
            ]
        );
    }

    #[test]
    fn invalid_prefix_never_saturates_into_a_route() {
        let invalid = facts(1, 11, 0);
        let decision = select_pool(input(), [invalid]).unwrap();
        assert_eq!(
            decision.ineligible[0].reason,
            IneligibleReason::PrefixExceedsQuery
        );
    }
}
