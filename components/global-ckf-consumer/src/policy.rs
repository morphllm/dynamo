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
pub struct LoadFact {
    pub active_prefill_tokens: u64,
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
    pub load: Option<LoadFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyInput {
    pub local_dc: DcId,
    pub query_block_count: u64,
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
    LoadMissing,
    LoadStale,
    PrefixExceedsQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EligiblePool {
    pub pool_id: PoolId,
    pub matched_prefix_blocks: u64,
    pub uncached_prefill_tokens: u64,
    pub active_prefill_tokens: u64,
    pub total_prefill_tokens: u64,
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
/// Eligibility is fail closed. Eligible pools are ordered by total prefill work, then local DC,
/// then a stable request keyed rank. Input order is never a tie breaker.
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
    eligible.sort_by_key(candidate_order);
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
            Some(_) => match facts.load {
                None => Some(IneligibleReason::LoadMissing),
                Some(load) if !load.freshness.is_fresh() => Some(IneligibleReason::LoadStale),
                Some(_) if facts.matched_prefix_blocks > input.query_block_count => {
                    Some(IneligibleReason::PrefixExceedsQuery)
                }
                Some(_) => None,
            },
        },
    };
    if let Some(reason) = reason {
        return Ok(Err(reason));
    }

    let load = facts.load.expect("eligible pool has load");
    let uncached_blocks = input.query_block_count - facts.matched_prefix_blocks;
    let uncached_prefill_tokens = uncached_blocks
        .checked_mul(input.native_block_size_tokens)
        .ok_or(PolicyError::TokenArithmeticOverflow {
            pool_id: facts.pool_id,
        })?;
    let total_prefill_tokens = uncached_prefill_tokens
        .checked_add(load.active_prefill_tokens)
        .ok_or(PolicyError::TokenArithmeticOverflow {
            pool_id: facts.pool_id,
        })?;
    Ok(Ok(EligiblePool {
        pool_id: facts.pool_id,
        matched_prefix_blocks: facts.matched_prefix_blocks,
        uncached_prefill_tokens,
        active_prefill_tokens: load.active_prefill_tokens,
        total_prefill_tokens,
        local: facts.pool_id.dc_id() == input.local_dc,
        stable_rank: stable_pool_rank(input.stable_tie_key, facts.pool_id),
    }))
}

fn candidate_order(candidate: &EligiblePool) -> (u64, bool, u64, PoolId) {
    (
        candidate.total_prefill_tokens,
        !candidate.local,
        candidate.stable_rank,
        candidate.pool_id,
    )
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
            query_block_count: 10,
            native_block_size_tokens: 256,
            stable_tie_key: 42,
        }
    }

    fn facts(dc: u64, prefix: u64, active: u64) -> PoolFacts {
        PoolFacts {
            pool_id: pool(dc),
            lane: LaneFact::Available,
            matched_prefix_blocks: prefix,
            readiness: Some(ReadinessFact {
                ready: true,
                freshness: FRESH,
            }),
            load: Some(LoadFact {
                active_prefill_tokens: active,
                freshness: FRESH,
            }),
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
    fn missing_and_stale_load_fail_closed() {
        let mut missing = facts(1, 10, 0);
        missing.load = None;
        let mut stale = facts(2, 10, 0);
        stale.load.as_mut().unwrap().freshness = STALE;
        let decision = select_pool(input(), [stale, missing]).unwrap();
        assert!(decision.selected.is_none());
        assert_eq!(
            decision
                .ineligible
                .iter()
                .map(|pool| pool.reason)
                .collect::<Vec<_>>(),
            vec![IneligibleReason::LoadMissing, IneligibleReason::LoadStale]
        );
    }

    #[test]
    fn absent_lane_is_never_zero_overlap_fallback() {
        let mut absent = facts(1, 0, 0);
        absent.lane = LaneFact::Absent;
        assert_eq!(selected([absent, facts(2, 0, 0)]), pool(2));
    }

    #[test]
    fn exact_prefix_wins_when_its_load_is_less_than_one_uncached_block() {
        assert_eq!(selected([facts(1, 9, 0), facts(2, 10, 255)]), pool(2));
    }

    #[test]
    fn active_prefill_load_can_outweigh_a_warmer_prefix() {
        assert_eq!(selected([facts(1, 10, 600), facts(2, 9, 0)]), pool(2));
    }

    #[test]
    fn local_pool_wins_an_exact_cost_tie() {
        assert_eq!(selected([facts(2, 9, 256), facts(1, 10, 512)]), pool(1));
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
    fn invalid_prefix_and_arithmetic_overflow_never_saturate_into_a_route() {
        let invalid = facts(1, 11, 0);
        let decision = select_pool(input(), [invalid]).unwrap();
        assert_eq!(
            decision.ineligible[0].reason,
            IneligibleReason::PrefixExceedsQuery
        );

        let mut huge = input();
        huge.query_block_count = u64::MAX;
        huge.native_block_size_tokens = 2;
        assert!(matches!(
            select_pool(huge, [facts(1, 0, 0)]),
            Err(PolicyError::TokenArithmeticOverflow { .. })
        ));
    }
}
