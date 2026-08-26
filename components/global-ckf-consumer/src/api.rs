// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dynamo_kv_router::identity::{IdentitySource, PoolId};
use dynamo_kv_router::protocols::{
    BlockExtraInfo, BlockHashOptions, LocalBlockHash, compute_block_hash_for_seq,
};
use serde::{Deserialize, Serialize};

use crate::contract::{DecisionOutcome, QueryRole, TokenDecisionRequest, TokenDecisionResponse};
use crate::lane::{LaneAvailability, LaneSet, LaneUnavailableReason};
use crate::policy::{
    Freshness, LaneFact, OccupancyFact, PolicyInput, PoolFacts, ReadinessFact, select_pool,
};

#[derive(Clone)]
pub struct PoolMetadata {
    pub pool_id: PoolId,
    pub relay: String,
    pub models: HashSet<String>,
    pub roles: HashSet<i32>,
}

pub struct PublishedGroup {
    pub lanes: Arc<LaneSet>,
    pub pools: Vec<PoolMetadata>,
}

#[derive(Debug, Clone, Default)]
pub struct PoolSignals {
    pub readiness: Vec<ReadinessSignal>,
    pub readiness_updated: Option<Instant>,
    pub load_window_sequence: Option<u64>,
    pub load_observed_ms: Option<u64>,
    pub load_window_ms: Option<u64>,
    pub load_updated: Option<Instant>,
    pub kv_used_blocks: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub kv_observed_ranks: Option<u32>,
    pub kv_expected_ranks: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadinessSignal {
    pub revision: u64,
    pub canonical_model_id: String,
    pub state: i32,
    pub present_roles: Vec<i32>,
    pub missing_roles: Vec<i32>,
}

#[derive(Default)]
struct PublishedState {
    generation: u64,
    groups: Vec<PublishedGroup>,
}

#[derive(Default)]
pub struct Metrics {
    queries: AtomicU64,
    query_errors: AtomicU64,
    catalog_rebuilds: AtomicU64,
    lane_rebuilds: AtomicU64,
    decisions_local: AtomicU64,
    decisions_remote: AtomicU64,
    decisions_none: AtomicU64,
    decision_errors: AtomicU64,
}

#[derive(Clone, Default)]
pub struct AppState {
    published: Arc<RwLock<PublishedState>>,
    signals: Arc<RwLock<HashMap<PoolId, PoolSignals>>>,
    pub metrics: Arc<Metrics>,
}

impl AppState {
    pub fn publish(&self, groups: Vec<PublishedGroup>) -> Result<u64, &'static str> {
        let mut state = self
            .published
            .write()
            .map_err(|_| "published state poisoned")?;
        state.generation = state.generation.saturating_add(1);
        state.groups = groups;
        self.metrics
            .catalog_rebuilds
            .fetch_add(1, Ordering::Relaxed);
        Ok(state.generation)
    }

    pub fn note_lane_rebuild(&self) {
        self.metrics.lane_rebuilds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn maintain(&self, now: Instant) {
        let Ok(state) = self.published.read() else {
            return;
        };
        for group in &state.groups {
            match group.lanes.retire_stale(now) {
                Ok(retired) => {
                    self.metrics
                        .lane_rebuilds
                        .fetch_add(retired.len() as u64, Ordering::Relaxed);
                }
                Err(error) => tracing::warn!(%error, "failed to enforce CKF lane freshness"),
            }
            match group.lanes.drain_faults() {
                Ok(faults) => {
                    for (pool, fault) in &faults {
                        tracing::warn!(pool = %pool, ?fault, "native CKF ingestion fault");
                    }
                    self.metrics
                        .lane_rebuilds
                        .fetch_add(faults.len() as u64, Ordering::Relaxed);
                }
                Err(error) => tracing::warn!(%error, "failed to drain native CKF faults"),
            }
        }
    }

    pub fn retire_relay_except(&self, relay: &str, retained: &HashSet<PoolId>) {
        let Ok(state) = self.published.read() else {
            return;
        };
        for group in &state.groups {
            let Ok(statuses) = group.lanes.statuses(Instant::now()) else {
                continue;
            };
            for metadata in group
                .pools
                .iter()
                .filter(|pool| pool.relay == relay && !retained.contains(&pool.pool_id))
            {
                if let Some(lease) = statuses
                    .iter()
                    .find(|status| status.pool_id == metadata.pool_id)
                    .and_then(|status| status.lease)
                {
                    let _ = group.lanes.retire(lease, LaneUnavailableReason::Transport);
                }
            }
        }
    }

    pub fn replace_signals(&self, signals: HashMap<PoolId, PoolSignals>) {
        if let Ok(mut current) = self.signals.write() {
            *current = signals;
        }
    }

    pub fn ready(&self) -> bool {
        self.published
            .read()
            .map(|state| {
                state.groups.iter().any(|group| {
                    group.lanes.statuses(Instant::now()).is_ok_and(|statuses| {
                        statuses
                            .iter()
                            .any(|status| status.availability == LaneAvailability::Available)
                    })
                })
            })
            .unwrap_or(false)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixMatchesRequest {
    model: String,
    role: QueryRole,
    local_block_hashes: Vec<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenPrefixMatchesRequest {
    model: String,
    role: QueryRole,
    token_ids: Vec<u32>,
    block_size: u32,
    #[serde(default)]
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    #[serde(default)]
    lora_name: Option<String>,
    #[serde(default)]
    cache_namespace: Option<String>,
    #[serde(default)]
    is_eagle: Option<bool>,
}

#[derive(Debug, Serialize)]
struct PrefixMatchesResponse {
    generation: u64,
    model: String,
    role: QueryRoleResponse,
    query_blocks: usize,
    pools: Vec<PoolFact>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum QueryRoleResponse {
    Aggregated,
    Prefill,
    Decode,
}

impl From<QueryRole> for QueryRoleResponse {
    fn from(value: QueryRole) -> Self {
        match value {
            QueryRole::Aggregated => Self::Aggregated,
            QueryRole::Prefill => Self::Prefill,
            QueryRole::Decode => Self::Decode,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct PoolFact {
    #[serde(skip)]
    pub(crate) native_pool_id: PoolId,
    pool_id: String,
    indexer_domain: IndexerDomainFact,
    pub(crate) relay: String,
    pub(crate) dc_id: u64,
    physical_lane: u8,
    availability: &'static str,
    unavailable_reason: Option<&'static str>,
    prefix_depth_blocks: Option<u32>,
    producer_incarnation: Option<u64>,
    pub(crate) layout_generation: Option<u64>,
    installed_sequence: Option<u64>,
    freshness_age_ms: Option<u64>,
    readiness: Vec<ReadinessSignal>,
    readiness_age_ms: Option<u64>,
    load_window_sequence: Option<u64>,
    load_observed_ms: Option<u64>,
    load_window_ms: Option<u64>,
    load_age_ms: Option<u64>,
    kv_used_blocks: Option<u64>,
    total_kv_blocks: Option<u64>,
    kv_observed_ranks: Option<u32>,
    kv_expected_ranks: Option<u32>,
}

#[derive(Debug, Serialize)]
struct IndexerDomainFact {
    cache_semantics_digest: String,
    cache_semantics_source: &'static str,
    routing_scope_digest: String,
    routing_scope_source: &'static str,
}

pub fn api_router(state: AppState, max_query_blocks: usize) -> Router {
    Router::new()
        .route("/readyz", get(ready))
        .route("/v1/status", get(status))
        .route(
            "/v1/prefix-matches",
            post(move |state, request| prefix_matches(state, request, max_query_blocks)),
        )
        .route(
            "/v1/prefix-matches/tokens",
            post(move |state, request| token_prefix_matches(state, request, max_query_blocks)),
        )
        .route(
            "/v1/decisions/tokens",
            post(move |state, request| token_decision(state, request, max_query_blocks)),
        )
        .with_state(state)
}

pub fn system_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn ready(State(state): State<AppState>) -> StatusCode {
    if state.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn status(State(state): State<AppState>) -> Response {
    match collect_facts(&state, None, QueryRole::Aggregated, &[]) {
        Ok((generation, pools)) => Json(serde_json::json!({
            "generation": generation,
            "ready": pools.iter().any(|pool| pool.availability == "available"),
            "pools": pools,
        }))
        .into_response(),
        Err(error) => error.into_response(),
    }
}

async fn prefix_matches(
    State(state): State<AppState>,
    Json(request): Json<PrefixMatchesRequest>,
    max_query_blocks: usize,
) -> Response {
    state.metrics.queries.fetch_add(1, Ordering::Relaxed);
    if request.model.trim().is_empty() {
        return ApiError::bad_request("model must not be empty").into_response();
    }
    if request.local_block_hashes.len() > max_query_blocks {
        return ApiError::payload_too_large("local_block_hashes exceeds the configured limit")
            .into_response();
    }
    match collect_facts(
        &state,
        Some(&request.model),
        request.role,
        &request.local_block_hashes,
    ) {
        Ok((_generation, pools)) if pools.is_empty() => {
            state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
            ApiError::not_found("no catalog pool matches model and role").into_response()
        }
        Ok((generation, pools)) => Json(PrefixMatchesResponse {
            generation,
            model: request.model,
            role: request.role.into(),
            query_blocks: request.local_block_hashes.len(),
            pools,
        })
        .into_response(),
        Err(error) => {
            state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
            error.into_response()
        }
    }
}

async fn token_prefix_matches(
    State(state): State<AppState>,
    Json(request): Json<TokenPrefixMatchesRequest>,
    max_query_blocks: usize,
) -> Response {
    state.metrics.queries.fetch_add(1, Ordering::Relaxed);
    if request.model.trim().is_empty() {
        return ApiError::bad_request("model must not be empty").into_response();
    }
    let hashes = match native_hashes(&request, max_query_blocks) {
        Ok(hashes) => hashes,
        Err(error) => {
            state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
            return error.into_response();
        }
    };
    match collect_facts(&state, Some(&request.model), request.role, &hashes) {
        Ok((_generation, pools)) if pools.is_empty() => {
            state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
            ApiError::not_found("no catalog pool matches model and role").into_response()
        }
        Ok((generation, pools)) => Json(PrefixMatchesResponse {
            generation,
            model: request.model,
            role: request.role.into(),
            query_blocks: hashes.len(),
            pools,
        })
        .into_response(),
        Err(error) => {
            state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
            error.into_response()
        }
    }
}

async fn token_decision(
    State(state): State<AppState>,
    Json(request): Json<TokenDecisionRequest>,
    max_query_blocks: usize,
) -> Response {
    state.metrics.queries.fetch_add(1, Ordering::Relaxed);
    if request.model.trim().is_empty() {
        return decision_error(&state, ApiError::bad_request("model must not be empty"));
    }
    let query = TokenPrefixMatchesRequest {
        model: request.model.clone(),
        role: request.role,
        token_ids: request.token_ids.clone(),
        block_size: request.block_size,
        block_mm_infos: request.block_mm_infos.clone(),
        lora_name: request.lora_name.clone(),
        cache_namespace: request.cache_namespace.clone(),
        is_eagle: request.is_eagle,
    };
    let hashes = match native_hashes(&query, max_query_blocks) {
        Ok(hashes) => hashes,
        Err(error) => return decision_error(&state, error),
    };
    let evaluated = match evaluate_decision(
        &state,
        &request.model,
        request.role,
        &hashes,
        request.block_size,
        request.local_dc,
        request.stable_tie_key,
        request.readiness_max_age_ms,
        request.load_max_age_ms,
    ) {
        Ok(evaluated) => evaluated,
        Err(error) => return decision_error(&state, error),
    };
    let (
        selected_pool_id,
        selected_dc,
        selected_region,
        outcome,
        matched_prefix_blocks,
        uncached_prefill_tokens,
    ) = match evaluated.decision.selected {
        Some(selected) => {
            let dc = selected.pool_id.dc_id().get();
            let selected_region = evaluated
                .pools
                .iter()
                .find(|pool| pool.native_pool_id == selected.pool_id)
                .map(|pool| pool.relay.clone());
            (
                Some(selected.pool_id.to_string()),
                Some(dc),
                selected_region,
                if dc == request.local_dc {
                    DecisionOutcome::Local
                } else {
                    DecisionOutcome::Remote
                },
                Some(selected.matched_prefix_blocks),
                Some(selected.uncached_prefill_tokens),
            )
        }
        None => (None, None, None, DecisionOutcome::None, None, None),
    };
    Json(TokenDecisionResponse {
        generation: evaluated.generation,
        selected_pool_id,
        selected_dc,
        selected_region,
        outcome,
        matched_prefix_blocks,
        uncached_prefill_tokens,
    })
    .into_response()
}

pub(crate) struct EvaluatedDecision {
    pub(crate) generation: u64,
    pub(crate) decision: crate::policy::PolicyDecision,
    pub(crate) pools: Vec<PoolFact>,
}

/// Evaluate the exact routing policy over the current published facts.
/// The single evaluation path for both the decision API and the dispatcher,
/// so their outcomes and decision metrics can never diverge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_decision(
    state: &AppState,
    model: &str,
    role: QueryRole,
    hashes: &[u64],
    block_size: u32,
    local_dc: u64,
    stable_tie_key: u64,
    readiness_max_age_ms: u64,
    load_max_age_ms: u64,
) -> Result<EvaluatedDecision, ApiError> {
    let (generation, pools) = collect_facts(state, Some(model), role, hashes)?;
    if pools.is_empty() {
        return Err(ApiError::not_found(
            "no catalog pool matches model and role",
        ));
    }
    let input = PolicyInput {
        local_dc: dynamo_kv_router::identity::DcId::new(local_dc),
        query_block_count: hashes.len() as u64,
        native_block_size_tokens: block_size as u64,
        stable_tie_key,
    };
    let candidates = pools.iter().map(|pool| PoolFacts {
        pool_id: pool.native_pool_id,
        lane: if pool.availability == "available" && pool.prefix_depth_blocks.is_some() {
            LaneFact::Available
        } else {
            LaneFact::Unavailable
        },
        matched_prefix_blocks: u64::from(pool.prefix_depth_blocks.unwrap_or(0)),
        readiness: pool.readiness_age_ms.map(|age| ReadinessFact {
            ready: pool
                .readiness
                .iter()
                .any(|signal| signal.canonical_model_id == model && signal.state == 2),
            freshness: Freshness {
                age: std::time::Duration::from_millis(age),
                maximum_age: std::time::Duration::from_millis(readiness_max_age_ms),
            },
        }),
        occupancy: match (
            pool.kv_used_blocks,
            pool.total_kv_blocks,
            pool.kv_observed_ranks,
            pool.kv_expected_ranks,
            pool.load_age_ms,
        ) {
            (
                Some(used_blocks),
                Some(total_blocks),
                Some(observed_ranks),
                Some(expected_ranks),
                Some(age),
            ) => Some(OccupancyFact {
                used_blocks,
                total_blocks,
                observed_ranks,
                expected_ranks,
                freshness: Freshness {
                    age: std::time::Duration::from_millis(age),
                    maximum_age: std::time::Duration::from_millis(load_max_age_ms),
                },
            }),
            _ => None,
        },
    });
    let decision =
        select_pool(input, candidates).map_err(|_| ApiError::internal("routing policy failed"))?;
    match decision.selected {
        Some(selected) if selected.pool_id.dc_id().get() == local_dc => {
            state
                .metrics
                .decisions_local
                .fetch_add(1, Ordering::Relaxed);
        }
        Some(_) => {
            state
                .metrics
                .decisions_remote
                .fetch_add(1, Ordering::Relaxed);
        }
        None => {
            state.metrics.decisions_none.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(EvaluatedDecision {
        generation,
        decision,
        pools,
    })
}

fn decision_error(state: &AppState, error: ApiError) -> Response {
    state.metrics.query_errors.fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .decision_errors
        .fetch_add(1, Ordering::Relaxed);
    error.into_response()
}

fn native_hashes(
    request: &TokenPrefixMatchesRequest,
    max_query_blocks: usize,
) -> Result<Vec<u64>, ApiError> {
    if request.block_size == 0 {
        return Err(ApiError::bad_request("block_size must be positive"));
    }
    if request
        .lora_name
        .as_ref()
        .is_some_and(|value| value.is_empty())
    {
        return Err(ApiError::bad_request("lora_name must not be empty"));
    }
    if request
        .cache_namespace
        .as_ref()
        .is_some_and(|value| value.is_empty())
    {
        return Err(ApiError::bad_request("cache_namespace must not be empty"));
    }
    let stride = request.block_size as usize;
    let is_eagle = request.is_eagle.unwrap_or(false);
    let block_count = if is_eagle {
        request.token_ids.len().saturating_sub(1) / stride
    } else {
        request.token_ids.len() / stride
    };
    if block_count == 0 && !request.token_ids.is_empty() {
        return Err(ApiError::bad_request(
            "token_ids does not contain one complete native block",
        ));
    }
    if block_count > max_query_blocks {
        return Err(ApiError::payload_too_large(
            "token_ids exceeds the configured block limit",
        ));
    }
    if let Some(infos) = &request.block_mm_infos {
        if infos.len() != block_count {
            return Err(ApiError::bad_request(
                "block_mm_infos must contain exactly one entry per complete block",
            ));
        }
        let window_size = stride + usize::from(is_eagle);
        if infos.iter().flatten().any(|info| {
            info.mm_objects.iter().any(|object| {
                object
                    .offsets
                    .iter()
                    .any(|&(start, end)| start >= end || end > window_size)
            })
        }) {
            return Err(ApiError::bad_request(
                "multimodal offsets must be nonempty ranges inside their block",
            ));
        }
    }
    Ok(compute_block_hash_for_seq(
        &request.token_ids,
        request.block_size,
        BlockHashOptions {
            block_mm_infos: request.block_mm_infos.as_deref(),
            lora_name: request.lora_name.as_deref(),
            cache_namespace: request.cache_namespace.as_deref(),
            is_eagle: request.is_eagle,
        },
    )
    .into_iter()
    .map(|hash| hash.0)
    .collect())
}

fn collect_facts(
    app: &AppState,
    model: Option<&str>,
    role: QueryRole,
    hashes: &[u64],
) -> Result<(u64, Vec<PoolFact>), ApiError> {
    let state = app
        .published
        .read()
        .map_err(|_| ApiError::internal("published state unavailable"))?;
    let signals = app
        .signals
        .read()
        .map_err(|_| ApiError::internal("relay signals unavailable"))?;
    let sequence: Vec<_> = hashes.iter().copied().map(LocalBlockHash).collect();
    let mut facts = Vec::new();
    let now = Instant::now();
    for group in &state.groups {
        let statuses = group
            .lanes
            .statuses(Instant::now())
            .map_err(|_| ApiError::internal("lane state unavailable"))?;
        let matches = group.lanes.indexer().find_prefix_matches(&sequence).ok();
        for metadata in &group.pools {
            if !pool_matches(metadata, model, role) {
                continue;
            }
            let Some(status) = statuses
                .iter()
                .find(|status| status.pool_id == metadata.pool_id)
            else {
                continue;
            };
            let depth = matches.as_ref().and_then(|result| {
                result
                    .lanes()
                    .iter()
                    .flatten()
                    .find(|lane| lane.pool_id() == metadata.pool_id)
                    .map(|lane| lane.prefix_depth())
            });
            let (availability, unavailable_reason) = availability(status.availability);
            let signal = signals.get(&metadata.pool_id).cloned().unwrap_or_default();
            facts.push(PoolFact {
                native_pool_id: metadata.pool_id,
                pool_id: metadata.pool_id.to_string(),
                indexer_domain: indexer_domain_fact(metadata.pool_id),
                relay: metadata.relay.clone(),
                dc_id: metadata.pool_id.dc_id().get(),
                physical_lane: status.physical_lane,
                availability,
                unavailable_reason,
                prefix_depth_blocks: depth,
                producer_incarnation: status
                    .producer
                    .map(|producer| producer.producer_incarnation()),
                layout_generation: status.producer.map(|producer| producer.layout_generation()),
                installed_sequence: status.last_sequence,
                freshness_age_ms: status
                    .last_update_age
                    .map(|age| age.as_millis().min(u128::from(u64::MAX)) as u64),
                readiness: signal.readiness,
                readiness_age_ms: signal
                    .readiness_updated
                    .map(|updated| duration_millis(now.saturating_duration_since(updated))),
                load_window_sequence: signal.load_window_sequence,
                load_observed_ms: signal.load_observed_ms,
                load_window_ms: signal.load_window_ms,
                load_age_ms: signal
                    .load_updated
                    .map(|updated| duration_millis(now.saturating_duration_since(updated))),
                kv_used_blocks: signal.kv_used_blocks,
                total_kv_blocks: signal.total_kv_blocks,
                kv_observed_ranks: signal.kv_observed_ranks,
                kv_expected_ranks: signal.kv_expected_ranks,
            });
        }
    }
    Ok((state.generation, facts))
}

fn indexer_domain_fact(pool_id: PoolId) -> IndexerDomainFact {
    let domain = pool_id.indexer_domain();
    IndexerDomainFact {
        cache_semantics_digest: digest_hex(domain.cache_semantics().digest()),
        cache_semantics_source: identity_source(domain.cache_semantics().source()),
        routing_scope_digest: digest_hex(domain.routing_scope().digest()),
        routing_scope_source: identity_source(domain.routing_scope().source()),
    }
}

fn digest_hex(digest: [u8; 16]) -> String {
    use std::fmt::Write;

    let mut output = String::with_capacity(32);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

const fn identity_source(source: IdentitySource) -> &'static str {
    match source {
        IdentitySource::DefaultDerived => "default_derived",
        IdentitySource::Explicit => "explicit",
    }
}

fn duration_millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn pool_matches(metadata: &PoolMetadata, model: Option<&str>, role: QueryRole) -> bool {
    let model_matches = model.is_none_or(|model| metadata.models.contains(model));
    let role_matches = model.is_none()
        || match role {
            QueryRole::Aggregated => metadata.roles.contains(&4),
            QueryRole::Prefill => metadata.roles.contains(&1),
            QueryRole::Decode => metadata.roles.contains(&2),
        };
    model_matches && role_matches
}

fn availability(value: LaneAvailability) -> (&'static str, Option<&'static str>) {
    match value {
        LaneAvailability::Available => ("available", None),
        LaneAvailability::Unavailable(reason) => (
            "unavailable",
            Some(match reason {
                LaneUnavailableReason::Absent => "absent",
                LaneUnavailableReason::Rebuilding => "rebuilding",
                LaneUnavailableReason::Stale => "stale",
                LaneUnavailableReason::Transport => "transport",
                LaneUnavailableReason::Ingestion => "ingestion",
            }),
        ),
    }
}

async fn metrics(State(state): State<AppState>) -> String {
    let ready = u8::from(state.ready());
    format!(
        concat!(
            "# TYPE global_ckf_consumer_ready gauge\n",
            "global_ckf_consumer_ready {}\n",
            "# TYPE global_ckf_consumer_queries_total counter\n",
            "global_ckf_consumer_queries_total {}\n",
            "# TYPE global_ckf_consumer_query_errors_total counter\n",
            "global_ckf_consumer_query_errors_total {}\n",
            "# TYPE global_ckf_consumer_catalog_rebuilds_total counter\n",
            "global_ckf_consumer_catalog_rebuilds_total {}\n",
            "# TYPE global_ckf_consumer_lane_rebuilds_total counter\n",
            "global_ckf_consumer_lane_rebuilds_total {}\n",
            "# TYPE global_ckf_consumer_decisions_total counter\n",
            "global_ckf_consumer_decisions_total{{outcome=\"local\"}} {}\n",
            "global_ckf_consumer_decisions_total{{outcome=\"remote\"}} {}\n",
            "global_ckf_consumer_decisions_total{{outcome=\"none\"}} {}\n",
            "# TYPE global_ckf_consumer_decision_errors_total counter\n",
            "global_ckf_consumer_decision_errors_total {}\n"
        ),
        ready,
        state.metrics.queries.load(Ordering::Relaxed),
        state.metrics.query_errors.load(Ordering::Relaxed),
        state.metrics.catalog_rebuilds.load(Ordering::Relaxed),
        state.metrics.lane_rebuilds.load(Ordering::Relaxed),
        state.metrics.decisions_local.load(Ordering::Relaxed),
        state.metrics.decisions_remote.load(Ordering::Relaxed),
        state.metrics.decisions_none.load(Ordering::Relaxed),
        state.metrics.decision_errors.load(Ordering::Relaxed),
    )
}

#[derive(Debug)]
pub(crate) struct ApiError(pub(crate) StatusCode, pub(crate) &'static str);

impl ApiError {
    fn bad_request(message: &'static str) -> Self {
        Self(StatusCode::BAD_REQUEST, message)
    }
    fn payload_too_large(message: &'static str) -> Self {
        Self(StatusCode::PAYLOAD_TOO_LARGE, message)
    }
    fn not_found(message: &'static str) -> Self {
        Self(StatusCode::NOT_FOUND, message)
    }
    fn internal(message: &'static str) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::BlockMmObjectInfo;

    fn request() -> TokenPrefixMatchesRequest {
        TokenPrefixMatchesRequest {
            model: "model".into(),
            role: QueryRole::Aggregated,
            token_ids: (1..=9).collect(),
            block_size: 4,
            block_mm_infos: Some(vec![
                Some(BlockExtraInfo {
                    mm_objects: vec![BlockMmObjectInfo {
                        mm_hash: 42,
                        offsets: vec![(1, 3)],
                    }],
                }),
                None,
            ]),
            lora_name: Some("adapter".into()),
            cache_namespace: Some("tenant".into()),
            is_eagle: Some(true),
        }
    }

    #[test]
    fn token_contract_matches_native_hashes_with_all_options() {
        let request = request();
        let expected = compute_block_hash_for_seq(
            &request.token_ids,
            request.block_size,
            BlockHashOptions {
                block_mm_infos: request.block_mm_infos.as_deref(),
                lora_name: request.lora_name.as_deref(),
                cache_namespace: request.cache_namespace.as_deref(),
                is_eagle: request.is_eagle,
            },
        );

        assert_eq!(
            native_hashes(&request, 8).unwrap(),
            expected.into_iter().map(|hash| hash.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn token_contract_rejects_ambiguous_hash_input() {
        let value = serde_json::json!({
            "model": "model",
            "role": "aggregated",
            "token_ids": [1, 2, 3, 4],
            "block_size": 4,
            "local_block_hashes": [1]
        });
        assert!(serde_json::from_value::<TokenPrefixMatchesRequest>(value).is_err());
    }

    #[test]
    fn token_contract_rejects_malformed_native_options() {
        let mut malformed = request();
        malformed.block_mm_infos = Some(vec![]);
        assert_eq!(
            native_hashes(&malformed, 8).unwrap_err().0,
            StatusCode::BAD_REQUEST
        );

        malformed.block_mm_infos = None;
        malformed.lora_name = Some(String::new());
        assert_eq!(
            native_hashes(&malformed, 8).unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
    }
}
