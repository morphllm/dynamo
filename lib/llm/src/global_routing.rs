// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoritative client contract for the global CKF consumer.

use std::sync::{LazyLock, OnceLock};
use std::time::Duration;

use prometheus::{Histogram, HistogramOpts, IntCounterVec, Registry};
use serde::{Deserialize, Serialize};

use crate::protocols::TokenIdType;

static CLIENT: OnceLock<Result<Option<Client>, String>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenDecisionRequest {
    pub model: String,
    pub role: QueryRole,
    pub token_ids: Vec<TokenIdType>,
    pub block_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_mm_infos: Option<Vec<Option<dynamo_kv_router::protocols::BlockExtraInfo>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_namespace: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_eagle: Option<bool>,
    pub local_dc: u64,
    pub stable_tie_key: u64,
    pub readiness_max_age_ms: u64,
    pub load_max_age_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryRole {
    Aggregated,
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenDecisionResponse {
    pub generation: u64,
    pub selected_pool_id: Option<String>,
    pub selected_dc: Option<u64>,
    pub selected_region: Option<String>,
    pub outcome: DecisionOutcome,
    pub matched_prefix_blocks: Option<u64>,
    pub uncached_prefill_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcome {
    Local,
    Remote,
    None,
}

#[derive(Debug, thiserror::Error)]
pub enum GlobalRoutingError {
    #[error("global routing does not yet support multimodal requests")]
    MultimodalUnsupported,
    #[error("global routing block size is invalid")]
    InvalidBlockSize,
    #[error("global routing decision request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("global routing consumer returned no eligible pool")]
    NoEligiblePool,
    #[error("global routing consumer returned an incomplete decision")]
    IncompleteDecision,
    #[error("global routing is misconfigured: {0}")]
    Config(String),
}

#[derive(Clone)]
struct Client {
    http: reqwest::Client,
    endpoint: String,
    local_dc: u64,
    readiness_max_age_ms: u64,
    load_max_age_ms: u64,
}

impl Client {
    fn from_env() -> Result<Option<Self>, String> {
        let Some(base) = std::env::var("DYN_GLOBAL_ROUTER_CONSUMER_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Ok(None);
        };
        let parse = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<u64>())
                .transpose()
                .map(|value| value.unwrap_or(default))
                .map_err(|error| format!("{name} must be an unsigned integer: {error}"))
        };
        let timeout = Duration::from_millis(parse("DYN_GLOBAL_ROUTER_DECISION_TIMEOUT_MS", 1_000)?);
        Ok(Some(Self {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|error| format!("failed to build decision HTTP client: {error}"))?,
            endpoint: format!("{}/v1/decisions/tokens", base.trim_end_matches('/')),
            local_dc: std::env::var("DYN_GLOBAL_ROUTER_LOCAL_DC_ID")
                .map_err(|_| "DYN_GLOBAL_ROUTER_LOCAL_DC_ID is required".to_owned())?
                .parse()
                .map_err(|error| format!("DYN_GLOBAL_ROUTER_LOCAL_DC_ID must be a u64: {error}"))?,
            readiness_max_age_ms: parse("DYN_GLOBAL_ROUTER_READINESS_MAX_AGE_MS", 45_000)?,
            load_max_age_ms: parse("DYN_GLOBAL_ROUTER_LOAD_MAX_AGE_MS", 15_000)?,
        }))
    }

    async fn decide(
        &self,
        request: TokenDecisionRequest,
    ) -> Result<TokenDecisionResponse, GlobalRoutingError> {
        let decision = self
            .http
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<TokenDecisionResponse>()
            .await?;
        match (
            decision.outcome,
            decision.selected_pool_id.as_ref(),
            decision.selected_dc,
            decision.selected_region.as_ref(),
        ) {
            (DecisionOutcome::None, _, _, _) => Err(GlobalRoutingError::NoEligiblePool),
            (_, Some(_), Some(_), Some(_)) => Ok(decision),
            _ => Err(GlobalRoutingError::IncompleteDecision),
        }
    }
}

static DECISIONS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        prometheus::Opts::new(
            "morph_global_routing_authoritative_decisions_total",
            "Authoritative global CKF decision outcomes owned by the frontend.",
        ),
        &["outcome"],
    )
    .expect("static metric options are valid")
});

static DECISION_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(HistogramOpts::new(
        "morph_global_routing_decision_duration_seconds",
        "Latency of authoritative global CKF decision requests.",
    ))
    .expect("static metric options are valid")
});

pub fn ensure_metrics_registered_prometheus(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(DECISIONS.clone()))?;
    registry.register(Box::new(DECISION_LATENCY.clone()))
}

/// Await and own exactly one global decision after native preprocessing.
///
/// When global routing is configured, every unsupported or unavailable input
/// fails closed. Dropping the returned decision would reintroduce shadow
/// semantics, so callers must retain it in request context until dispatch.
pub async fn decide(
    model: &str,
    token_ids: &[TokenIdType],
    block_size: usize,
    request_id: &str,
    has_multimodal_data: bool,
) -> Result<Option<TokenDecisionResponse>, GlobalRoutingError> {
    let client = match CLIENT.get_or_init(Client::from_env) {
        Ok(Some(client)) => client,
        Ok(None) => return Ok(None),
        Err(error) => return Err(GlobalRoutingError::Config(error.clone())),
    };
    if has_multimodal_data {
        DECISIONS.with_label_values(&["error"]).inc();
        return Err(GlobalRoutingError::MultimodalUnsupported);
    }
    let block_size = u32::try_from(block_size).map_err(|_| GlobalRoutingError::InvalidBlockSize)?;
    if block_size == 0 {
        return Err(GlobalRoutingError::InvalidBlockSize);
    }
    let request = TokenDecisionRequest {
        model: model.to_owned(),
        role: QueryRole::Aggregated,
        token_ids: token_ids.to_vec(),
        block_size,
        block_mm_infos: None,
        lora_name: None,
        cache_namespace: None,
        is_eagle: None,
        local_dc: client.local_dc,
        stable_tie_key: stable_hash(request_id.as_bytes()),
        readiness_max_age_ms: client.readiness_max_age_ms,
        load_max_age_ms: client.load_max_age_ms,
    };
    let timer = DECISION_LATENCY.start_timer();
    let result = client.decide(request).await;
    timer.observe_duration();
    match &result {
        Ok(decision) => DECISIONS
            .with_label_values(&[match decision.outcome {
                DecisionOutcome::Local => "local",
                DecisionOutcome::Remote => "remote",
                DecisionOutcome::None => "none",
            }])
            .inc(),
        Err(_) => DECISIONS.with_label_values(&["error"]).inc(),
    }
    result.map(Some)
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_contract_roundtrips_and_rejects_unknown_fields() {
        let json = r#"{"generation":42,"selected_pool_id":"pool","selected_dc":2,"selected_region":"us-west1-financial","outcome":"remote","matched_prefix_blocks":3,"uncached_prefill_tokens":256}"#;
        let decision: TokenDecisionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(decision.outcome, DecisionOutcome::Remote);
        assert_eq!(decision.selected_dc, Some(2));
        assert!(
            serde_json::from_str::<TokenDecisionResponse>(&format!(
                "{}",
                json.replace('}', ",\"extra\":1}")
            ))
            .is_err()
        );
    }

    #[test]
    fn stable_tie_key_is_repeatable_and_request_specific() {
        assert_eq!(stable_hash(b"request"), stable_hash(b"request"));
        assert_ne!(stable_hash(b"request"), stable_hash(b"other"));
    }
}
