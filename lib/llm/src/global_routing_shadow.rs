// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best effort, bounded observation of native global CKF routing decisions.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{Semaphore, mpsc};

use crate::protocols::TokenIdType;

const QUEUE_CAPACITY: usize = 1024;
const MAX_IN_FLIGHT: usize = 16;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

static OBSERVER: OnceLock<Option<Observer>> = OnceLock::new();

#[derive(Clone)]
struct Observer {
    sender: mpsc::Sender<Observation>,
    config: Arc<Config>,
}

#[derive(Debug, Serialize)]
struct Observation {
    model: String,
    role: &'static str,
    token_ids: Vec<TokenIdType>,
    block_size: u32,
    local_dc: u64,
    stable_tie_key: u64,
    readiness_max_age_ms: u64,
    load_max_age_ms: u64,
}

#[derive(Debug)]
struct Config {
    endpoint: String,
    local_dc: u64,
    readiness_max_age_ms: u64,
    load_max_age_ms: u64,
}

impl Config {
    fn from_env() -> Option<Self> {
        let endpoint = std::env::var("DYN_GLOBAL_ROUTER_SHADOW_URL").ok()?;
        let parse = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<u64>())
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        Some(Self {
            endpoint: format!("{}/v1/decisions/tokens", endpoint.trim_end_matches('/')),
            local_dc: std::env::var("DYN_GLOBAL_ROUTER_LOCAL_DC_ID")
                .ok()?
                .parse()
                .ok()?,
            readiness_max_age_ms: parse("DYN_GLOBAL_ROUTER_READINESS_MAX_AGE_MS", 45_000).ok()?,
            load_max_age_ms: parse("DYN_GLOBAL_ROUTER_LOAD_MAX_AGE_MS", 15_000).ok()?,
        })
    }
}

/// Queue a shadow decision after native tokenization. This never waits for network I/O.
pub fn observe(
    model: &str,
    token_ids: &[TokenIdType],
    block_size: usize,
    request_id: &str,
    has_multimodal_data: bool,
) {
    if block_size == 0 || token_ids.len() < block_size || has_multimodal_data {
        return;
    }
    let Some(observer) = OBSERVER.get_or_init(Observer::from_env) else {
        return;
    };
    let Ok(block_size) = u32::try_from(block_size) else {
        return;
    };
    let _ = observer.sender.try_send(Observation {
        model: model.to_owned(),
        role: "aggregated",
        token_ids: token_ids.to_vec(),
        block_size,
        local_dc: observer.config.local_dc,
        stable_tie_key: stable_hash(request_id.as_bytes()),
        readiness_max_age_ms: observer.config.readiness_max_age_ms,
        load_max_age_ms: observer.config.load_max_age_ms,
    });
}

impl Observer {
    fn from_env() -> Option<Self> {
        let config = Arc::new(Config::from_env()?);
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .ok()?;
        let (sender, mut receiver) = mpsc::channel::<Observation>(QUEUE_CAPACITY);
        let permits = Arc::new(Semaphore::new(MAX_IN_FLIGHT));
        let worker_config = Arc::clone(&config);
        tokio::spawn(async move {
            while let Some(observation) = receiver.recv().await {
                let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
                    break;
                };
                let client = client.clone();
                let endpoint = worker_config.endpoint.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    if let Err(error) = client
                        .post(endpoint)
                        .json(&observation)
                        .send()
                        .await
                        .and_then(reqwest::Response::error_for_status)
                    {
                        tracing::debug!(%error, "global CKF shadow observation failed");
                    }
                });
            }
        });
        Some(Self { sender, config })
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::stable_hash;

    #[test]
    fn stable_hash_is_repeatable_and_request_specific() {
        assert_eq!(stable_hash(b"request"), stable_hash(b"request"));
        assert_ne!(stable_hash(b"request"), stable_hash(b"other"));
    }
}
