// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::component::Instance;
use dynamo_runtime::instances::list_all_instances;
use parking_lot::RwLock;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub(super) struct HealthInstanceReconciler {
    last_nonempty: RwLock<Vec<Instance>>,
    retaining_last_known: AtomicBool,
}

pub(super) struct ReconciledHealthInstances {
    pub instances: Vec<Instance>,
    pub retained_last_known: bool,
    pub entered_retention: bool,
}

impl HealthInstanceReconciler {
    pub(super) fn reconcile(
        &self,
        observed: Option<Vec<Instance>>,
        has_ready_model: bool,
    ) -> ReconciledHealthInstances {
        if let Some(instances) = observed.as_ref().filter(|instances| !instances.is_empty()) {
            *self.last_nonempty.write() = instances.clone();
            self.retaining_last_known.store(false, Ordering::Relaxed);
            return ReconciledHealthInstances {
                instances: instances.clone(),
                retained_last_known: false,
                entered_retention: false,
            };
        }

        // KubeDiscoveryClient::list() may transiently observe its metadata watch before the
        // daemon has published a snapshot. The request router is fed by a long-lived model
        // discovery stream, so a serving-ready model is independent evidence that an empty
        // one-shot read is not yet authoritative.
        if has_ready_model {
            let last_nonempty = self.last_nonempty.read().clone();
            if !last_nonempty.is_empty() {
                let entered_retention = !self.retaining_last_known.swap(true, Ordering::Relaxed);
                return ReconciledHealthInstances {
                    instances: last_nonempty,
                    retained_last_known: true,
                    entered_retention,
                };
            }
        }

        self.last_nonempty.write().clear();
        self.retaining_last_known.store(false, Ordering::Relaxed);
        ReconciledHealthInstances {
            instances: Vec::new(),
            retained_last_known: false,
            entered_retention: false,
        }
    }
}

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let health_path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &health_path)];

    let router = Router::new()
        .route(&health_path, get(health_handler))
        .with_state(state);

    (docs, router)
}

pub fn live_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let live_path = path.unwrap_or_else(|| "/live".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &live_path)];

    let router = Router::new()
        .route(&live_path, get(live_handler))
        .with_state(state);

    (docs, router)
}

async fn live_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    // Check if the http service is being cancelled/shutdown
    if state.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "shutting_down",
                "message": "Service is shutting down"
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "live",
            "message": "Service is live"
        })),
    )
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    if !state.is_ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "stage": state.service_stage().to_string(),
                "message": "Service is not ready"
            })),
        );
    }

    let observed_instances = match list_all_instances(state.discovery()).await {
        Ok(instances) => Some(instances),
        Err(err) => {
            tracing::warn!(%err, "Failed to fetch instances from discovery");
            None
        }
    };
    let observed_empty = observed_instances.as_ref().is_some_and(Vec::is_empty);
    let instances = state.reconcile_health_instances(observed_instances);
    if instances.entered_retention {
        tracing::warn!(
            observed_empty,
            retained_last_known = instances.retained_last_known,
            retained_instances = instances.instances.len(),
            "Discovery snapshot conflicts with serving-ready model state; retaining last-known health instances"
        );
    }
    let mut endpoints: Vec<String> = instances
        .instances
        .iter()
        .map(|instance| instance.endpoint_id().as_url())
        .collect();
    endpoints.sort();
    endpoints.dedup();
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "endpoints": endpoints,
            "instances": instances.instances
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::{Instance, TransportType};

    fn instance(id: u64) -> Instance {
        Instance {
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            namespace: "test".to_string(),
            instance_id: id,
            transport: TransportType::Tcp("127.0.0.1:8000".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    #[test]
    fn retains_last_nonempty_snapshot_when_routing_still_has_a_ready_model() {
        let reconciler = HealthInstanceReconciler::default();
        let expected = vec![instance(1)];

        let initial = reconciler.reconcile(Some(expected.clone()), true);
        assert_eq!(initial.instances, expected);
        assert!(!initial.retained_last_known);
        assert!(!initial.entered_retention);

        let reconciled = reconciler.reconcile(Some(Vec::new()), true);
        assert_eq!(reconciled.instances, expected);
        assert!(reconciled.retained_last_known);
        assert!(reconciled.entered_retention);

        let repeated = reconciler.reconcile(Some(Vec::new()), true);
        assert!(repeated.retained_last_known);
        assert!(!repeated.entered_retention);
    }

    #[test]
    fn retains_last_nonempty_snapshot_when_discovery_read_fails() {
        let reconciler = HealthInstanceReconciler::default();
        let expected = vec![instance(1)];
        reconciler.reconcile(Some(expected.clone()), true);

        let reconciled = reconciler.reconcile(None, true);
        assert_eq!(reconciled.instances, expected);
        assert!(reconciled.retained_last_known);
        assert!(reconciled.entered_retention);
    }

    #[test]
    fn publishes_empty_after_routing_confirms_no_ready_model() {
        let reconciler = HealthInstanceReconciler::default();
        reconciler.reconcile(Some(vec![instance(1)]), true);

        let reconciled = reconciler.reconcile(Some(Vec::new()), false);
        assert!(reconciled.instances.is_empty());
        assert!(!reconciled.retained_last_known);
        assert!(!reconciled.entered_retention);

        let later_conflict = reconciler.reconcile(None, true);
        assert!(later_conflict.instances.is_empty());
        assert!(!later_conflict.retained_last_known);
        assert!(!later_conflict.entered_retention);
    }
}
