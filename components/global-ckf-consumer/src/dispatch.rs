// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signed-envelope dispatcher.
//!
//! The consumer owns both halves of the routing decision: it evaluates the
//! policy and it ships the resulting signed internal request to the selected
//! region's pod proxy. `POST /v1/dispatch` accepts a [`SignedEnvelope`],
//! verifies it (signature, version, validity window — never forwarding an
//! envelope it cannot verify), resolves `routing.selected_region` against the
//! configured targets, forwards the identical wire object over HTTPS/mTLS,
//! and relays the response byte stream back unmodified. Dropping the inbound
//! connection drops the upstream request, so client cancellation propagates
//! hop-by-hop through the dispatcher.
//!
//! Addressing is deliberately not enforced here: only the target region's own
//! verifier may assert "this envelope is for me". The dispatcher enforces
//! everything else fail-closed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Context;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use dynamo_llm::global_routing_envelope::{EnvelopeError, EnvelopeVerifier, SignedEnvelope};

use crate::api::AppState;
use crate::config::Config;

/// Response headers relayed from the target pod proxy. Everything else is
/// hop metadata the dispatcher must not impersonate.
const RELAYED_HEADERS: [&str; 5] = [
    "content-type",
    "cache-control",
    "x-accel-buffering",
    "x-request-id",
    "retry-after",
];

pub struct Dispatcher {
    verifier: EnvelopeVerifier,
    targets: HashMap<String, String>,
    client: reqwest::Client,
    state: AppState,
}

impl Dispatcher {
    /// Build the dispatcher when dispatch targets are configured; `None`
    /// leaves the consumer as a pure decision service.
    pub fn from_config(config: &Config, state: AppState) -> anyhow::Result<Option<Arc<Self>>> {
        if config.dispatch_targets.is_empty() {
            return Ok(None);
        }
        let key_path = config
            .dispatch_envelope_public_key
            .as_ref()
            .context("dispatch requires --dispatch-envelope-public-key")?;
        let key_id = config
            .dispatch_envelope_key_id
            .as_ref()
            .context("dispatch requires --dispatch-envelope-key-id")?;
        let pem = std::fs::read_to_string(key_path)
            .with_context(|| format!("cannot read {}", key_path.display()))?;
        let verifier = EnvelopeVerifier::from_public_key_pem(key_id.clone(), &pem)
            .context("invalid dispatch envelope public key")?;

        let mut builder = reqwest::Client::builder()
            .use_rustls_tls()
            .connect_timeout(std::time::Duration::from_secs(
                config.dispatch_connect_timeout_seconds,
            ));
        if let (Some(cert), Some(key)) = (&config.dispatch_client_cert, &config.dispatch_client_key)
        {
            let mut identity = std::fs::read(cert)
                .with_context(|| format!("cannot read {}", cert.display()))?;
            identity.extend(
                std::fs::read(key).with_context(|| format!("cannot read {}", key.display()))?,
            );
            builder = builder.identity(
                reqwest::Identity::from_pem(&identity)
                    .context("invalid dispatch client certificate or key")?,
            );
        }
        if let Some(ca) = &config.dispatch_ca {
            let bundle =
                std::fs::read(ca).with_context(|| format!("cannot read {}", ca.display()))?;
            for certificate in reqwest::Certificate::from_pem_bundle(&bundle)
                .context("invalid dispatch CA bundle")?
            {
                builder = builder.add_root_certificate(certificate);
            }
        }
        Ok(Some(Arc::new(Self {
            verifier,
            targets: config
                .dispatch_targets
                .iter()
                .map(|target| (target.region.clone(), target.url.clone()))
                .collect(),
            client: builder.build().context("cannot build dispatch client")?,
            state,
        })))
    }

    #[cfg(test)]
    fn for_tests(
        verifier: EnvelopeVerifier,
        targets: HashMap<String, String>,
        state: AppState,
    ) -> Arc<Self> {
        Arc::new(Self {
            verifier,
            targets,
            client: reqwest::Client::new(),
            state,
        })
    }
}

pub fn dispatch_router(dispatcher: Arc<Dispatcher>) -> Router {
    Router::new()
        .route("/v1/dispatch", post(dispatch))
        .with_state(dispatcher)
}

async fn dispatch(
    State(dispatcher): State<Arc<Dispatcher>>,
    Json(wire): Json<SignedEnvelope>,
) -> Response {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let envelope = match dispatcher.verifier.verify_addressed(&wire, now) {
        Ok(envelope) => envelope,
        Err(error) => {
            dispatcher.state.metrics.note_dispatch_reject();
            let status = match error {
                EnvelopeError::Malformed | EnvelopeError::UnsupportedVersion => {
                    StatusCode::BAD_REQUEST
                }
                EnvelopeError::BadSignature
                | EnvelopeError::UnknownKey
                | EnvelopeError::Expired
                | EnvelopeError::WrongRegion { .. } => StatusCode::FORBIDDEN,
            };
            return reject(status, error.to_string());
        }
    };
    let Some(url) = dispatcher.targets.get(&envelope.routing.selected_region) else {
        dispatcher.state.metrics.note_dispatch_reject();
        return reject(
            StatusCode::NOT_FOUND,
            format!(
                "no dispatch target for region {:?}",
                envelope.routing.selected_region
            ),
        );
    };
    match dispatcher.client.post(url).json(&wire).send().await {
        Ok(upstream) => {
            dispatcher.state.metrics.note_dispatch();
            relay(upstream)
        }
        Err(error) => {
            dispatcher.state.metrics.note_dispatch_failure();
            tracing::warn!(
                %error,
                region = %envelope.routing.selected_region,
                request_id = %envelope.request_id,
                "dispatch to selected region failed"
            );
            reject(
                StatusCode::BAD_GATEWAY,
                "dispatch to the selected region failed".to_string(),
            )
        }
    }
}

/// Relay the target's status, allow-listed headers, and body byte stream.
fn relay(upstream: reqwest::Response) -> Response {
    let mut builder = Response::builder().status(upstream.status().as_u16());
    for name in RELAYED_HEADERS {
        if let Some(value) = upstream.headers().get(name) {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|error| {
            tracing::error!(%error, "failed to assemble relayed response");
            StatusCode::BAD_GATEWAY.into_response()
        })
}

fn reject(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::global_routing_envelope::{
        Account, ENVELOPE_VERSION, Envelope, EnvelopeSigner, Request as EnvelopeRequest, Routing,
        prompt_token_digest,
    };

    fn signer() -> EnvelopeSigner {
        EnvelopeSigner::from_key_bytes("test-key", &[7u8; 32])
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// A currently-valid envelope; tests mutate the window to expire it.
    fn envelope(selected_region: &str) -> Envelope {
        Envelope {
            version: ENVELOPE_VERSION,
            request_id: "req_1".into(),
            decision_id: "decision_1".into(),
            issuer: "consumer-test".into(),
            audience: format!("dsv4flash-{selected_region}"),
            issued_at: now_unix().saturating_sub(1),
            expires_at: now_unix() + 30,
            nonce: "nonce_1".into(),
            account: Account {
                api_key_id: "key_1".into(),
                user_id: "user_1".into(),
                org_id: None,
                billing_tier: "usage".into(),
                service_tier: "default".into(),
                zero_data_retention: false,
                lifetime_requests: None,
                service_tier_explicit: false,
            },
            request: EnvelopeRequest {
                endpoint: "/v1/chat/completions".into(),
                canonical_model: "morph-dsv4flash".into(),
                normalized_body: serde_json::json!({"model": "morph-dsv4flash"}),
                body_sha256: "sha256:0".into(),
                token_ids_sha256: prompt_token_digest(&[1, 2, 3]),
            },
            routing: Routing {
                selected_region: selected_region.into(),
                selected_pool: "pool_1".into(),
                policy_version: "v1".into(),
                ckf_generation: 7,
            },
        }
    }

    async fn spawn_target(
        response: &'static str,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<SignedEnvelope>) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let app = Router::new().route(
            "/global",
            post(move |Json(wire): Json<SignedEnvelope>| {
                let sender = sender.clone();
                async move {
                    let _ = sender.send(wire);
                    (
                        [("content-type", "text/event-stream"), ("x-request-id", "req_1")],
                        response,
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}/global"), receiver)
    }

    async fn call(
        dispatcher: Arc<Dispatcher>,
        wire: &SignedEnvelope,
    ) -> (StatusCode, axum::http::HeaderMap, String) {
        use tower::ServiceExt;

        let response = dispatch_router(dispatcher)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/dispatch")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(wire).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let body = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, headers, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn verified_envelopes_are_forwarded_verbatim_and_streamed_back() {
        let signer = signer();
        let (url, mut received) = spawn_target("data: {\"ok\":true}\n\ndata: [DONE]\n\n").await;
        let dispatcher = Dispatcher::for_tests(
            signer.verifier(),
            HashMap::from([("us-east5-financial".to_string(), url)]),
            AppState::default(),
        );
        let wire = signer.sign(&envelope("us-east5-financial")).unwrap();

        let (status, headers, body) = call(dispatcher, &wire).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["content-type"], "text/event-stream");
        assert_eq!(headers["x-request-id"], "req_1");
        assert!(body.contains("[DONE]"));
        // The target received the identical signed wire object.
        assert_eq!(received.recv().await.unwrap(), wire);
    }

    #[tokio::test]
    async fn unverifiable_and_unroutable_envelopes_are_rejected() {
        let signer = signer();
        let dispatcher = Dispatcher::for_tests(
            signer.verifier(),
            HashMap::from([(
                "us-east5-financial".to_string(),
                "http://127.0.0.1:9/never".to_string(),
            )]),
            AppState::default(),
        );

        // Foreign signature.
        let foreign = EnvelopeSigner::from_key_bytes("test-key", &[9u8; 32])
            .sign(&envelope("us-east5-financial"))
            .unwrap();
        let (status, _, _) = call(dispatcher.clone(), &foreign).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        // Expired.
        let mut stale = envelope("us-east5-financial");
        stale.issued_at = now_unix().saturating_sub(120);
        stale.expires_at = now_unix().saturating_sub(60);
        let expired = signer.sign(&stale).unwrap();
        let (status, _, _) = call(dispatcher.clone(), &expired).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        // Valid but addressed to a region with no target.
        let unroutable = signer.sign(&envelope("nowhere")).unwrap();
        let (status, _, body) = call(dispatcher.clone(), &unroutable).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("nowhere"));

        // Valid and routable, but the target is unreachable.
        let unreachable = signer.sign(&envelope("us-east5-financial")).unwrap();
        let (status, _, _) = call(dispatcher, &unreachable).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
    }
}
