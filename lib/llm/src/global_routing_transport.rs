// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoritative WAN dispatch to a selected regional pod proxy.
//!
//! A configured global route owns the request. Dispatch is attempted exactly
//! once against the region named by the signed envelope. There is deliberately
//! no local or alternate-region fallback in this module.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;
use futures::Stream;
use prometheus::{IntCounterVec, Opts, Registry};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::de::{MapAccess, Visitor};
use url::Url;

use crate::global_routing_envelope::{
    ENVELOPE_VERSION, Envelope, SignedEnvelope, SignedRoutingDecision,
};

const INTERNAL_CHAT_PATH: &str = "/internal/v1/chat/completions";
const ENV_CERT_PATH: &str = "DYN_GLOBAL_ROUTER_MTLS_CERT_PATH";
const ENV_KEY_PATH: &str = "DYN_GLOBAL_ROUTER_MTLS_KEY_PATH";
const ENV_CA_PATH: &str = "DYN_GLOBAL_ROUTER_MTLS_CA_PATH";
const ENV_ENDPOINTS: &str = "DYN_GLOBAL_ROUTER_REGIONAL_ENDPOINTS_JSON";

pub type ResponseBody = Pin<Box<dyn Stream<Item = Result<Bytes, DispatchError>> + Send + 'static>>;

pub struct WanResponse {
    pub status: reqwest::StatusCode,
    pub headers: HeaderMap,
    /// The live upstream body. It is never buffered. Dropping it cancels the
    /// HTTP/2 stream and therefore propagates client cancellation upstream.
    pub body: ResponseBody,
}

/// Registry-safe handoff from the preprocessing pipeline to the HTTP source.
/// The mutex is only an ownership cell; response bytes are never buffered.
pub struct WanResponseHandle(Mutex<Option<WanResponse>>);

impl WanResponseHandle {
    pub fn new(response: WanResponse) -> Self {
        Self(Mutex::new(Some(response)))
    }

    pub fn take(&self) -> Option<WanResponse> {
        self.0.lock().ok()?.take()
    }
}

pub const WAN_RESPONSE_CONTEXT_KEY: &str = "global_routing.wan_response";

/// Remove headers that describe the regional HTTP hop rather than the outer
/// client response. `Connection` may nominate additional hop-scoped fields.
pub fn sanitize_response_headers(headers: &mut HeaderMap) {
    let connection_scoped = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in connection_scoped {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        // The outer Hyper response computes framing for its streaming body.
        "content-length",
    ] {
        headers.remove(name);
    }
}

static DISPATCHER: OnceLock<Result<RegionalDispatcher, String>> = OnceLock::new();

static WAN_ATTEMPTS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_wan_attempts_total",
            "Authoritative WAN dispatch attempts by selected region.",
        ),
        &["selected_region"],
    )
    .expect("static metric options are valid")
});

static WAN_OUTCOMES: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_wan_outcomes_total",
            "WAN transport outcomes by selected region and bounded phase.",
        ),
        &["selected_region", "phase", "outcome"],
    )
    .expect("static metric options are valid")
});

static WAN_RESPONSE_STATUS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_wan_response_status_total",
            "Regional pod proxy HTTP response statuses.",
        ),
        &["selected_region", "status"],
    )
    .expect("static metric options are valid")
});

static WAN_BYTES: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_wan_bytes_relayed_total",
            "Response bytes relayed from regional pod proxies.",
        ),
        &["selected_region"],
    )
    .expect("static metric options are valid")
});

pub fn ensure_metrics_registered_prometheus(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(WAN_ATTEMPTS.clone()))?;
    registry.register(Box::new(WAN_OUTCOMES.clone()))?;
    registry.register(Box::new(WAN_RESPONSE_STATUS.clone()))?;
    registry.register(Box::new(WAN_BYTES.clone()))
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DispatchError {
    #[error("global routing WAN transport is misconfigured: {0}")]
    Config(String),
    #[error("signed global routing envelope is malformed")]
    MalformedEnvelope,
    #[error("signed envelope and authoritative decision select different regions")]
    DecisionMismatch,
    #[error("no regional pod proxy endpoint is configured for {0}")]
    UnknownRegion(String),
    #[error("regional pod proxy request failed before a response: {0}")]
    BeforeResponse(String),
    #[error("regional pod proxy response stream failed: {0}")]
    Midstream(String),
}

#[derive(Clone)]
pub struct RegionalDispatcher {
    transport: Arc<dyn RegionalTransport>,
}

impl RegionalDispatcher {
    pub fn from_env() -> Result<Self, DispatchError> {
        let config = TransportConfig::from_env()?;
        Ok(Self {
            transport: Arc::new(Http2Transport::new(config)?),
        })
    }

    #[cfg(test)]
    fn with_transport(transport: Arc<dyn RegionalTransport>) -> Self {
        Self { transport }
    }

    pub async fn dispatch(
        &self,
        signed: &SignedRoutingDecision,
    ) -> Result<WanResponse, DispatchError> {
        let payload = decode_payload(&signed.envelope)?;
        let decision_region = signed
            .decision
            .selected_region
            .as_deref()
            .ok_or(DispatchError::DecisionMismatch)?;
        if decision_region != payload.routing.selected_region {
            return Err(DispatchError::DecisionMismatch);
        }
        self.transport
            .send(
                &payload.routing.selected_region,
                &payload.request_id,
                &signed.envelope,
            )
            .await
    }
}

/// The only non-WAN result means global routing is absent. Once configured,
/// callers receive a WAN response or an error and must never call the local
/// pipeline's `next.generate`.
pub enum DispatchOwnership {
    GlobalRoutingDisabled,
    Wan(WanResponse),
}

pub async fn dispatch_authoritatively(
    dispatcher: Option<&RegionalDispatcher>,
    signed: Option<&SignedRoutingDecision>,
) -> Result<DispatchOwnership, DispatchError> {
    match dispatcher {
        None => Ok(DispatchOwnership::GlobalRoutingDisabled),
        Some(dispatcher) => {
            let signed = signed.ok_or(DispatchError::MalformedEnvelope)?;
            dispatcher
                .dispatch(signed)
                .await
                .map(DispatchOwnership::Wan)
        }
    }
}

/// Dispatch a minted request using the process-wide persistent HTTP/2 client.
/// This is called from the preprocessor before its local `next.generate` site.
pub async fn dispatch_signed(signed: &SignedRoutingDecision) -> Result<WanResponse, DispatchError> {
    let dispatcher = DISPATCHER
        .get_or_init(|| RegionalDispatcher::from_env().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| DispatchError::Config(error.clone()))?;
    dispatcher.dispatch(signed).await
}

#[async_trait]
trait RegionalTransport: Send + Sync {
    async fn send(
        &self,
        region: &str,
        request_id: &str,
        envelope: &SignedEnvelope,
    ) -> Result<WanResponse, DispatchError>;
}

struct Http2Transport {
    client: reqwest::Client,
    endpoints: BTreeMap<String, Url>,
}

impl Http2Transport {
    fn new(config: TransportConfig) -> Result<Self, DispatchError> {
        let mut identity_pem =
            std::fs::read(&config.cert_path).map_err(|error| config_error(ENV_CERT_PATH, error))?;
        identity_pem.push(b'\n');
        identity_pem.extend(
            std::fs::read(&config.key_path).map_err(|error| config_error(ENV_KEY_PATH, error))?,
        );
        let identity = reqwest::Identity::from_pem(&identity_pem)
            .map_err(|error| DispatchError::Config(format!("invalid mTLS identity: {error}")))?;
        let ca_pem =
            std::fs::read(&config.ca_path).map_err(|error| config_error(ENV_CA_PATH, error))?;
        let ca = reqwest::Certificate::from_pem(&ca_pem)
            .map_err(|error| DispatchError::Config(format!("invalid mTLS CA: {error}")))?;
        let client = reqwest::Client::builder()
            .identity(identity)
            .add_root_certificate(ca)
            .https_only(true)
            .http2_prior_knowledge()
            .http2_adaptive_window(true)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| {
                DispatchError::Config(format!("failed to build regional HTTP/2 client: {error}"))
            })?;
        Ok(Self {
            client,
            endpoints: config.endpoints,
        })
    }
}

#[async_trait]
impl RegionalTransport for Http2Transport {
    async fn send(
        &self,
        region: &str,
        request_id: &str,
        envelope: &SignedEnvelope,
    ) -> Result<WanResponse, DispatchError> {
        let base = self
            .endpoints
            .get(region)
            .ok_or_else(|| DispatchError::UnknownRegion(region.to_owned()))?;
        let endpoint = base.join(INTERNAL_CHAT_PATH).map_err(|error| {
            DispatchError::Config(format!("invalid endpoint for region {region}: {error}"))
        })?;
        let request_id =
            HeaderValue::from_str(request_id).map_err(|_| DispatchError::MalformedEnvelope)?;

        WAN_ATTEMPTS.with_label_values(&[region]).inc();

        // One send, one selected region. reqwest does not retry requests.
        let response = self
            .client
            .post(endpoint)
            .header(CONTENT_TYPE, "application/json")
            .header("x-request-id", request_id)
            .header("x-morph-envelope-version", ENVELOPE_VERSION.to_string())
            .json(envelope)
            .send()
            .await
            .map_err(|error| {
                WAN_OUTCOMES
                    .with_label_values(&[region, "before_response", "failure"])
                    .inc();
                DispatchError::BeforeResponse(error.to_string())
            })?;
        let status = response.status();
        let headers = response.headers().clone();
        WAN_RESPONSE_STATUS
            .with_label_values(&[region, &status.as_u16().to_string()])
            .inc();
        WAN_OUTCOMES
            .with_label_values(&[region, "response", "received"])
            .inc();
        let body =
            InstrumentedResponseBody::new(region.to_owned(), Box::pin(response.bytes_stream()));
        Ok(WanResponse {
            status,
            headers,
            body: Box::pin(body),
        })
    }
}

struct InstrumentedResponseBody {
    selected_region: String,
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    terminal: bool,
}

impl InstrumentedResponseBody {
    fn new(
        selected_region: String,
        inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    ) -> Self {
        Self {
            selected_region,
            inner,
            terminal: false,
        }
    }
}

impl Stream for InstrumentedResponseBody {
    type Item = Result<Bytes, DispatchError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                WAN_BYTES
                    .with_label_values(&[&self.selected_region])
                    .inc_by(bytes.len() as u64);
                std::task::Poll::Ready(Some(Ok(bytes)))
            }
            std::task::Poll::Ready(Some(Err(error))) => {
                WAN_OUTCOMES
                    .with_label_values(&[&self.selected_region, "stream", "failure"])
                    .inc();
                self.terminal = true;
                std::task::Poll::Ready(Some(Err(DispatchError::Midstream(error.to_string()))))
            }
            std::task::Poll::Ready(None) => {
                WAN_OUTCOMES
                    .with_label_values(&[&self.selected_region, "stream", "complete"])
                    .inc();
                self.terminal = true;
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for InstrumentedResponseBody {
    fn drop(&mut self) {
        if !self.terminal {
            WAN_OUTCOMES
                .with_label_values(&[&self.selected_region, "stream", "cancelled"])
                .inc();
        }
    }
}

#[derive(Debug)]
struct TransportConfig {
    cert_path: String,
    key_path: String,
    ca_path: String,
    endpoints: BTreeMap<String, Url>,
}

impl TransportConfig {
    fn from_env() -> Result<Self, DispatchError> {
        Self::parse(
            required_env(ENV_CERT_PATH)?,
            required_env(ENV_KEY_PATH)?,
            required_env(ENV_CA_PATH)?,
            &required_env(ENV_ENDPOINTS)?,
        )
    }

    fn parse(
        cert_path: String,
        key_path: String,
        ca_path: String,
        endpoints_json: &str,
    ) -> Result<Self, DispatchError> {
        let RawEndpoints(raw) = serde_json::from_str(endpoints_json).map_err(|error| {
            DispatchError::Config(format!("{ENV_ENDPOINTS} must be a JSON object: {error}"))
        })?;
        if raw.is_empty() {
            return Err(DispatchError::Config(format!(
                "{ENV_ENDPOINTS} must not be empty"
            )));
        }
        let mut endpoints = BTreeMap::new();
        for (region, raw_url) in raw {
            if region.trim().is_empty() || region != region.trim() {
                return Err(DispatchError::Config(
                    "regional endpoint keys must be nonempty exact region names".to_owned(),
                ));
            }
            let url = Url::parse(&raw_url).map_err(|error| {
                DispatchError::Config(format!("invalid endpoint for region {region}: {error}"))
            })?;
            if url.scheme() != "https"
                || url.cannot_be_a_base()
                || url.host_str().is_none()
                || url.query().is_some()
                || url.fragment().is_some()
                || url.path() != "/"
            {
                return Err(DispatchError::Config(format!(
                    "endpoint for region {region} must be an HTTPS base URL without a path, query, or fragment"
                )));
            }
            endpoints.insert(region, url);
        }
        Ok(Self {
            cert_path,
            key_path,
            ca_path,
            endpoints,
        })
    }
}

struct RawEndpoints(Vec<(String, String)>);

impl<'de> serde::Deserialize<'de> for RawEndpoints {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct EndpointsVisitor;
        impl<'de> Visitor<'de> for EndpointsVisitor {
            type Value = RawEndpoints;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object mapping exact region names to HTTPS base URLs")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = Vec::<(String, String)>::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    if entries.iter().any(|(existing, _)| existing == &key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate regional endpoint {key}"
                        )));
                    }
                    entries.push((key, value));
                }
                Ok(RawEndpoints(entries))
            }
        }
        deserializer.deserialize_map(EndpointsVisitor)
    }
}

fn required_env(name: &str) -> Result<String, DispatchError> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| DispatchError::Config(format!("{name} is required")))
}

fn config_error(name: &str, error: std::io::Error) -> DispatchError {
    DispatchError::Config(format!("failed to read {name}: {error}"))
}

fn decode_payload(wire: &SignedEnvelope) -> Result<Envelope, DispatchError> {
    if wire.version != ENVELOPE_VERSION {
        return Err(DispatchError::MalformedEnvelope);
    }
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&wire.payload)
        .map_err(|_| DispatchError::MalformedEnvelope)?;
    serde_json::from_slice(&payload).map_err(|_| DispatchError::MalformedEnvelope)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use futures::{StreamExt, stream};

    use super::*;
    use crate::global_routing::{DecisionOutcome, TokenDecisionResponse};
    use crate::global_routing_envelope::{Account, Request, Routing};

    struct MockTransport {
        calls: AtomicUsize,
        regions: Mutex<Vec<String>>,
        request_ids: Mutex<Vec<String>>,
        result: Mutex<Option<Result<WanResponse, DispatchError>>>,
    }

    #[async_trait]
    impl RegionalTransport for MockTransport {
        async fn send(
            &self,
            region: &str,
            request_id: &str,
            _envelope: &SignedEnvelope,
        ) -> Result<WanResponse, DispatchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.regions.lock().unwrap().push(region.to_owned());
            self.request_ids.lock().unwrap().push(request_id.to_owned());
            self.result.lock().unwrap().take().unwrap()
        }
    }

    fn signed(region: &str) -> SignedRoutingDecision {
        let payload = Envelope {
            version: 1,
            request_id: "req_1".into(),
            decision_id: "decision_1".into(),
            issuer: "global-router".into(),
            audience: format!("dsv4flash-{region}"),
            issued_at: 1,
            expires_at: 2,
            nonce: "nonce_1".into(),
            account: Account {
                api_key_id: "key_1".into(),
                user_id: "user_1".into(),
                org_id: None,
                billing_tier: "pro".into(),
                service_tier: "default".into(),
                zero_data_retention: false,
                lifetime_requests: None,
                service_tier_explicit: false,
            },
            request: Request {
                endpoint: "/v1/chat/completions".into(),
                canonical_model: "morph-dsv4flash".into(),
                normalized_body: serde_json::json!({"model":"morph-dsv4flash"}),
                body_sha256: "body".into(),
                token_ids_sha256: "tokens".into(),
            },
            routing: Routing {
                selected_region: region.into(),
                selected_pool: "pool".into(),
                policy_version: "v1".into(),
                ckf_generation: 1,
            },
        };
        SignedRoutingDecision {
            decision: TokenDecisionResponse {
                generation: 1,
                selected_pool_id: Some("pool".into()),
                selected_dc: Some(2),
                selected_region: Some(region.into()),
                outcome: DecisionOutcome::Remote,
                matched_prefix_blocks: Some(1),
                uncached_prefill_tokens: Some(1),
            },
            envelope: SignedEnvelope {
                version: 1,
                key_id: "key".into(),
                payload: base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .encode(serde_json::to_vec(&payload).unwrap()),
                signature: "signature".into(),
            },
        }
    }

    fn response(body: ResponseBody) -> WanResponse {
        WanResponse {
            status: reqwest::StatusCode::OK,
            headers: HeaderMap::new(),
            body,
        }
    }

    fn mock(result: Result<WanResponse, DispatchError>) -> Arc<MockTransport> {
        Arc::new(MockTransport {
            calls: AtomicUsize::new(0),
            regions: Mutex::new(Vec::new()),
            request_ids: Mutex::new(Vec::new()),
            result: Mutex::new(Some(result)),
        })
    }

    #[tokio::test]
    async fn dispatches_only_to_selected_region() {
        let transport = mock(Ok(response(Box::pin(stream::empty()))));
        let dispatcher = RegionalDispatcher::with_transport(transport.clone());
        dispatcher
            .dispatch(&signed("us-west1-financial"))
            .await
            .unwrap();
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
        assert_eq!(&*transport.regions.lock().unwrap(), &["us-west1-financial"]);
        assert_eq!(&*transport.request_ids.lock().unwrap(), &["req_1"]);
    }

    #[tokio::test]
    async fn rejects_a_decision_envelope_region_mismatch_before_network_io() {
        let transport = mock(Ok(response(Box::pin(stream::empty()))));
        let dispatcher = RegionalDispatcher::with_transport(transport.clone());
        let mut signed = signed("us-east5-financial");
        signed.decision.selected_region = Some("us-west1-financial".into());

        assert_eq!(
            dispatcher.dispatch(&signed).await.err(),
            Some(DispatchError::DecisionMismatch)
        );
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn pre_response_failure_is_not_retried() {
        let transport = mock(Err(DispatchError::BeforeResponse("connect".into())));
        let dispatcher = RegionalDispatcher::with_transport(transport.clone());
        assert_eq!(
            dispatcher
                .dispatch(&signed("us-east5-financial"))
                .await
                .err(),
            Some(DispatchError::BeforeResponse("connect".into()))
        );
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    struct DropGuard(Arc<AtomicBool>);
    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn dropping_body_propagates_cancellation_without_buffering() {
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(dropped.clone());
        let body = async_stream::stream! {
            let _guard = guard;
            yield Ok(Bytes::from_static(b"data: first\n\n"));
            futures::future::pending::<()>().await;
        };
        let transport = mock(Ok(response(Box::pin(body))));
        let dispatcher = RegionalDispatcher::with_transport(transport);
        let mut response = dispatcher
            .dispatch(&signed("us-east5-financial"))
            .await
            .unwrap();
        assert_eq!(
            response.body.next().await.unwrap().unwrap(),
            "data: first\n\n"
        );
        drop(response);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn midstream_failure_preserves_prior_bytes_and_does_not_retry() {
        let body = stream::iter(vec![
            Ok(Bytes::from_static(b"data: first\n\n")),
            Err(DispatchError::Midstream("reset".into())),
        ]);
        let transport = mock(Ok(response(Box::pin(body))));
        let dispatcher = RegionalDispatcher::with_transport(transport.clone());
        let mut response = dispatcher
            .dispatch(&signed("us-west1-financial"))
            .await
            .unwrap();
        assert_eq!(
            response.body.next().await.unwrap().unwrap(),
            "data: first\n\n"
        );
        assert_eq!(
            response.body.next().await.unwrap().unwrap_err(),
            DispatchError::Midstream("reset".into())
        );
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn configured_dispatch_never_returns_local_ownership() {
        let transport = mock(Ok(response(Box::pin(stream::empty()))));
        let dispatcher = RegionalDispatcher::with_transport(transport);
        let outcome = dispatch_authoritatively(Some(&dispatcher), Some(&signed("east")))
            .await
            .unwrap();
        assert!(matches!(outcome, DispatchOwnership::Wan(_)));

        assert!(
            dispatch_authoritatively(Some(&dispatcher), None)
                .await
                .is_err()
        );
    }

    #[test]
    fn endpoint_contract_is_strict() {
        let parse =
            |json: &str| TransportConfig::parse("cert".into(), "key".into(), "ca".into(), json);
        assert!(parse(r#"{"east":"https://east.internal"}"#).is_ok());
        assert!(parse(r#"{"east":"http://east.internal"}"#).is_err());
        assert!(parse(r#"{"east":"https://east.internal/path"}"#).is_err());
        assert!(parse(r#"{"east":"https://one","east":"https://two"}"#).is_err());
        assert!(parse(r#"[]"#).is_err());
        assert!(parse(r#"{}"#).is_err());
    }

    #[test]
    fn response_headers_drop_hop_and_framing_state() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", HeaderValue::from_static("x-hop, keep-alive"));
        headers.insert("x-hop", HeaderValue::from_static("private"));
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        headers.insert("content-length", HeaderValue::from_static("123"));
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );

        sanitize_response_headers(&mut headers);

        assert!(!headers.contains_key("connection"));
        assert!(!headers.contains_key("x-hop"));
        assert!(!headers.contains_key("keep-alive"));
        assert!(!headers.contains_key("transfer-encoding"));
        assert!(!headers.contains_key("content-length"));
        assert_eq!(headers["content-type"], "text/event-stream");
    }

    #[test]
    fn wan_metrics_register_with_bounded_label_contracts() {
        let registry = Registry::new();
        ensure_metrics_registered_prometheus(&registry).unwrap();
        WAN_ATTEMPTS.with_label_values(&["test-region"]).inc();
        WAN_OUTCOMES
            .with_label_values(&["test-region", "stream", "complete"])
            .inc();
        WAN_RESPONSE_STATUS
            .with_label_values(&["test-region", "200"])
            .inc();
        WAN_BYTES.with_label_values(&["test-region"]).inc();
        let names = registry
            .gather()
            .into_iter()
            .map(|family| family.name().to_owned())
            .collect::<Vec<_>>();
        for expected in [
            "morph_global_routing_wan_attempts_total",
            "morph_global_routing_wan_outcomes_total",
            "morph_global_routing_wan_response_status_total",
            "morph_global_routing_wan_bytes_relayed_total",
        ] {
            assert!(
                names.iter().any(|name| name == expected),
                "missing {expected}"
            );
        }
    }

    #[test]
    fn dropping_live_stream_records_observable_cancellation() {
        let counter = WAN_OUTCOMES.with_label_values(&["cancel-test", "stream", "cancelled"]);
        let before = counter.get();
        let never_polled = stream::pending::<Result<Bytes, reqwest::Error>>();
        drop(InstrumentedResponseBody::new(
            "cancel-test".to_owned(),
            Box::pin(never_polled),
        ));
        assert_eq!(counter.get(), before + 1);
    }

    #[test]
    fn production_call_graph_returns_wan_ownership_before_local_generate() {
        let preprocessor = include_str!("preprocessor.rs");
        let dispatch_sites = preprocessor
            .match_indices("global_routing_transport::dispatch_signed(&signed)")
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        assert_eq!(dispatch_sites.len(), 1, "only chat may dispatch over WAN");
        for &dispatch in &dispatch_sites {
            let tail = &preprocessor[dispatch..];
            let owned_return = tail
                .find("return Ok(ResponseStream::new")
                .expect("globally routed request must return WAN ownership");
            let local_generate = tail
                .find("let response_stream = next.generate")
                .expect("test must observe the local generation site");
            assert!(
                owned_return < local_generate,
                "local next.generate became reachable after authoritative dispatch"
            );
        }

        let http_source = include_str!("http/service/openai.rs");
        assert_eq!(
            http_source
                .matches("take_global_wan_response(&ctx, &request_id)?")
                .count(),
            1,
            "only the chat source may consume a WAN response"
        );

        assert!(
            !preprocessor.contains("\"/v1/completions\",\n                &common_request.model"),
            "legacy completions must not own WAN dispatch"
        );
        assert!(
            preprocessor.contains("if supplied_prompt_token_digest.is_none()"),
            "a selected regional request must skip the second global decision"
        );
        let ownership_guard = preprocessor
            .find("if supplied_prompt_token_digest.is_none()")
            .unwrap();
        assert!(
            ownership_guard < dispatch_sites[0],
            "the sole WAN dispatch must be inside the unselected request guard"
        );
    }
}
