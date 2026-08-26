// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signed internal request envelope for global routing.
//!
//! The envelope is the trust boundary between global processing
//! (authentication, canonical normalization, native tokenization, the single
//! regional routing decision) and regional execution (pod proxy admission,
//! serving, billing finalization). The dispatcher mints one envelope per
//! request; the selected region's pod proxy verifies it and then trusts the
//! authenticated account identity and canonical request without repeating
//! public authentication or normalization.
//!
//! # Wire format
//!
//! ```json
//! {"version":1,"key_id":"...","payload":"...","signature":"..."}
//! ```
//!
//! The signature covers the exact payload bytes as transmitted. Verifiers in
//! any language MUST verify the signature over the received base64-decoded
//! payload bytes and only then parse the JSON — never re-serialize. This
//! removes every cross-language canonicalization hazard by construction.
//!
//! Downstream of the pod proxy, the execute-exact contract is carried by
//! `request.token_ids_sha256`: the serving frontend recomputes the digest
//! from its own native preprocessing and rejects any mismatch, so a request
//! can never be routed with one token sequence and executed with another.

use std::sync::{LazyLock, OnceLock};

use prometheus::{IntCounterVec, Opts, Registry};
use serde::{Deserialize, Serialize};

use crate::protocols::TokenIdType;

pub const ENVELOPE_VERSION: u32 = 1;
pub const SIGNATURE_DOMAIN: &[u8] = b"morph.global-routing.envelope.v1\0";
pub const TRUSTED_AUTH_METADATA_KEY: &str = "global-routing-auth-v1";
pub const SIGNED_ENVELOPE_CONTEXT_KEY: &str = "global_routing.signed_envelope";

static SIGNING_CONFIG: OnceLock<Result<SigningConfig, String>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub version: u32,
    pub request_id: String,
    pub decision_id: String,
    pub issuer: String,
    pub audience: String,
    /// Unix seconds.
    pub issued_at: u64,
    /// Unix seconds. Envelopes are short-lived dispatch instructions.
    pub expires_at: u64,
    pub nonce: String,
    pub account: Account,
    pub request: Request,
    pub routing: Routing,
}

/// Immutable internal identifiers resolved by global authentication.
/// Never a raw API key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub api_key_id: String,
    pub user_id: String,
    pub org_id: Option<String>,
    pub billing_tier: String,
    pub service_tier: String,
    pub zero_data_retention: bool,
    pub lifetime_requests: Option<u64>,
    pub service_tier_explicit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// The public endpoint the client called, e.g. `/v1/chat/completions`.
    pub endpoint: String,
    pub canonical_model: String,
    /// The canonically normalized OpenAI request body. This exact body is
    /// what the selected region executes.
    pub normalized_body: serde_json::Value,
    /// Digest of the exact native token sequence used for routing
    /// The selected frontend must execute this exact normalized request.
    pub body_sha256: String,
    pub token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    /// Region key of the selected data center, e.g. `us-east5-financial`.
    pub selected_region: String,
    pub selected_pool: String,
    pub policy_version: String,
    pub ckf_generation: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EnvelopeError {
    #[error("envelope is malformed")]
    Malformed,
    #[error("envelope version is unsupported")]
    UnsupportedVersion,
    #[error("envelope signature is invalid")]
    BadSignature,
    #[error("envelope key identifier is unknown")]
    UnknownKey,
    #[error("envelope is expired or not yet valid")]
    Expired,
    #[error("envelope is addressed to region {selected}, not {local}")]
    WrongRegion { selected: String, local: String },
}

pub struct EnvelopeSigner {
    key_id: String,
    key: ed25519_dalek::SigningKey,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedEnvelope {
    pub version: u32,
    pub key_id: String,
    pub payload: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedAuthMetadata {
    pub version: u32,
    pub api_key_id: String,
    pub user_id: String,
    pub org_id: Option<String>,
    pub billing_tier: String,
    pub service_tier: String,
    pub zero_data_retention: bool,
    pub lifetime_requests: Option<u64>,
    pub service_tier_explicit: bool,
}

impl From<TrustedAuthMetadata> for Account {
    fn from(value: TrustedAuthMetadata) -> Self {
        Self {
            api_key_id: value.api_key_id,
            user_id: value.user_id,
            org_id: value.org_id,
            billing_tier: value.billing_tier,
            service_tier: value.service_tier,
            zero_data_retention: value.zero_data_retention,
            lifetime_requests: value.lifetime_requests,
            service_tier_explicit: value.service_tier_explicit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SignedRoutingDecision {
    pub decision: crate::global_routing::TokenDecisionResponse,
    pub envelope: SignedEnvelope,
}

struct SigningConfig {
    issuer: String,
    signer: EnvelopeSigner,
}

impl SigningConfig {
    fn from_env() -> Result<Self, String> {
        let issuer = required_env("DYN_GLOBAL_ROUTER_ISSUER")?;
        let key_id = required_env("DYN_GLOBAL_ROUTER_SIGNING_KEY_ID")?;
        let path = required_env("DYN_GLOBAL_ROUTER_SIGNING_KEY_PATH")?;
        let pem = std::fs::read_to_string(&path)
            .map_err(|error| format!("failed to read global routing signing key: {error}"))?;
        let signer = EnvelopeSigner::from_pkcs8_pem(key_id, &pem)
            .map_err(|error| format!("failed to parse global routing signing key: {error}"))?;
        Ok(Self { issuer, signer })
    }
}

fn required_env(name: &str) -> Result<String, String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} is required when global routing is enabled"))
}

#[derive(Debug, thiserror::Error)]
pub enum MintError {
    #[error("trusted global routing authentication metadata is missing")]
    MissingAuthMetadata,
    #[error("trusted global routing authentication metadata is malformed")]
    MalformedAuthMetadata,
    #[error("trusted global routing authentication metadata is incomplete")]
    IncompleteAuthMetadata,
    #[error("global routing decision is incomplete")]
    IncompleteDecision,
    #[error("global routing signing is misconfigured: {0}")]
    SigningConfig(String),
    #[error("failed to serialize signed internal request: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn decode_trusted_auth(value: Option<&str>) -> Result<TrustedAuthMetadata, MintError> {
    use base64::Engine;

    let encoded = value.ok_or(MintError::MissingAuthMetadata)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| MintError::MalformedAuthMetadata)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| MintError::MalformedAuthMetadata)?;
    let object = value.as_object().ok_or(MintError::MalformedAuthMetadata)?;
    if !object.contains_key("org_id") || !object.contains_key("lifetime_requests") {
        return Err(MintError::MalformedAuthMetadata);
    }
    let metadata: TrustedAuthMetadata =
        serde_json::from_value(value).map_err(|_| MintError::MalformedAuthMetadata)?;
    if metadata.version != ENVELOPE_VERSION
        || metadata.api_key_id.trim().is_empty()
        || metadata.user_id.trim().is_empty()
        || metadata
            .org_id
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        || metadata.billing_tier.trim().is_empty()
        || !matches!(metadata.service_tier.as_str(), "default" | "standby")
    {
        return Err(MintError::IncompleteAuthMetadata);
    }
    Ok(metadata)
}

pub fn mint_signed_routing_decision(
    decision: crate::global_routing::TokenDecisionResponse,
    auth_metadata: Option<&str>,
    request_id: &str,
    endpoint: &str,
    canonical_model: &str,
    normalized_body: serde_json::Value,
    token_ids: &[TokenIdType],
    now_unix_seconds: u64,
) -> Result<SignedRoutingDecision, MintError> {
    let selected_region = decision
        .selected_region
        .as_ref()
        .ok_or(MintError::IncompleteDecision)?;
    let selected_pool = decision
        .selected_pool_id
        .as_ref()
        .ok_or(MintError::IncompleteDecision)?;
    let account = decode_trusted_auth(auth_metadata)?.into();
    let config = SIGNING_CONFIG
        .get_or_init(SigningConfig::from_env)
        .as_ref()
        .map_err(|error| MintError::SigningConfig(error.clone()))?;
    let model_audience = canonical_model
        .strip_prefix("morph-")
        .unwrap_or(canonical_model);
    let payload = Envelope {
        version: ENVELOPE_VERSION,
        request_id: request_id.to_owned(),
        decision_id: uuid::Uuid::new_v4().to_string(),
        issuer: config.issuer.clone(),
        audience: format!("{model_audience}-{selected_region}"),
        issued_at: now_unix_seconds,
        expires_at: now_unix_seconds.saturating_add(30),
        nonce: uuid::Uuid::new_v4().to_string(),
        account,
        request: Request {
            endpoint: endpoint.to_owned(),
            canonical_model: canonical_model.to_owned(),
            body_sha256: body_digest(&normalized_body)?,
            token_ids_sha256: prompt_token_digest(token_ids),
            normalized_body,
        },
        routing: Routing {
            selected_region: selected_region.clone(),
            selected_pool: selected_pool.clone(),
            policy_version: "v1".to_owned(),
            ckf_generation: decision.generation,
        },
    };
    let envelope = config.signer.sign(&payload)?;
    ENVELOPES.with_label_values(&["success"]).inc();
    Ok(SignedRoutingDecision { decision, envelope })
}

impl EnvelopeSigner {
    /// Load a PKCS#8 PEM Ed25519 private key
    /// (`openssl genpkey -algorithm ed25519`).
    pub fn from_pkcs8_pem(
        key_id: impl Into<String>,
        pem: &str,
    ) -> Result<Self, ed25519_dalek::pkcs8::Error> {
        use ed25519_dalek::pkcs8::DecodePrivateKey;
        Ok(Self {
            key_id: key_id.into(),
            key: ed25519_dalek::SigningKey::from_pkcs8_pem(pem)?,
        })
    }

    /// Build a signer from raw Ed25519 seed bytes, for keys sourced from a
    /// secret manager rather than a PEM file.
    pub fn from_key_bytes(key_id: impl Into<String>, seed: &[u8; 32]) -> Self {
        Self {
            key_id: key_id.into(),
            key: ed25519_dalek::SigningKey::from_bytes(seed),
        }
    }

    pub fn verifier(&self) -> EnvelopeVerifier {
        EnvelopeVerifier {
            keys: [(self.key_id.clone(), self.key.verifying_key())]
                .into_iter()
                .collect(),
        }
    }

    /// Serialize and sign the envelope into its wire form.
    pub fn sign(&self, envelope: &Envelope) -> Result<SignedEnvelope, serde_json::Error> {
        use base64::Engine;
        use ed25519_dalek::Signer;

        let engine = &base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let payload = serde_json::to_vec(envelope)?;
        let mut signed_bytes = Vec::with_capacity(SIGNATURE_DOMAIN.len() + payload.len());
        signed_bytes.extend_from_slice(SIGNATURE_DOMAIN);
        signed_bytes.extend_from_slice(&payload);
        let signature = self.key.sign(&signed_bytes);
        Ok(SignedEnvelope {
            version: ENVELOPE_VERSION,
            key_id: self.key_id.clone(),
            payload: engine.encode(payload),
            signature: engine.encode(signature.to_bytes()),
        })
    }
}

pub struct EnvelopeVerifier {
    keys: std::collections::HashMap<String, ed25519_dalek::VerifyingKey>,
}

impl EnvelopeVerifier {
    /// Load a SPKI PEM Ed25519 public key (`openssl pkey -pubout`).
    pub fn from_public_key_pem(
        key_id: impl Into<String>,
        pem: &str,
    ) -> Result<Self, ed25519_dalek::pkcs8::spki::Error> {
        use ed25519_dalek::pkcs8::DecodePublicKey;
        Ok(Self {
            keys: [(
                key_id.into(),
                ed25519_dalek::VerifyingKey::from_public_key_pem(pem)?,
            )]
            .into_iter()
            .collect(),
        })
    }

    /// Verify signature, version, validity window, and addressing.
    ///
    /// `local_region` is the verifier's own region key; an envelope selected
    /// for any other region is rejected. Replay protection beyond the validity
    /// window (the `nonce`) is the caller's responsibility, since it needs
    /// shared state.
    pub fn verify(
        &self,
        wire: &SignedEnvelope,
        local_region: &str,
        now_unix_seconds: u64,
    ) -> Result<Envelope, EnvelopeError> {
        let envelope = self.verify_addressed(wire, now_unix_seconds)?;
        if envelope.routing.selected_region != local_region {
            return Err(EnvelopeError::WrongRegion {
                selected: envelope.routing.selected_region,
                local: local_region.to_owned(),
            });
        }
        Ok(envelope)
    }

    /// Verify signature, version, and validity window without asserting the
    /// verifier is the addressed region. This is the dispatcher's check: it
    /// forwards to whatever region the envelope selects, so only the target
    /// region's own verification may enforce addressing.
    pub fn verify_addressed(
        &self,
        wire: &SignedEnvelope,
        now_unix_seconds: u64,
    ) -> Result<Envelope, EnvelopeError> {
        use base64::Engine;

        let engine = &base64::engine::general_purpose::URL_SAFE_NO_PAD;
        if wire.version != ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion);
        }
        let key = self
            .keys
            .get(&wire.key_id)
            .ok_or(EnvelopeError::UnknownKey)?;
        let payload = engine
            .decode(&wire.payload)
            .map_err(|_| EnvelopeError::Malformed)?;
        let signature = engine
            .decode(&wire.signature)
            .ok()
            .and_then(|bytes| ed25519_dalek::Signature::from_slice(&bytes).ok())
            .ok_or(EnvelopeError::Malformed)?;
        let mut signed_bytes = Vec::with_capacity(SIGNATURE_DOMAIN.len() + payload.len());
        signed_bytes.extend_from_slice(SIGNATURE_DOMAIN);
        signed_bytes.extend_from_slice(&payload);
        key.verify_strict(&signed_bytes, &signature)
            .map_err(|_| EnvelopeError::BadSignature)?;

        let envelope: Envelope =
            serde_json::from_slice(&payload).map_err(|_| EnvelopeError::Malformed)?;
        if envelope.version != ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion);
        }
        if now_unix_seconds < envelope.issued_at || now_unix_seconds > envelope.expires_at {
            return Err(EnvelopeError::Expired);
        }
        Ok(envelope)
    }
}

static EXECUTE_EXACT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_execute_exact_total",
            "Execute-exact digest comparisons on globally routed requests.",
        ),
        &["outcome"],
    )
    .expect("static metric options are valid")
});

static ENVELOPES: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "morph_global_routing_signed_envelopes_total",
            "Signed global routing envelope mint outcomes.",
        ),
        &["outcome"],
    )
    .expect("static metric options are valid")
});

/// Register this module's frontend-side collectors.
pub fn ensure_metrics_registered_prometheus(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(EXECUTE_EXACT.clone()))?;
    registry.register(Box::new(ENVELOPES.clone()))
}

/// Enforce the execute-exact contract on the serving frontend.
///
/// `expected` is `nvext.prompt_token_digest`, stamped by the regional pod
/// proxy from the signed routing envelope. A mismatch means the frontend's
/// preprocessing produced a different token sequence than the one that drove
/// the routing decision — normalization drift between global processing and
/// this frontend — and the request must be rejected, never served.
pub fn enforce_prompt_token_digest(
    expected: Option<&str>,
    token_ids: &[TokenIdType],
    request_id: &str,
) -> Result<(), crate::http::service::error::HttpError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let executed = prompt_token_digest(token_ids);
    if executed == expected {
        EXECUTE_EXACT.with_label_values(&["match"]).inc();
        return Ok(());
    }
    EXECUTE_EXACT.with_label_values(&["mismatch"]).inc();
    tracing::warn!(
        request_id,
        expected,
        executed,
        token_count = token_ids.len(),
        "execute-exact violation: routed and executed token sequences differ"
    );
    Err(crate::http::service::error::HttpError {
        code: 409,
        message: "prompt token digest mismatch: the routed and executed token sequences differ"
            .to_string(),
    })
}

/// SHA256 digest of the exact native token sequence, as lowercase hex.
///
/// Token ids are encoded as unsigned big endian `u32`, matching the frozen
/// cross language contract vector.
pub fn prompt_token_digest(token_ids: &[TokenIdType]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    for token_id in token_ids {
        hasher.update(token_id.to_be_bytes());
    }
    hex_digest(hasher.finalize())
}

/// SHA256 over compact JSON with object keys sorted recursively.
///
/// The payload itself preserves the normalized body's field order. This digest
/// uses a language neutral ordering so Rust and Python can recompute it without
/// relying on their map implementations.
pub fn body_digest(normalized_body: &serde_json::Value) -> Result<String, serde_json::Error> {
    use sha2::Digest;
    let mut bytes = Vec::new();
    write_canonical_json(normalized_body, &mut bytes)?;
    Ok(hex_digest(sha2::Sha256::digest(&bytes)))
}

fn write_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), serde_json::Error> {
    match value {
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)?;
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
        scalar => serde_json::to_writer(output, scalar)?,
    }
    Ok(())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let mut hex = String::with_capacity(64);
    for byte in bytes.as_ref() {
        use std::fmt::Write;
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic key for cross-implementation test vectors only.
    fn test_signer() -> EnvelopeSigner {
        EnvelopeSigner {
            key_id: "test-key-1".into(),
            key: ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]),
        }
    }

    fn envelope() -> Envelope {
        let normalized_body = serde_json::json!({
            "model": "morph-dsv4flash",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
        });
        Envelope {
            version: ENVELOPE_VERSION,
            request_id: "req_1".into(),
            decision_id: "decision_1".into(),
            issuer: "morph-global-router".into(),
            audience: "dsv4flash-us-east5-financial".into(),
            issued_at: 1_787_780_000,
            expires_at: 1_787_780_030,
            nonce: "nonce_1".into(),
            account: Account {
                api_key_id: "key_1".into(),
                user_id: "user_1".into(),
                org_id: Some("org_1".into()),
                billing_tier: "pro".into(),
                service_tier: "default".into(),
                zero_data_retention: false,
                lifetime_requests: Some(7),
                service_tier_explicit: true,
            },
            request: Request {
                endpoint: "/v1/chat/completions".into(),
                canonical_model: "morph-dsv4flash".into(),
                normalized_body: normalized_body.clone(),
                body_sha256: body_digest(&normalized_body).unwrap(),
                token_ids_sha256: prompt_token_digest(&[1, 2, 3, 4]),
            },
            routing: Routing {
                selected_region: "us-east5-financial".into(),
                selected_pool: "pool_1".into(),
                policy_version: "policy_v1".into(),
                ckf_generation: 42,
            },
        }
    }

    #[test]
    fn sign_verify_roundtrip() {
        let signer = test_signer();
        let envelope = envelope();
        let wire = signer.sign(&envelope).unwrap();
        let verified = signer
            .verifier()
            .verify(&wire, "us-east5-financial", 1_787_780_010)
            .unwrap();
        assert_eq!(verified, envelope);
    }

    #[test]
    fn verification_rejects_tampering_expiry_and_misaddressing() {
        let signer = test_signer();
        let verifier = signer.verifier();
        let wire = signer.sign(&envelope()).unwrap();
        let now = 1_787_780_010;

        // Payload tampering: flip one payload character.
        let mut tampered = wire.clone();
        tampered.payload.replace_range(
            ..1,
            if tampered.payload.starts_with('A') {
                "B"
            } else {
                "A"
            },
        );
        assert!(matches!(
            verifier.verify(&tampered, "us-east5-financial", now),
            Err(EnvelopeError::BadSignature | EnvelopeError::Malformed)
        ));

        // Signature from a different key.
        let other = EnvelopeSigner {
            key_id: "test-key-1".into(),
            key: ed25519_dalek::SigningKey::from_bytes(&[8u8; 32]),
        };
        let foreign = other.sign(&envelope()).unwrap();
        assert_eq!(
            verifier.verify(&foreign, "us-east5-financial", now),
            Err(EnvelopeError::BadSignature)
        );

        // Expired and not-yet-valid.
        assert_eq!(
            verifier.verify(&wire, "us-east5-financial", 1_787_780_031),
            Err(EnvelopeError::Expired)
        );
        assert_eq!(
            verifier.verify(&wire, "us-east5-financial", 1_787_779_999),
            Err(EnvelopeError::Expired)
        );

        // Addressed to another region.
        assert_eq!(
            verifier.verify(&wire, "us-west1-financial", now),
            Err(EnvelopeError::WrongRegion {
                selected: "us-east5-financial".into(),
                local: "us-west1-financial".into(),
            })
        );

        // Malformed encoding.
        let malformed = SignedEnvelope {
            version: 1,
            key_id: "test-key-1".into(),
            payload: "***".into(),
            signature: "***".into(),
        };
        assert_eq!(
            verifier.verify(&malformed, "us-east5-financial", now),
            Err(EnvelopeError::Malformed)
        );
    }

    #[test]
    fn matches_cross_language_contract_vector() {
        use base64::Engine;

        let engine = &base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let seed = engine
            .decode("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8")
            .unwrap();
        let signer = EnvelopeSigner {
            key_id: "contract-test-key".into(),
            key: ed25519_dalek::SigningKey::from_bytes(seed.as_slice().try_into().unwrap()),
        };
        let payload = r#"{"version":1,"request_id":"req_contract_1","decision_id":"decision_contract_1","issuer":"morph-global-router","audience":"dsv4flash-us-east5-financial","issued_at":1787780000,"expires_at":1787780030,"nonce":"nonce_contract_1","account":{"api_key_id":"key_1","user_id":"user_1","org_id":"org_1","billing_tier":"pro","service_tier":"default","zero_data_retention":false,"lifetime_requests":7,"service_tier_explicit":true},"request":{"endpoint":"/v1/chat/completions","canonical_model":"morph-dsv4flash","normalized_body":{"model":"morph-dsv4flash","messages":[{"role":"user","content":"hello"}],"stream":true},"body_sha256":"b26a7dbf98b6cda6b6efad40e14948127007a81c1dacc0e18c5c71604a5504d9","token_ids_sha256":"7b0b5ea3ff36958c8e32ccf24b71da9ac68e51d0881bf75e62b837ec9ea6f3a5"},"routing":{"selected_region":"us-east5-financial","selected_pool":"dsv4flash-east","policy_version":"v1","ckf_generation":42}}"#;
        let envelope: Envelope = serde_json::from_str(payload).unwrap();
        let signed = signer.sign(&envelope).unwrap();
        assert_eq!(signed.payload, engine.encode(payload.as_bytes()));
        assert_eq!(
            signed.signature,
            "JEEHnV8xOgdVVpdVjgzQ4UjA-DeOeMfT17g-ATCxPEHXJC7KZVfu5hqzZ61WAFgJfiD_husTUfzFmB77vsOiCA"
        );
        assert_eq!(
            prompt_token_digest(&[1, 2, 3]),
            "7b0b5ea3ff36958c8e32ccf24b71da9ac68e51d0881bf75e62b837ec9ea6f3a5"
        );
        assert_eq!(
            body_digest(&envelope.request.normalized_body).unwrap(),
            "b26a7dbf98b6cda6b6efad40e14948127007a81c1dacc0e18c5c71604a5504d9"
        );
    }

    #[test]
    fn execute_exact_enforces_only_when_a_digest_is_expected() {
        let tokens = [1, 2, 3, 4];
        assert!(enforce_prompt_token_digest(None, &tokens, "req").is_ok());
        assert!(
            enforce_prompt_token_digest(Some(&prompt_token_digest(&tokens)), &tokens, "req")
                .is_ok()
        );
        let error = enforce_prompt_token_digest(Some(&prompt_token_digest(&[9])), &tokens, "req")
            .unwrap_err();
        assert_eq!(error.code, 409);
    }

    #[test]
    fn trusted_auth_metadata_is_versioned_strict_and_complete() {
        use base64::Engine;

        let json = r#"{"version":1,"api_key_id":"key","user_id":"user","org_id":"org","billing_tier":"pro","service_tier":"default","service_tier_explicit":true,"zero_data_retention":false,"lifetime_requests":7}"#;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json);
        let metadata = decode_trusted_auth(Some(&encoded)).unwrap();
        assert_eq!(metadata.org_id.as_deref(), Some("org"));
        assert!(metadata.service_tier_explicit);

        let unknown = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json.replace(
            "\"lifetime_requests\":7",
            "\"lifetime_requests\":7,\"extra\":true",
        ));
        assert!(matches!(
            decode_trusted_auth(Some(&unknown)),
            Err(MintError::MalformedAuthMetadata)
        ));
        assert!(matches!(
            decode_trusted_auth(None),
            Err(MintError::MissingAuthMetadata)
        ));
    }

    #[test]
    fn token_digest_is_order_and_length_sensitive() {
        assert_eq!(
            prompt_token_digest(&[1, 2, 3]),
            prompt_token_digest(&[1, 2, 3])
        );
        assert_ne!(
            prompt_token_digest(&[1, 2, 3]),
            prompt_token_digest(&[3, 2, 1])
        );
        assert_ne!(
            prompt_token_digest(&[1, 2, 3]),
            prompt_token_digest(&[1, 2])
        );
        assert_eq!(prompt_token_digest(&[]).len(), 64);
    }

    /// The envelope is fully deterministic (struct-order serialization plus
    /// RFC 8032 deterministic signatures), so cross-implementation verifier
    /// vectors can be generated by printing `sign` output for a fixed key.
    #[test]
    fn wire_form_is_deterministic() {
        use base64::Engine;

        let first = test_signer().sign(&envelope()).unwrap();
        let second = test_signer().sign(&envelope()).unwrap();
        assert_eq!(first, second);

        assert_eq!(first.version, ENVELOPE_VERSION);
        assert_eq!(first.key_id, "test-key-1");
        // Ed25519 signatures are 64 bytes; base64url without padding is 86.
        assert_eq!(first.signature.len(), 86);
        // The payload is the plain serde serialization of the envelope.
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(first.payload)
            .unwrap();
        assert_eq!(payload, serde_json::to_vec(&envelope()).unwrap());
    }
}
