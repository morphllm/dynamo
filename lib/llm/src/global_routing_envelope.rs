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
//! ```text
//! env1.<base64url(payload JSON)>.<base64url(Ed25519 signature)>
//! ```
//!
//! The signature covers the exact payload bytes as transmitted. Verifiers in
//! any language MUST verify the signature over the received base64-decoded
//! payload bytes and only then parse the JSON — never re-serialize. This
//! removes every cross-language canonicalization hazard by construction.
//!
//! Downstream of the pod proxy, the execute-exact contract is carried by
//! `request.prompt_token_digest`: the serving frontend recomputes the digest
//! from its own native preprocessing and rejects any mismatch, so a request
//! can never be routed with one token sequence and executed with another.

use std::sync::LazyLock;

use prometheus::{IntCounterVec, Opts, Registry};
use serde::{Deserialize, Serialize};

use crate::protocols::TokenIdType;

/// Envelope wire prefix. Bump only with a coordinated verifier rollout.
const WIRE_PREFIX: &str = "env1";
pub const ENVELOPE_VERSION: u32 = 1;

/// Domain separation context for the native prompt token digest.
const TOKEN_DIGEST_CONTEXT: &str = "dynamo/global-routing/prompt-tokens/v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub version: u32,
    pub request_id: String,
    pub decision_id: String,
    /// Unix seconds.
    pub issued_at: u64,
    /// Unix seconds. Envelopes are short-lived dispatch instructions.
    pub expires_at: u64,
    pub account: Account,
    pub request: Request,
    pub routing: Routing,
    pub integrity: Integrity,
}

/// Immutable internal identifiers resolved by global authentication.
/// Never a raw API key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub api_key_id: String,
    pub user_id: String,
    pub billing_owner_id: String,
    pub service_tier: String,
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
    /// (see [`prompt_token_digest`]). The serving frontend must reproduce it.
    pub prompt_token_digest: String,
    pub prompt_token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    /// Region key of the selected data center, e.g. `us-east5-financial`.
    pub selected_region: String,
    pub selected_pool: String,
    pub policy_version: String,
    pub ckf_version: String,
    pub cache_overlap_blocks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Integrity {
    /// `sha256:<hex>` over this payload's own serialization of
    /// `request.normalized_body`. Informational belt-and-braces inside an
    /// already-signed payload; the authoritative execution contract is
    /// `prompt_token_digest`.
    pub body_digest: String,
    pub nonce: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EnvelopeError {
    #[error("envelope is malformed")]
    Malformed,
    #[error("envelope version is unsupported")]
    UnsupportedVersion,
    #[error("envelope signature is invalid")]
    BadSignature,
    #[error("envelope is expired or not yet valid")]
    Expired,
    #[error("envelope is addressed to region {selected}, not {local}")]
    WrongRegion { selected: String, local: String },
}

pub struct EnvelopeSigner {
    key: ed25519_dalek::SigningKey,
}

impl EnvelopeSigner {
    /// Load a PKCS#8 PEM Ed25519 private key
    /// (`openssl genpkey -algorithm ed25519`).
    pub fn from_pkcs8_pem(pem: &str) -> Result<Self, ed25519_dalek::pkcs8::Error> {
        use ed25519_dalek::pkcs8::DecodePrivateKey;
        Ok(Self {
            key: ed25519_dalek::SigningKey::from_pkcs8_pem(pem)?,
        })
    }

    pub fn verifier(&self) -> EnvelopeVerifier {
        EnvelopeVerifier {
            key: self.key.verifying_key(),
        }
    }

    /// Serialize and sign the envelope into its wire form.
    pub fn sign(&self, envelope: &Envelope) -> Result<String, serde_json::Error> {
        use base64::Engine;
        use ed25519_dalek::Signer;

        let engine = &base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let payload = serde_json::to_vec(envelope)?;
        let signature = self.key.sign(&payload);
        Ok(format!(
            "{WIRE_PREFIX}.{}.{}",
            engine.encode(&payload),
            engine.encode(signature.to_bytes())
        ))
    }
}

pub struct EnvelopeVerifier {
    key: ed25519_dalek::VerifyingKey,
}

impl EnvelopeVerifier {
    /// Load a SPKI PEM Ed25519 public key (`openssl pkey -pubout`).
    pub fn from_public_key_pem(pem: &str) -> Result<Self, ed25519_dalek::pkcs8::spki::Error> {
        use ed25519_dalek::pkcs8::DecodePublicKey;
        Ok(Self {
            key: ed25519_dalek::VerifyingKey::from_public_key_pem(pem)?,
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
        wire: &str,
        local_region: &str,
        now_unix_seconds: u64,
    ) -> Result<Envelope, EnvelopeError> {
        use base64::Engine;

        let engine = &base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let mut parts = wire.split('.');
        let (Some(WIRE_PREFIX), Some(payload), Some(signature), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(EnvelopeError::Malformed);
        };
        let payload = engine.decode(payload).map_err(|_| EnvelopeError::Malformed)?;
        let signature = engine
            .decode(signature)
            .ok()
            .and_then(|bytes| ed25519_dalek::Signature::from_slice(&bytes).ok())
            .ok_or(EnvelopeError::Malformed)?;
        self.key
            .verify_strict(&payload, &signature)
            .map_err(|_| EnvelopeError::BadSignature)?;

        let envelope: Envelope =
            serde_json::from_slice(&payload).map_err(|_| EnvelopeError::Malformed)?;
        if envelope.version != ENVELOPE_VERSION {
            return Err(EnvelopeError::UnsupportedVersion);
        }
        if now_unix_seconds < envelope.issued_at || now_unix_seconds > envelope.expires_at {
            return Err(EnvelopeError::Expired);
        }
        if envelope.routing.selected_region != local_region {
            return Err(EnvelopeError::WrongRegion {
                selected: envelope.routing.selected_region,
                local: local_region.to_owned(),
            });
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

/// Register this module's frontend-side collectors.
pub fn ensure_metrics_registered_prometheus(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(EXECUTE_EXACT.clone()))
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

/// Digest of the exact native token sequence, as `blake3:<hex>`.
///
/// Domain separated and length prefixed; token ids are hashed as little
/// endian `u32`. Both the dispatcher (over the routing tokens) and the
/// serving frontend (over its own preprocessing output) must use this
/// function so the execute-exact comparison is meaningful.
pub fn prompt_token_digest(token_ids: &[TokenIdType]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(TOKEN_DIGEST_CONTEXT);
    hasher.update(&(token_ids.len() as u64).to_le_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// `sha256:<hex>` over the payload's serialization of the normalized body.
pub fn body_digest(normalized_body: &serde_json::Value) -> Result<String, serde_json::Error> {
    use sha2::Digest;
    let bytes = serde_json::to_vec(normalized_body)?;
    let mut hex = String::with_capacity(7 + 64);
    hex.push_str("sha256:");
    for byte in sha2::Sha256::digest(&bytes) {
        use std::fmt::Write;
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic key for cross-implementation test vectors only.
    fn test_signer() -> EnvelopeSigner {
        EnvelopeSigner {
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
            issued_at: 1_787_780_000,
            expires_at: 1_787_780_030,
            account: Account {
                api_key_id: "key_1".into(),
                user_id: "user_1".into(),
                billing_owner_id: "org_1".into(),
                service_tier: "normal".into(),
            },
            request: Request {
                endpoint: "/v1/chat/completions".into(),
                canonical_model: "morph-dsv4flash".into(),
                normalized_body: normalized_body.clone(),
                prompt_token_digest: prompt_token_digest(&[1, 2, 3, 4]),
                prompt_token_count: 4,
            },
            routing: Routing {
                selected_region: "us-east5-financial".into(),
                selected_pool: "pool_1".into(),
                policy_version: "policy_v1".into(),
                ckf_version: "ckf_v1".into(),
                cache_overlap_blocks: 3,
            },
            integrity: Integrity {
                body_digest: body_digest(&normalized_body).unwrap(),
                nonce: "nonce_1".into(),
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
        let mut tampered = wire.clone().into_bytes();
        let payload_start = WIRE_PREFIX.len() + 1;
        tampered[payload_start] = if tampered[payload_start] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(tampered).unwrap();
        assert!(matches!(
            verifier.verify(&tampered, "us-east5-financial", now),
            Err(EnvelopeError::BadSignature | EnvelopeError::Malformed)
        ));

        // Signature from a different key.
        let other = EnvelopeSigner {
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

        // Garbage.
        assert_eq!(
            verifier.verify("env1.zzz", "us-east5-financial", now),
            Err(EnvelopeError::Malformed)
        );
        assert_eq!(
            verifier.verify("env2.a.b", "us-east5-financial", now),
            Err(EnvelopeError::Malformed)
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
        let error =
            enforce_prompt_token_digest(Some(&prompt_token_digest(&[9])), &tokens, "req")
                .unwrap_err();
        assert_eq!(error.code, 409);
    }

    #[test]
    fn token_digest_is_order_and_length_sensitive() {
        assert_eq!(prompt_token_digest(&[1, 2, 3]), prompt_token_digest(&[1, 2, 3]));
        assert_ne!(prompt_token_digest(&[1, 2, 3]), prompt_token_digest(&[3, 2, 1]));
        assert_ne!(prompt_token_digest(&[1, 2, 3]), prompt_token_digest(&[1, 2]));
        assert!(prompt_token_digest(&[]).starts_with("blake3:"));
    }

    /// The wire form is fully deterministic (struct-order serialization plus
    /// RFC 8032 deterministic signatures), so cross-implementation verifier
    /// vectors can be generated by printing `sign` output for a fixed key.
    #[test]
    fn wire_form_is_deterministic() {
        use base64::Engine;

        let first = test_signer().sign(&envelope()).unwrap();
        let second = test_signer().sign(&envelope()).unwrap();
        assert_eq!(first, second);

        let parts: Vec<&str> = first.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "env1");
        // Ed25519 signatures are 64 bytes; base64url without padding is 86.
        assert_eq!(parts[2].len(), 86);
        // The payload is the plain serde serialization of the envelope.
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(parts[1])
            .unwrap();
        assert_eq!(payload, serde_json::to_vec(&envelope()).unwrap());
    }
}
