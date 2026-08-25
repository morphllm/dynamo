// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::{Context, Result};
use dynamo_llm::kv_dc_relay::protocol::{
    KvEventRelayClient, RELAY_CONTRACT_MARKER, RelayIdentity, RelayInfoRequest,
    validate_protocol_envelope,
};
use tonic::codec::CompressionEncoding;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};

use crate::config::{Config, RelayConfig};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(20);
const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedRelay {
    pub name: String,
    pub expected_dc_id: u64,
    pub identity: RelayIdentity,
}

pub async fn connect_and_verify(
    relay: &RelayConfig,
    config: &Config,
) -> Result<(KvEventRelayClient<Channel>, VerifiedRelay)> {
    let cert = std::fs::read(&config.tls_cert).context("read client TLS certificate")?;
    let key = std::fs::read(&config.tls_key).context("read client TLS private key")?;
    let ca = std::fs::read(&config.tls_ca).context("read relay CA certificate")?;
    let tls = ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(ca))
        .identity(Identity::from_pem(cert, key))
        .domain_name(relay.tls_server_name.clone());
    let endpoint = Endpoint::from_shared(format!("https://{}", relay.address))?
        .connect_timeout(CONNECT_TIMEOUT)
        .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .keep_alive_while_idle(true)
        .tls_config(tls)?;
    let channel = endpoint
        .connect()
        .await
        .with_context(|| format!("connect relay {}", relay.name))?;
    let mut client = KvEventRelayClient::new(channel)
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd)
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES);
    let info = tokio::time::timeout(
        Duration::from_secs(config.handshake_timeout_seconds),
        client.get_relay_info(RelayInfoRequest {
            contract_marker: RELAY_CONTRACT_MARKER,
        }),
    )
    .await
    .with_context(|| format!("relay {} handshake timed out", relay.name))??
    .into_inner();
    validate_protocol_envelope(info.protocol_version, info.contract_marker)
        .with_context(|| format!("relay {} returned an incompatible protocol", relay.name))?;
    let identity = info
        .relay
        .with_context(|| format!("relay {} omitted its identity", relay.name))?;
    Ok((
        client,
        VerifiedRelay {
            name: relay.name.clone(),
            expected_dc_id: relay.expected_dc_id,
            identity,
        },
    ))
}

pub fn validate_stream_identity(
    expected: &RelayIdentity,
    actual: Option<&RelayIdentity>,
) -> Result<()> {
    let actual = actual.context("relay stream update omitted its identity")?;
    if actual != expected {
        anyhow::bail!("relay identity changed during the connection generation");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_identity_requires_complete_generation_match() {
        let expected = RelayIdentity {
            drt_instance_id: 7,
            relay_incarnation: 11,
        };
        assert!(validate_stream_identity(&expected, Some(&expected)).is_ok());
        assert!(validate_stream_identity(&expected, None).is_err());
        assert!(
            validate_stream_identity(
                &expected,
                Some(&RelayIdentity {
                    drt_instance_id: 7,
                    relay_incarnation: 12,
                })
            )
            .is_err()
        );
    }
}
