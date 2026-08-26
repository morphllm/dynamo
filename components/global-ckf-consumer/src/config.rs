// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Debug, Clone, Parser)]
pub struct Config {
    #[arg(long = "relay", required = true, value_parser = RelayConfig::from_str)]
    pub relays: Vec<RelayConfig>,
    #[arg(long, default_value = "/tls/tls.crt")]
    pub tls_cert: PathBuf,
    #[arg(long, default_value = "/tls/tls.key")]
    pub tls_key: PathBuf,
    #[arg(long, default_value = "/tls/ca.crt")]
    pub tls_ca: PathBuf,
    #[arg(long, default_value_t = 10)]
    pub handshake_timeout_seconds: u64,
    #[arg(long, default_value = "0.0.0.0:8095")]
    pub listen_address: SocketAddr,
    #[arg(long, default_value = "0.0.0.0:9090")]
    pub metrics_listen_address: SocketAddr,
    #[arg(long, default_value = "morph-global-ckf-consumer")]
    pub subscriber_id: String,
    #[arg(long, default_value_t = 45)]
    pub freshness_timeout_seconds: u64,
    #[arg(long, default_value_t = 16_384)]
    pub max_query_blocks: usize,
    /// Region-keyed dispatch targets, `region=https://url`. Setting any
    /// target turns on the dispatch endpoint; the region keys must match the
    /// configured relay names so every routable decision has a destination.
    #[arg(long = "dispatch-target", value_parser = DispatchTarget::from_str)]
    pub dispatch_targets: Vec<DispatchTarget>,
    /// SPKI PEM Ed25519 public key used to verify envelopes before dispatch.
    #[arg(long)]
    pub dispatch_envelope_public_key: Option<PathBuf>,
    /// Key identifier the verified envelopes must carry.
    #[arg(long)]
    pub dispatch_envelope_key_id: Option<String>,
    /// Optional client certificate and key (PEM) for mTLS toward targets.
    #[arg(long, requires = "dispatch_client_key")]
    pub dispatch_client_cert: Option<PathBuf>,
    #[arg(long, requires = "dispatch_client_cert")]
    pub dispatch_client_key: Option<PathBuf>,
    /// Optional additional CA bundle (PEM) trusted for target certificates.
    #[arg(long)]
    pub dispatch_ca: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    pub dispatch_connect_timeout_seconds: u64,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.handshake_timeout_seconds == 0 {
            bail!("handshake timeout must be greater than zero");
        }
        if self.freshness_timeout_seconds == 0 {
            bail!("freshness timeout must be greater than zero");
        }
        if self.max_query_blocks == 0 {
            bail!("maximum query blocks must be greater than zero");
        }
        validate_text("subscriber ID", &self.subscriber_id)?;
        let mut names = HashSet::new();
        let mut dc_ids = HashSet::new();
        for relay in &self.relays {
            if !names.insert(relay.name.as_str()) {
                bail!("relay name {:?} is duplicated", relay.name);
            }
            if !dc_ids.insert(relay.expected_dc_id) {
                bail!("expected DC ID {} is duplicated", relay.expected_dc_id);
            }
        }
        if !self.dispatch_targets.is_empty() {
            if self.dispatch_envelope_public_key.is_none()
                || self.dispatch_envelope_key_id.is_none()
            {
                bail!(
                    "dispatch targets require --dispatch-envelope-public-key and \
                     --dispatch-envelope-key-id; the dispatcher never forwards an \
                     envelope it cannot verify"
                );
            }
            if self.dispatch_connect_timeout_seconds == 0 {
                bail!("dispatch connect timeout must be greater than zero");
            }
            let mut regions = HashSet::new();
            for target in &self.dispatch_targets {
                if !regions.insert(target.region.as_str()) {
                    bail!("dispatch region {:?} is duplicated", target.region);
                }
                if !names.contains(target.region.as_str()) {
                    bail!(
                        "dispatch region {:?} does not match any configured relay name",
                        target.region
                    );
                }
            }
            for relay in &self.relays {
                if !regions.contains(relay.name.as_str()) {
                    bail!(
                        "relay {:?} has no dispatch target; every routable decision \
                         needs a destination",
                        relay.name
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchTarget {
    pub region: String,
    pub url: String,
}

impl FromStr for DispatchTarget {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (region, url) = value
            .split_once('=')
            .context("dispatch target must be region=https://url")?;
        validate_text("dispatch region", region)?;
        validate_text("dispatch URL", url)?;
        if !url.starts_with("https://") {
            bail!("dispatch target URL must use HTTPS");
        }
        Ok(Self {
            region: region.to_string(),
            url: url.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayConfig {
    pub name: String,
    pub address: String,
    pub tls_server_name: String,
    pub expected_dc_id: u64,
}

impl FromStr for RelayConfig {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let fields: Vec<_> = value.split(',').collect();
        let [name, address, tls_server_name, expected_dc_id] = fields.as_slice() else {
            bail!("relay must be name,host:port,tls-server-name,expected-dc-id");
        };
        validate_text("relay name", name)?;
        validate_text("relay address", address)?;
        validate_text("TLS server name", tls_server_name)?;
        if address.starts_with("http://") || address.starts_with("https://") {
            bail!("relay address must be host:port without a URL scheme");
        }
        let (_, port) = address
            .rsplit_once(':')
            .context("relay address must contain host:port")?;
        let port: u16 = port.parse().context("relay port must be a valid u16")?;
        if port == 0 {
            bail!("relay port must be greater than zero");
        }
        let expected_dc_id = expected_dc_id
            .parse::<u64>()
            .context("expected DC ID must be a valid u64")?;
        Ok(Self {
            name: (*name).to_string(),
            address: (*address).to_string(),
            tls_server_name: (*tls_server_name).to_string(),
            expected_dc_id,
        })
    }
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{field} must not be empty or contain surrounding whitespace");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_config_parses_exact_identity_boundary() {
        let relay =
            RelayConfig::from_str("ue5,relay.internal:4443,relay.dynamo.svc.cluster.local,17")
                .unwrap();
        assert_eq!(relay.name, "ue5");
        assert_eq!(relay.address, "relay.internal:4443");
        assert_eq!(relay.tls_server_name, "relay.dynamo.svc.cluster.local");
        assert_eq!(relay.expected_dc_id, 17);
    }

    #[test]
    fn relay_config_rejects_ambiguous_or_insecure_addresses() {
        assert!(RelayConfig::from_str("ue5,https://relay:4443,relay,17").is_err());
        assert!(RelayConfig::from_str("ue5,relay,relay,17").is_err());
        assert!(RelayConfig::from_str("ue5,relay:0,relay,17").is_err());
        assert!(RelayConfig::from_str("ue5,relay:4443,relay").is_err());
    }

    fn config(relays: Vec<RelayConfig>) -> Config {
        Config {
            relays,
            tls_cert: "cert".into(),
            tls_key: "key".into(),
            tls_ca: "ca".into(),
            handshake_timeout_seconds: 10,
            listen_address: "127.0.0.1:8095".parse().unwrap(),
            metrics_listen_address: "127.0.0.1:9090".parse().unwrap(),
            subscriber_id: "consumer".into(),
            freshness_timeout_seconds: 45,
            max_query_blocks: 16_384,
            dispatch_targets: vec![],
            dispatch_envelope_public_key: None,
            dispatch_envelope_key_id: None,
            dispatch_client_cert: None,
            dispatch_client_key: None,
            dispatch_ca: None,
            dispatch_connect_timeout_seconds: 5,
        }
    }

    #[test]
    fn config_rejects_duplicate_names_and_dc_ids() {
        let relay = RelayConfig::from_str("ue5,relay:4443,relay,17").unwrap();
        let mut config = config(vec![relay.clone(), relay]);
        assert!(config.validate().is_err());
        config.relays[1].name = "uw1".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn dispatch_target_parses_and_requires_https() {
        let target = DispatchTarget::from_str("ue5=https://pod-proxy.example/global").unwrap();
        assert_eq!(target.region, "ue5");
        assert_eq!(target.url, "https://pod-proxy.example/global");
        assert!(DispatchTarget::from_str("ue5=http://pod-proxy.example").is_err());
        assert!(DispatchTarget::from_str("https://pod-proxy.example").is_err());
    }

    #[test]
    fn dispatch_configuration_is_all_or_nothing_and_covers_every_relay() {
        let ue5 = RelayConfig::from_str("ue5,relay:4443,relay,17").unwrap();
        let uw1 = RelayConfig::from_str("uw1,relay2:4443,relay2,18").unwrap();
        let mut config = config(vec![ue5, uw1]);
        assert!(config.validate().is_ok());

        // Targets without a verification key are rejected.
        config.dispatch_targets = vec![
            DispatchTarget::from_str("ue5=https://ue5.example/global").unwrap(),
            DispatchTarget::from_str("uw1=https://uw1.example/global").unwrap(),
        ];
        assert!(config.validate().is_err());

        config.dispatch_envelope_public_key = Some("key.pem".into());
        config.dispatch_envelope_key_id = Some("2026-08".into());
        assert!(config.validate().is_ok());

        // A relay without a destination or an unknown region is rejected.
        let missing = config.dispatch_targets.split_off(1);
        assert!(config.validate().is_err());
        config.dispatch_targets.extend(missing);
        config
            .dispatch_targets
            .push(DispatchTarget::from_str("nowhere=https://n.example").unwrap());
        assert!(config.validate().is_err());
    }
}
