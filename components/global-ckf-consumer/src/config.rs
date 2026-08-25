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
        Ok(())
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

    #[test]
    fn config_rejects_duplicate_names_and_dc_ids() {
        let relay = RelayConfig::from_str("ue5,relay:4443,relay,17").unwrap();
        let mut config = Config {
            relays: vec![relay.clone(), relay],
            tls_cert: "cert".into(),
            tls_key: "key".into(),
            tls_ca: "ca".into(),
            handshake_timeout_seconds: 10,
            listen_address: "127.0.0.1:8095".parse().unwrap(),
            metrics_listen_address: "127.0.0.1:9090".parse().unwrap(),
            subscriber_id: "consumer".into(),
            freshness_timeout_seconds: 45,
            max_query_blocks: 16_384,
        };
        assert!(config.validate().is_err());
        config.relays[1].name = "uw1".into();
        assert!(config.validate().is_err());
    }
}
