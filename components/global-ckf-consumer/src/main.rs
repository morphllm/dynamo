// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use clap::Parser;
use global_ckf_consumer::api::{AppState, api_router, system_router};
use global_ckf_consumer::config::Config;
use global_ckf_consumer::coordinator;
use global_ckf_consumer::dispatch::{Dispatcher, dispatch_router};
use global_ckf_consumer::supervisor::spawn_relay_supervisors;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let config = Config::parse();
    config.validate()?;

    let events = spawn_relay_supervisors(&config);
    let state = AppState::default();
    let mut app = api_router(state.clone(), config.max_query_blocks);
    if let Some(dispatcher) = Dispatcher::from_config(&config, state.clone())? {
        app = app.merge(dispatch_router(dispatcher));
    }
    let api = tokio::net::TcpListener::bind(config.listen_address).await?;
    let system = tokio::net::TcpListener::bind(config.metrics_listen_address).await?;
    tokio::select! {
        result = coordinator::run(config.clone(), state.clone(), events) => result?,
        result = axum::serve(api, app) => result?,
        result = axum::serve(system, system_router(state)) => result?,
        _ = tokio::signal::ctrl_c() => {}
    }
    Ok(())
}
