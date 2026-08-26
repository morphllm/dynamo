// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared decision wire contract. The frontend client owns the canonical types.

pub use dynamo_llm::global_routing::{
    DecisionOutcome, QueryRole, TokenDecisionRequest, TokenDecisionResponse,
};
