use crate::{Model, SimpleStreamOptions, ThinkingBudgets, ThinkingLevel};

const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_000;
const CONTEXT_WINDOW_OUTPUT_TOLERANCE: u64 = 1_024;
const MIN_OUTPUT_TOKENS_WITH_THINKING: u64 = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingTokenAdjustment {
    pub max_tokens: u64,
    pub thinking_budget: u64,
}

pub fn default_simple_max_tokens(model: &Model) -> Option<u64> {
    if model.max_tokens == 0 {
        return None;
    }

    if model.max_tokens
        >= model
            .context_window
            .saturating_sub(CONTEXT_WINDOW_OUTPUT_TOLERANCE)
    {
        Some(model.max_tokens.min(DEFAULT_MAX_OUTPUT_TOKENS))
    } else {
        Some(model.max_tokens)
    }
}

pub fn apply_simple_stream_defaults(
    model: &Model,
    mut options: SimpleStreamOptions,
) -> SimpleStreamOptions {
    if options.stream.max_tokens.is_none() {
        options.stream.max_tokens = default_simple_max_tokens(model);
    }
    options
}

pub fn clamp_reasoning_for_budget(level: ThinkingLevel) -> ThinkingLevel {
    if level == ThinkingLevel::XHigh {
        ThinkingLevel::High
    } else {
        level
    }
}

pub fn default_thinking_budget(
    level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u64 {
    let level = clamp_reasoning_for_budget(level);
    match level {
        ThinkingLevel::Minimal => custom_budgets
            .and_then(|budget| budget.minimal)
            .unwrap_or(1_024),
        ThinkingLevel::Low => custom_budgets
            .and_then(|budget| budget.low)
            .unwrap_or(2_048),
        ThinkingLevel::Medium => custom_budgets
            .and_then(|budget| budget.medium)
            .unwrap_or(8_192),
        ThinkingLevel::High => custom_budgets
            .and_then(|budget| budget.high)
            .unwrap_or(16_384),
        ThinkingLevel::Off => 0,
        ThinkingLevel::XHigh => unreachable!("level was clamped"),
    }
}

pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: u64,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> ThinkingTokenAdjustment {
    let mut thinking_budget = default_thinking_budget(reasoning_level, custom_budgets);
    let max_tokens = base_max_tokens
        .saturating_add(thinking_budget)
        .min(model_max_tokens);

    if max_tokens <= thinking_budget {
        thinking_budget = max_tokens.saturating_sub(MIN_OUTPUT_TOKENS_WITH_THINKING);
    }

    ThinkingTokenAdjustment {
        max_tokens,
        thinking_budget,
    }
}
