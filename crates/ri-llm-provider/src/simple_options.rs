use crate::{Context, Model, SimpleStreamOptions, ThinkingBudgets, ThinkingLevel, estimate};

const CONTEXT_SAFETY_TOKENS: u64 = 4_096;
const MIN_MAX_TOKENS: u64 = 1;
const MIN_OUTPUT_TOKENS_WITH_THINKING: u64 = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingTokenAdjustment {
    pub max_tokens: u64,
    pub thinking_budget: u64,
}

pub fn clamp_max_tokens_to_context(model: &Model, context: &Context, max_tokens: u64) -> u64 {
    if model.context_window == 0 {
        return max_tokens.max(MIN_MAX_TOKENS);
    }
    let available = model.context_window as i128
        - estimate::estimate_context_tokens(context).tokens as i128
        - CONTEXT_SAFETY_TOKENS as i128;
    max_tokens.min(available.max(MIN_MAX_TOKENS as i128) as u64)
}

pub fn apply_simple_stream_defaults(
    model: &Model,
    context: &Context,
    mut options: SimpleStreamOptions,
) -> SimpleStreamOptions {
    let base_max_tokens = options.stream.max_tokens.unwrap_or(model.max_tokens);
    let clamped = clamp_max_tokens_to_context(model, context, base_max_tokens);
    // A zero result only happens for models without a max-token cap; pi's
    // falsy-number handling omits the cap from payloads, which `None` mirrors.
    options.stream.max_tokens = (clamped > 0).then_some(clamped);
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
