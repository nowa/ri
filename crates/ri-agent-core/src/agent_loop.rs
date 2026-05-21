use crate::types::{
    AgentContext, AgentContextTransformer, AgentEvent, AgentEventSink, AgentLoopTurnUpdate,
    AgentMessage, AgentMessageConverter, AgentNextTurnContext, AgentNextTurnPreparer,
    AgentQueuedMessageProvider, AgentShouldStopAfterTurn, AgentStreamProvider, AgentToolCallHook,
    AgentToolCallHookContext, AgentToolResult, AgentToolResultContent, AgentToolResultHook,
    AgentToolResultHookContext, AgentToolUpdateCallback, ToolExecutionMode, assistant_tool_calls,
};
use futures::{StreamExt, stream::FuturesUnordered};
use ri_llm_provider::{
    AssistantMessage, Model, SimpleStreamOptions, StopReason, ToolCall, ToolResultContent,
    ToolResultMessage, Usage, now_millis, stream_simple,
};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub stream_options: SimpleStreamOptions,
    pub tool_call_hooks: Vec<Arc<dyn AgentToolCallHook>>,
    pub tool_result_hooks: Vec<Arc<dyn AgentToolResultHook>>,
    pub transform_context: Option<Arc<dyn AgentContextTransformer>>,
    pub convert_to_llm: Option<Arc<dyn AgentMessageConverter>>,
    pub stream_provider: Option<Arc<dyn AgentStreamProvider>>,
    pub prepare_next_turn: Option<Arc<dyn AgentNextTurnPreparer>>,
    pub should_stop_after_turn: Option<Arc<dyn AgentShouldStopAfterTurn>>,
    pub queued_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    pub follow_up_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    pub event_sink: Option<Arc<dyn AgentEventSink>>,
    pub skip_initial_queued_message_poll: bool,
    pub tool_execution: ToolExecutionMode,
    pub max_turns: usize,
}

impl AgentLoopConfig {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            stream_options: SimpleStreamOptions::default(),
            tool_call_hooks: Vec::new(),
            tool_result_hooks: Vec::new(),
            transform_context: None,
            convert_to_llm: None,
            stream_provider: None,
            prepare_next_turn: None,
            should_stop_after_turn: None,
            queued_message_provider: None,
            follow_up_message_provider: None,
            event_sink: None,
            skip_initial_queued_message_poll: false,
            tool_execution: ToolExecutionMode::Parallel,
            max_turns: 16,
        }
    }
}

struct TurnOutcome {
    messages: Vec<AgentMessage>,
    tool_results: Vec<ToolResultMessage>,
    terminate: bool,
}

struct PreparedToolCall {
    index: usize,
    tool_call_id: String,
    tool_name: String,
    args: serde_json::Value,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    context: AgentContext,
    tool: crate::types::AgentTool,
}

#[derive(Clone)]
struct ToolExecutionOutcome {
    index: usize,
    tool_call_id: String,
    tool_name: String,
    result: crate::types::AgentToolResult,
    is_error: bool,
    update_events: Vec<AgentEvent>,
}

impl std::fmt::Debug for dyn AgentToolCallHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentToolCallHook")
    }
}

impl std::fmt::Debug for dyn AgentStreamProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentStreamProvider")
    }
}

impl std::fmt::Debug for dyn AgentToolResultHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentToolResultHook")
    }
}

impl std::fmt::Debug for dyn AgentContextTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentContextTransformer")
    }
}

impl std::fmt::Debug for dyn AgentMessageConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentMessageConverter")
    }
}

impl std::fmt::Debug for dyn AgentNextTurnPreparer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentNextTurnPreparer")
    }
}

impl std::fmt::Debug for dyn AgentShouldStopAfterTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentShouldStopAfterTurn")
    }
}

impl std::fmt::Debug for dyn AgentQueuedMessageProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentQueuedMessageProvider")
    }
}

impl std::fmt::Debug for dyn AgentEventSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AgentEventSink")
    }
}

async fn record_event(events: &mut Vec<AgentEvent>, config: &AgentLoopConfig, event: AgentEvent) {
    if let Some(sink) = &config.event_sink {
        sink.on_event(&event).await;
    }
    events.push(event);
}

pub async fn agent_loop_prompt(
    context: AgentContext,
    prompt: impl Into<String>,
    config: AgentLoopConfig,
) -> Result<(Vec<AgentMessage>, Vec<AgentEvent>), String> {
    let user_message = AgentMessage::User(ri_llm_provider::UserMessage::text(prompt.into()));
    agent_loop_prompt_messages(context, vec![user_message], config).await
}

pub async fn agent_loop_prompt_messages<M>(
    mut context: AgentContext,
    prompt_messages: Vec<M>,
    config: AgentLoopConfig,
) -> Result<(Vec<AgentMessage>, Vec<AgentEvent>), String>
where
    M: Into<AgentMessage>,
{
    let prompt_messages = prompt_messages
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>();
    let original_len = context.messages.len();
    context.messages.extend(prompt_messages.iter().cloned());
    let mut events = Vec::new();
    record_event(&mut events, &config, AgentEvent::AgentStart).await;
    record_event(&mut events, &config, AgentEvent::TurnStart).await;
    for message in &prompt_messages {
        record_event(
            &mut events,
            &config,
            AgentEvent::MessageStart {
                message: message.clone(),
            },
        )
        .await;
        record_event(
            &mut events,
            &config,
            AgentEvent::MessageEnd {
                message: message.clone(),
            },
        )
        .await;
    }
    let mut new_messages = prompt_messages;
    match run_until_done(&mut context, &config, &mut events, &new_messages).await {
        Ok(messages) => new_messages.extend(messages),
        Err(error) => {
            new_messages = finish_agent_loop_with_error(
                &mut context,
                &config,
                &mut events,
                original_len,
                error,
            )
            .await;
        }
    }
    record_event(
        &mut events,
        &config,
        AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        },
    )
    .await;
    Ok((new_messages, events))
}

pub async fn agent_loop_continue(
    mut context: AgentContext,
    config: AgentLoopConfig,
) -> Result<(Vec<AgentMessage>, Vec<AgentEvent>), String> {
    if context.messages.is_empty() {
        return Err("Cannot continue without existing messages".to_owned());
    }
    if matches!(context.messages.last(), Some(AgentMessage::Assistant(_))) {
        return Err("Cannot continue from message role: assistant".to_owned());
    }
    let original_len = context.messages.len();
    let mut events = Vec::new();
    record_event(&mut events, &config, AgentEvent::AgentStart).await;
    record_event(&mut events, &config, AgentEvent::TurnStart).await;
    let new_messages = match run_until_done(&mut context, &config, &mut events, &[]).await {
        Ok(messages) => messages,
        Err(error) => {
            finish_agent_loop_with_error(&mut context, &config, &mut events, original_len, error)
                .await
        }
    };
    record_event(
        &mut events,
        &config,
        AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        },
    )
    .await;
    Ok((new_messages, events))
}

async fn finish_agent_loop_with_error(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
    original_len: usize,
    error: String,
) -> Vec<AgentMessage> {
    let assistant = AssistantMessage {
        content: Vec::new(),
        api: config.model.api.clone(),
        provider: config.model.provider.clone(),
        model: config.model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Error,
        error_message: Some(error),
        timestamp: now_millis(),
    };
    let assistant_message = AgentMessage::Assistant(assistant);
    record_event(
        events,
        config,
        AgentEvent::MessageStart {
            message: assistant_message.clone(),
        },
    )
    .await;
    context.messages.push(assistant_message.clone());
    record_event(
        events,
        config,
        AgentEvent::MessageEnd {
            message: assistant_message.clone(),
        },
    )
    .await;
    record_event(
        events,
        config,
        AgentEvent::TurnEnd {
            message: assistant_message,
            tool_results: Vec::new(),
        },
    )
    .await;
    let new_messages = context.messages[original_len..].to_vec();
    new_messages
}

async fn run_until_done(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
    initial_new_messages: &[AgentMessage],
) -> Result<Vec<AgentMessage>, String> {
    let mut all_messages = Vec::new();
    let mut hook_new_messages = initial_new_messages.to_vec();
    let mut active_config = config.clone();
    let max_turns = config.max_turns.max(1);
    if !active_config.skip_initial_queued_message_poll {
        let queued_messages = inject_queued_messages(context, &active_config, events).await?;
        hook_new_messages.extend(queued_messages.clone());
        all_messages.extend(queued_messages);
    }
    active_config.skip_initial_queued_message_poll = false;
    for turn_index in 0..max_turns {
        if turn_index > 0 {
            record_event(events, &active_config, AgentEvent::TurnStart).await;
        }
        let outcome = run_one_turn(context, &active_config, events).await?;
        let should_continue = !outcome.terminate && !outcome.tool_results.is_empty();
        hook_new_messages.extend(outcome.messages.clone());
        all_messages.extend(outcome.messages.clone());
        prepare_next_turn(context, &mut active_config, &outcome, &hook_new_messages).await?;
        if should_stop_after_turn(context, &active_config, &outcome, &hook_new_messages).await? {
            return Ok(all_messages);
        }
        let queued_messages = inject_queued_messages(context, &active_config, events).await?;
        let has_queued_messages = !queued_messages.is_empty();
        hook_new_messages.extend(queued_messages.clone());
        all_messages.extend(queued_messages);
        if should_continue || has_queued_messages {
            continue;
        }
        let follow_up_messages = inject_follow_up_messages(context, &active_config, events).await?;
        if follow_up_messages.is_empty() {
            return Ok(all_messages);
        }
        hook_new_messages.extend(follow_up_messages.clone());
        all_messages.extend(follow_up_messages);
    }
    Err(format!(
        "Agent loop exceeded maximum tool continuation turns: {max_turns}"
    ))
}

async fn prepare_next_turn(
    context: &mut AgentContext,
    config: &mut AgentLoopConfig,
    outcome: &TurnOutcome,
    new_messages: &[AgentMessage],
) -> Result<(), String> {
    let Some(preparer) = &config.prepare_next_turn else {
        return Ok(());
    };
    let Some(AgentMessage::Assistant(message)) = outcome.messages.first() else {
        return Ok(());
    };
    let turn_update = preparer
        .prepare_next_turn(AgentNextTurnContext {
            message: message.clone(),
            tool_results: outcome.tool_results.clone(),
            context: context.clone(),
            new_messages: new_messages.to_vec(),
        })
        .await?;
    apply_turn_update(context, config, turn_update);
    Ok(())
}

fn apply_turn_update(
    context: &mut AgentContext,
    config: &mut AgentLoopConfig,
    turn_update: Option<AgentLoopTurnUpdate>,
) {
    let Some(turn_update) = turn_update else {
        return;
    };
    if let Some(next_context) = turn_update.context {
        *context = next_context;
    }
    if let Some(model) = turn_update.model {
        config.model = model;
    }
    if let Some(stream_options) = turn_update.stream_options {
        config.stream_options = stream_options;
    }
    if let Some(thinking_level) = turn_update.thinking_level {
        config.stream_options.reasoning = match thinking_level {
            ri_llm_provider::ThinkingLevel::Off => None,
            level => Some(level),
        };
    }
}

async fn should_stop_after_turn(
    context: &AgentContext,
    config: &AgentLoopConfig,
    outcome: &TurnOutcome,
    new_messages: &[AgentMessage],
) -> Result<bool, String> {
    let Some(hook) = &config.should_stop_after_turn else {
        return Ok(false);
    };
    let Some(AgentMessage::Assistant(message)) = outcome.messages.first() else {
        return Ok(false);
    };
    hook.should_stop_after_turn(AgentNextTurnContext {
        message: message.clone(),
        tool_results: outcome.tool_results.clone(),
        context: context.clone(),
        new_messages: new_messages.to_vec(),
    })
    .await
}

async fn inject_queued_messages(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
) -> Result<Vec<AgentMessage>, String> {
    inject_messages_from_provider(
        context,
        config.queued_message_provider.as_ref(),
        config,
        events,
    )
    .await
}

async fn inject_follow_up_messages(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
) -> Result<Vec<AgentMessage>, String> {
    inject_messages_from_provider(
        context,
        config.follow_up_message_provider.as_ref(),
        config,
        events,
    )
    .await
}

async fn inject_messages_from_provider(
    context: &mut AgentContext,
    provider: Option<&Arc<dyn AgentQueuedMessageProvider>>,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
) -> Result<Vec<AgentMessage>, String> {
    let Some(provider) = provider else {
        return Ok(Vec::new());
    };
    let queued_messages = provider.get_queued_messages().await?;
    for message in &queued_messages {
        record_event(
            events,
            config,
            AgentEvent::MessageStart {
                message: message.clone(),
            },
        )
        .await;
        context.messages.push(message.clone());
        record_event(
            events,
            config,
            AgentEvent::MessageEnd {
                message: message.clone(),
            },
        )
        .await;
    }
    Ok(queued_messages)
}

async fn run_one_turn(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    events: &mut Vec<AgentEvent>,
) -> Result<TurnOutcome, String> {
    let llm_context = build_llm_context(context, config).await?;
    let stream = match &config.stream_provider {
        Some(provider) => {
            provider.stream(&config.model, llm_context, config.stream_options.clone())?
        }
        None => stream_simple(&config.model, llm_context, config.stream_options.clone())
            .map_err(|error| error.to_string())?,
    };

    let mut stream = Box::pin(stream);
    let mut final_message: Option<AssistantMessage> = None;
    let mut emitted_message_start = false;
    while let Some(event) = stream.next().await {
        if let Some(message) = current_partial_message(&event) {
            match &event {
                ri_llm_provider::AssistantMessageEvent::Start { .. } => {
                    emitted_message_start = true;
                    record_event(
                        events,
                        config,
                        AgentEvent::MessageStart {
                            message: AgentMessage::Assistant(message.clone()),
                        },
                    )
                    .await;
                }
                ri_llm_provider::AssistantMessageEvent::Done { .. }
                | ri_llm_provider::AssistantMessageEvent::Error { .. } => {
                    final_message = Some(message.clone());
                }
                _ => {
                    record_event(
                        events,
                        config,
                        AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(message.clone()),
                            assistant_message_event: event,
                        },
                    )
                    .await;
                }
            }
        }
    }

    let assistant =
        final_message.ok_or_else(|| "Provider stream ended without final message".to_owned())?;
    let assistant_message = AgentMessage::Assistant(assistant.clone());
    let mut new_messages = Vec::new();
    if !emitted_message_start {
        record_event(
            events,
            config,
            AgentEvent::MessageStart {
                message: assistant_message.clone(),
            },
        )
        .await;
    }
    context.messages.push(assistant_message.clone());
    new_messages.push(assistant_message.clone());
    record_event(
        events,
        config,
        AgentEvent::MessageEnd {
            message: assistant_message.clone(),
        },
    )
    .await;

    let mut tool_results = Vec::new();
    let mut terminate = false;
    if assistant.stop_reason == StopReason::ToolUse {
        let mut prepared_calls = Vec::new();
        let mut execution_outcomes = Vec::new();
        'tool_calls: for (index, tool_call) in assistant_tool_calls(&assistant).enumerate() {
            let mut args = serde_json::Value::Object(tool_call.arguments.clone());
            let Some(tool) = context
                .tools
                .iter()
                .find(|tool| tool.definition.name == tool_call.name)
            else {
                record_event(
                    events,
                    config,
                    AgentEvent::ToolExecutionStart {
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        args,
                    },
                )
                .await;
                execution_outcomes.push(error_tool_execution_outcome(
                    index,
                    tool_call.id.clone(),
                    tool_call.name.clone(),
                    format!("Tool {} not found", tool_call.name),
                ));
                continue 'tool_calls;
            };
            if let Some(preparer) = &tool.argument_preparer {
                match preparer.prepare_arguments(args.clone()) {
                    Ok(prepared_args) => args = prepared_args,
                    Err(error) => {
                        record_event(
                            events,
                            config,
                            AgentEvent::ToolExecutionStart {
                                tool_call_id: tool_call.id.clone(),
                                tool_name: tool_call.name.clone(),
                                args,
                            },
                        )
                        .await;
                        execution_outcomes.push(error_tool_execution_outcome(
                            index,
                            tool_call.id.clone(),
                            tool_call.name.clone(),
                            error,
                        ));
                        continue 'tool_calls;
                    }
                }
            }
            let mut hook_context = AgentToolCallHookContext {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                input: args.clone(),
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                context: context.clone(),
            };
            for hook in &config.tool_call_hooks {
                match hook.on_tool_call(hook_context.clone()).await {
                    Ok(Some(result)) => {
                        if let Some(replacement) = result.input {
                            args = replacement;
                            hook_context.input = args.clone();
                        }
                        if result.block {
                            record_event(
                                events,
                                config,
                                AgentEvent::ToolExecutionStart {
                                    tool_call_id: tool_call.id.clone(),
                                    tool_name: tool_call.name.clone(),
                                    args,
                                },
                            )
                            .await;
                            execution_outcomes.push(error_tool_execution_outcome(
                                index,
                                tool_call.id.clone(),
                                tool_call.name.clone(),
                                result
                                    .reason
                                    .unwrap_or_else(|| "Tool execution was blocked".to_owned()),
                            ));
                            continue 'tool_calls;
                        }
                    }
                    Ok(None) => {}
                    Err(error) => {
                        record_event(
                            events,
                            config,
                            AgentEvent::ToolExecutionStart {
                                tool_call_id: tool_call.id.clone(),
                                tool_name: tool_call.name.clone(),
                                args,
                            },
                        )
                        .await;
                        execution_outcomes.push(error_tool_execution_outcome(
                            index,
                            tool_call.id.clone(),
                            tool_call.name.clone(),
                            error,
                        ));
                        continue 'tool_calls;
                    }
                }
            }
            prepared_calls.push(PreparedToolCall {
                index,
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args,
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                context: context.clone(),
                tool: tool.clone(),
            });
        }

        for call in &prepared_calls {
            record_event(
                events,
                config,
                AgentEvent::ToolExecutionStart {
                    tool_call_id: call.tool_call_id.clone(),
                    tool_name: call.tool_name.clone(),
                    args: call.args.clone(),
                },
            )
            .await;
        }

        execution_outcomes.extend(execute_tool_calls(prepared_calls, config, events).await?);
        terminate |= !execution_outcomes.is_empty()
            && execution_outcomes
                .iter()
                .all(|outcome| outcome.result.terminate);
        for outcome in &execution_outcomes {
            events.extend(outcome.update_events.iter().cloned());
            record_event(
                events,
                config,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id: outcome.tool_call_id.clone(),
                    tool_name: outcome.tool_name.clone(),
                    result: outcome.result.clone(),
                    is_error: outcome.is_error,
                },
            )
            .await;
        }

        let mut source_order_outcomes = execute_tool_calls_in_source_order(execution_outcomes);
        for outcome in source_order_outcomes.drain(..).flatten() {
            let message = tool_result_message_from_outcome(outcome);
            let tool_result_message = AgentMessage::ToolResult(message.clone());
            record_event(
                events,
                config,
                AgentEvent::MessageStart {
                    message: tool_result_message.clone(),
                },
            )
            .await;
            context.messages.push(tool_result_message.clone());
            new_messages.push(tool_result_message.clone());
            record_event(
                events,
                config,
                AgentEvent::MessageEnd {
                    message: tool_result_message,
                },
            )
            .await;
            tool_results.push(message);
        }
    }

    record_event(
        events,
        config,
        AgentEvent::TurnEnd {
            message: assistant_message.clone(),
            tool_results: tool_results.clone(),
        },
    )
    .await;

    Ok(TurnOutcome {
        messages: new_messages,
        tool_results,
        terminate,
    })
}

async fn execute_tool_calls(
    prepared_calls: Vec<PreparedToolCall>,
    config: &AgentLoopConfig,
    _events: &mut Vec<AgentEvent>,
) -> Result<Vec<ToolExecutionOutcome>, String> {
    let use_parallel = config.tool_execution == ToolExecutionMode::Parallel
        && prepared_calls
            .iter()
            .all(|call| call.tool.execution_mode != Some(ToolExecutionMode::Sequential));
    if !use_parallel {
        let mut outcomes = Vec::new();
        for call in prepared_calls {
            outcomes.push(
                execute_prepared_tool_call(
                    call,
                    config.tool_result_hooks.clone(),
                    config.event_sink.clone(),
                )
                .await?,
            );
        }
        return Ok(outcomes);
    }

    let mut pending = FuturesUnordered::new();
    for call in prepared_calls {
        let hooks = config.tool_result_hooks.clone();
        let event_sink = config.event_sink.clone();
        pending.push(async move { execute_prepared_tool_call(call, hooks, event_sink).await });
    }
    let mut outcomes = Vec::new();
    while let Some(outcome) = pending.next().await {
        outcomes.push(outcome?);
    }
    Ok(outcomes)
}

async fn execute_prepared_tool_call(
    call: PreparedToolCall,
    tool_result_hooks: Vec<Arc<dyn AgentToolResultHook>>,
    event_sink: Option<Arc<dyn AgentEventSink>>,
) -> Result<ToolExecutionOutcome, String> {
    let result_input = call.args.clone();
    let update_args = call.args.clone();
    let update_tool_call_id = call.tool_call_id.clone();
    let update_tool_name = call.tool_name.clone();
    let update_events = Arc::new(Mutex::new(Vec::new()));
    let update_events_ref = update_events.clone();
    let on_update: AgentToolUpdateCallback = Arc::new(move |partial_result: AgentToolResult| {
        let event_sink = event_sink.clone();
        let update_events = update_events_ref.clone();
        let event = AgentEvent::ToolExecutionUpdate {
            tool_call_id: update_tool_call_id.clone(),
            tool_name: update_tool_name.clone(),
            args: update_args.clone(),
            partial_result,
        };
        Box::pin(async move {
            if let Some(sink) = event_sink {
                sink.on_event(&event).await;
            }
            update_events
                .lock()
                .expect("tool update events")
                .push(event);
        })
    });
    let mut is_error = false;
    let mut result = match call
        .tool
        .executor
        .execute_with_updates(&call.tool_call_id, call.args, on_update)
        .await
    {
        Ok(result) => result,
        Err(error) => {
            is_error = true;
            error_tool_result(error)
        }
    };
    let mut hook_context = AgentToolResultHookContext {
        tool_call_id: call.tool_call_id.clone(),
        tool_name: call.tool_name.clone(),
        input: result_input.clone(),
        result: result.clone(),
        is_error,
        assistant_message: call.assistant_message.clone(),
        tool_call: call.tool_call.clone(),
        context: call.context.clone(),
    };
    for hook in &tool_result_hooks {
        match hook.on_tool_result(hook_context.clone()).await {
            Ok(Some(replacement)) => {
                if let Some(replacement_result) = replacement.result {
                    result = replacement_result;
                    hook_context.result = result.clone();
                }
                if let Some(replacement_is_error) = replacement.is_error {
                    is_error = replacement_is_error;
                    hook_context.is_error = is_error;
                }
            }
            Ok(None) => {}
            Err(error) => {
                is_error = true;
                result = error_tool_result(error);
                break;
            }
        }
    }
    Ok(ToolExecutionOutcome {
        index: call.index,
        tool_call_id: call.tool_call_id,
        tool_name: call.tool_name,
        result,
        is_error,
        update_events: update_events.lock().expect("tool update events").clone(),
    })
}

fn error_tool_execution_outcome(
    index: usize,
    tool_call_id: String,
    tool_name: String,
    error: impl Into<String>,
) -> ToolExecutionOutcome {
    ToolExecutionOutcome {
        index,
        tool_call_id,
        tool_name,
        result: error_tool_result(error),
        is_error: true,
        update_events: Vec::new(),
    }
}

fn error_tool_result(error: impl Into<String>) -> crate::types::AgentToolResult {
    crate::types::AgentToolResult {
        content: vec![AgentToolResultContent::Text(
            ri_llm_provider::TextContent::new(error),
        )],
        details: Some(serde_json::Value::Object(Default::default())),
        terminate: false,
    }
}

fn execute_tool_calls_in_source_order(
    outcomes: Vec<ToolExecutionOutcome>,
) -> Vec<Option<ToolExecutionOutcome>> {
    let len = outcomes
        .iter()
        .map(|outcome| outcome.index)
        .max()
        .map(|index| index + 1)
        .unwrap_or(0);
    let mut ordered = vec![None; len];
    for outcome in outcomes {
        let index = outcome.index;
        if let Some(slot) = ordered.get_mut(index) {
            *slot = Some(outcome);
        }
    }
    ordered
}

fn tool_result_message_from_outcome(outcome: ToolExecutionOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: outcome.tool_call_id,
        tool_name: outcome.tool_name,
        content: outcome
            .result
            .content
            .into_iter()
            .map(|content| match content {
                AgentToolResultContent::Text(text) => ToolResultContent::Text(text),
                AgentToolResultContent::Image(image) => ToolResultContent::Image(image),
            })
            .collect(),
        details: outcome.result.details,
        is_error: outcome.is_error,
        timestamp: ri_llm_provider::now_millis(),
    }
}

async fn build_llm_context(
    context: &AgentContext,
    config: &AgentLoopConfig,
) -> Result<ri_llm_provider::Context, String> {
    let messages = if let Some(transformer) = &config.transform_context {
        transformer
            .transform_context(context.messages.clone())
            .await?
    } else {
        context.messages.clone()
    };
    let llm_messages = if let Some(converter) = &config.convert_to_llm {
        converter.convert_to_llm(&messages)?
    } else {
        messages
            .iter()
            .filter_map(AgentMessage::to_llm_message)
            .collect()
    };
    Ok(ri_llm_provider::Context {
        system_prompt: (!context.system_prompt.is_empty()).then_some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: context
            .tools
            .iter()
            .map(|tool| tool.definition.clone())
            .collect(),
    })
}

fn current_partial_message(
    event: &ri_llm_provider::AssistantMessageEvent,
) -> Option<&AssistantMessage> {
    match event {
        ri_llm_provider::AssistantMessageEvent::Start { partial }
        | ri_llm_provider::AssistantMessageEvent::TextStart { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::TextDelta { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::TextEnd { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ThinkingStart { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ThinkingDelta { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ThinkingEnd { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ToolcallStart { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ToolcallDelta { partial, .. }
        | ri_llm_provider::AssistantMessageEvent::ToolcallEnd { partial, .. } => Some(partial),
        ri_llm_provider::AssistantMessageEvent::Done { message, .. } => Some(message),
        ri_llm_provider::AssistantMessageEvent::Error { error, .. } => Some(error),
    }
}
