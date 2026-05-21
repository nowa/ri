use async_trait::async_trait;
use ri_agent_core::*;
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

fn context_with_model(_model: &Model) -> AgentContext {
    AgentContext {
        system_prompt: "You are helpful.".to_owned(),
        messages: Vec::new(),
        tools: Vec::new(),
    }
}

fn event_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

fn llm_role_of(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn role_of(message: &AgentMessage) -> &'static str {
    match message {
        AgentMessage::User(_) => "user",
        AgentMessage::Assistant(_) => "assistant",
        AgentMessage::ToolResult(_) => "toolResult",
        AgentMessage::Custom(_) => "custom",
    }
}

fn llm_text_of(message: &Message) -> Option<&str> {
    match message {
        Message::Assistant(assistant) => match assistant.content.first()? {
            AssistantContent::Text(text) => Some(&text.text),
            _ => None,
        },
        Message::ToolResult(result) => match result.content.first()? {
            ToolResultContent::Text(text) => Some(&text.text),
            _ => None,
        },
        _ => None,
    }
}

fn text_of(message: &AgentMessage) -> Option<&str> {
    match message {
        AgentMessage::Assistant(assistant) => match assistant.content.first()? {
            AssistantContent::Text(text) => Some(&text.text),
            _ => None,
        },
        AgentMessage::ToolResult(result) => match result.content.first()? {
            ToolResultContent::Text(text) => Some(&text.text),
            _ => None,
        },
        _ => None,
    }
}

struct EchoExecutor {
    executed: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentToolExecutor for EchoExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        self.executed.lock().expect("mutex").push(value.clone());
        Ok(AgentToolResult::text(format!("echoed: {value}")))
    }
}

struct RecordingValueExecutor {
    executed: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentToolExecutor for RecordingValueExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        self.executed.lock().expect("mutex").push(params.clone());
        Ok(AgentToolResult::text(format!(
            "echoed: {}",
            params.get("value").cloned().unwrap_or(Value::Null)
        )))
    }
}

struct FailingExecutor;

#[async_trait]
impl AgentToolExecutor for FailingExecutor {
    async fn execute(
        &self,
        _tool_call_id: &str,
        _params: Value,
    ) -> Result<AgentToolResult, String> {
        Err("boom".to_owned())
    }
}

struct UpdatingExecutor {
    update_observed_before_return: Arc<AtomicBool>,
}

#[async_trait]
impl AgentToolExecutor for UpdatingExecutor {
    async fn execute(
        &self,
        _tool_call_id: &str,
        _params: Value,
    ) -> Result<AgentToolResult, String> {
        panic!("agent loop should call execute_with_updates")
    }

    async fn execute_with_updates(
        &self,
        _tool_call_id: &str,
        params: Value,
        on_update: AgentToolUpdateCallback,
    ) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        on_update(AgentToolResult::text(format!("partial: {value}"))).await;
        self.update_observed_before_return
            .store(true, Ordering::SeqCst);
        Ok(AgentToolResult::text(format!("final: {value}")))
    }
}

struct RecordingApiKeyProvider {
    keys: Arc<Mutex<VecDeque<Option<String>>>>,
    providers: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentApiKeyProvider for RecordingApiKeyProvider {
    async fn get_api_key(&self, provider: &str) -> Result<Option<String>, String> {
        self.providers
            .lock()
            .expect("mutex")
            .push(provider.to_owned());
        Ok(self.keys.lock().expect("mutex").pop_front().flatten())
    }
}

struct RecordingOptionsStreamProvider {
    seen_options: Arc<Mutex<Vec<SimpleStreamOptions>>>,
    responses: Arc<Mutex<VecDeque<AssistantMessage>>>,
}

impl AgentStreamProvider for RecordingOptionsStreamProvider {
    fn stream(
        &self,
        _model: &Model,
        _context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, String> {
        self.seen_options.lock().expect("mutex").push(options);
        let message = self
            .responses
            .lock()
            .expect("mutex")
            .pop_front()
            .ok_or_else(|| "missing response".to_owned())?;
        let (sender, stream) = assistant_message_event_stream();
        sender.push(AssistantMessageEvent::Done {
            reason: message.stop_reason.clone(),
            message,
        });
        Ok(stream)
    }
}

struct EndOnlyStreamProvider {
    response: AssistantMessage,
}

impl AgentStreamProvider for EndOnlyStreamProvider {
    fn stream(
        &self,
        _model: &Model,
        _context: Context,
        _options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, String> {
        let (sender, stream) = assistant_message_event_stream();
        sender.end(self.response.clone());
        Ok(stream)
    }
}

struct ConditionalTerminateExecutor;

#[async_trait]
impl AgentToolExecutor for ConditionalTerminateExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let terminate = params
            .get("terminate")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(AgentToolResult {
            content: vec![AgentToolResultContent::Text(TextContent::new(format!(
                "echoed: {value}"
            )))],
            details: Some(json!({ "value": value })),
            terminate,
        })
    }
}

struct ParallelEchoExecutor {
    first_can_finish: Arc<tokio::sync::Notify>,
    first_resolved: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
}

#[async_trait]
impl AgentToolExecutor for ParallelEchoExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if value == "first" {
            let _ = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                self.first_can_finish.notified(),
            )
            .await;
            self.first_resolved.store(true, Ordering::SeqCst);
        }
        if value == "second" {
            if !self.first_resolved.load(Ordering::SeqCst) {
                self.parallel_observed.store(true, Ordering::SeqCst);
            }
            self.first_can_finish.notify_one();
        }
        Ok(AgentToolResult::text(format!("echoed: {value}")))
    }
}

struct SequentialProbeExecutor {
    first_resolved: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
}

#[async_trait]
impl AgentToolExecutor for SequentialProbeExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if value == "first" {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.first_resolved.store(true, Ordering::SeqCst);
        }
        if value == "second" && !self.first_resolved.load(Ordering::SeqCst) {
            self.parallel_observed.store(true, Ordering::SeqCst);
        }
        Ok(AgentToolResult::text(format!("echoed: {value}")))
    }
}

struct OrderedDelayExecutor {
    prefix: &'static str,
    order: Arc<Mutex<Vec<String>>>,
    delay_ms: u64,
}

#[async_trait]
impl AgentToolExecutor for OrderedDelayExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if self.delay_ms > 0 {
            self.order
                .lock()
                .expect("mutex")
                .push(format!("{}:start:{value}", self.prefix));
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            self.order
                .lock()
                .expect("mutex")
                .push(format!("{}:end:{value}", self.prefix));
        } else {
            self.order
                .lock()
                .expect("mutex")
                .push(format!("{}:{value}", self.prefix));
        }
        Ok(AgentToolResult::text(format!("{}: {value}", self.prefix)))
    }
}

struct RecordingToolCallHook {
    seen: Arc<Mutex<Vec<(String, String, String)>>>,
}

#[async_trait]
impl AgentToolCallHook for RecordingToolCallHook {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String> {
        self.seen.lock().expect("mutex").push((
            context.tool_call_id,
            context.tool_name,
            context
                .input
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        ));
        Ok(None)
    }
}

struct ReplacingToolCallHook;

#[async_trait]
impl AgentToolCallHook for ReplacingToolCallHook {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String> {
        assert_eq!(
            context.input.get("value").and_then(Value::as_str),
            Some("original")
        );
        Ok(Some(AgentToolCallHookResult::replace_input(
            json!({ "value": "hooked" }),
        )))
    }
}

struct NumberReplacingToolCallHook;

#[async_trait]
impl AgentToolCallHook for NumberReplacingToolCallHook {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String> {
        assert_eq!(
            context.input.get("value").and_then(Value::as_str),
            Some("original")
        );
        Ok(Some(AgentToolCallHookResult::replace_input(
            json!({ "value": 123 }),
        )))
    }
}

struct BlockingToolCallHook;

#[async_trait]
impl AgentToolCallHook for BlockingToolCallHook {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String> {
        assert_eq!(context.tool_call_id, "tool-1");
        assert_eq!(context.tool_name, "echo");
        assert_eq!(
            context.input.get("value").and_then(Value::as_str),
            Some("blocked")
        );
        Ok(Some(AgentToolCallHookResult::block("blocked by policy")))
    }
}

struct EditArgumentPreparer;

impl AgentToolArgumentPreparer for EditArgumentPreparer {
    fn prepare_arguments(&self, args: Value) -> Result<Value, String> {
        let old_text = args
            .get("oldText")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing oldText".to_owned())?;
        let new_text = args
            .get("newText")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing newText".to_owned())?;
        Ok(json!({
            "edits": [
                { "oldText": old_text, "newText": new_text }
            ]
        }))
    }
}

struct EditExecutor {
    executed: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl AgentToolExecutor for EditExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let edit = params
            .get("edits")
            .and_then(Value::as_array)
            .and_then(|edits| edits.first())
            .ok_or_else(|| "missing edit".to_owned())?;
        let old_text = edit
            .get("oldText")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing oldText".to_owned())?
            .to_owned();
        let new_text = edit
            .get("newText")
            .and_then(Value::as_str)
            .ok_or_else(|| "missing newText".to_owned())?
            .to_owned();
        self.executed
            .lock()
            .expect("mutex")
            .push((old_text, new_text));
        Ok(AgentToolResult::text("edited 1"))
    }
}

struct ReplacingToolResultHook;

#[async_trait]
impl AgentToolResultHook for ReplacingToolResultHook {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        assert_eq!(context.tool_call_id, "tool-1");
        assert_eq!(context.tool_name, "echo");
        assert!(!context.is_error);
        assert_eq!(
            context.input.get("value").and_then(Value::as_str),
            Some("abc")
        );
        assert_eq!(
            context.result.content,
            vec![AgentToolResultContent::Text(TextContent::new(
                "echoed: abc"
            ))]
        );
        Ok(Some(AgentToolResultHookResult::replace_result(
            AgentToolResult {
                content: vec![AgentToolResultContent::Text(TextContent::new(
                    "patched result",
                ))],
                details: Some(json!({ "patched": true })),
                terminate: true,
            },
        )))
    }
}

struct FailingToolResultHook;

#[async_trait]
impl AgentToolResultHook for FailingToolResultHook {
    async fn on_tool_result(
        &self,
        _context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        Err("hook exploded".to_owned())
    }
}

struct TerminatingToolResultHook;

#[async_trait]
impl AgentToolResultHook for TerminatingToolResultHook {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        assert_eq!(
            context.result.content,
            vec![AgentToolResultContent::Text(TextContent::new(
                "echoed: hello"
            ))]
        );
        assert_eq!(context.result.details, None);
        Ok(Some(AgentToolResultHookResult::patch_terminate(true)))
    }
}

struct ClearingToolErrorHook;

#[async_trait]
impl AgentToolResultHook for ClearingToolErrorHook {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        assert_eq!(context.tool_call_id, "tool-1");
        assert_eq!(context.tool_name, "echo");
        assert!(context.is_error);
        assert_eq!(
            context.result.content,
            vec![AgentToolResultContent::Text(TextContent::new("boom"))]
        );
        Ok(Some(AgentToolResultHookResult {
            result: Some(AgentToolResult::text("recovered")),
            content: None,
            details: None,
            terminate: None,
            is_error: Some(false),
        }))
    }
}

struct InspectingToolHookContext {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentToolCallHook for InspectingToolHookContext {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String> {
        assert_eq!(context.tool_call_id, "tool-1");
        assert_eq!(context.tool_name, "echo");
        assert_eq!(context.tool_call.id, "tool-1");
        assert_eq!(context.tool_call.name, "echo");
        assert_eq!(context.assistant_message.stop_reason, StopReason::ToolUse);
        assert_eq!(
            context
                .context
                .messages
                .iter()
                .filter_map(|message| message.role())
                .collect::<Vec<_>>(),
            vec!["user", "assistant"]
        );
        assert_eq!(context.context.tools.len(), 1);
        self.seen.lock().expect("mutex").push("before".to_owned());
        Ok(None)
    }
}

#[async_trait]
impl AgentToolResultHook for InspectingToolHookContext {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        assert_eq!(context.tool_call_id, "tool-1");
        assert_eq!(context.tool_name, "echo");
        assert_eq!(context.tool_call.id, "tool-1");
        assert_eq!(context.tool_call.name, "echo");
        assert_eq!(context.assistant_message.stop_reason, StopReason::ToolUse);
        assert_eq!(
            context
                .context
                .messages
                .iter()
                .filter_map(|message| message.role())
                .collect::<Vec<_>>(),
            vec!["user", "assistant"]
        );
        assert_eq!(context.context.tools.len(), 1);
        assert!(!context.is_error);
        self.seen.lock().expect("mutex").push("after".to_owned());
        Ok(None)
    }
}

struct TailTransformer {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentContextTransformer for TailTransformer {
    async fn transform_context(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, String> {
        self.seen
            .lock()
            .expect("mutex")
            .extend(messages.iter().filter_map(|message| match message {
                AgentMessage::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        Ok(messages
            .into_iter()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect())
    }
}

struct RecordingConverter {
    seen_roles: Arc<Mutex<Vec<&'static str>>>,
}

impl AgentMessageConverter for RecordingConverter {
    fn convert_to_llm(&self, messages: &[AgentMessage]) -> Result<Vec<Message>, String> {
        self.seen_roles
            .lock()
            .expect("mutex")
            .extend(messages.iter().map(role_of));
        Ok(messages
            .iter()
            .filter_map(AgentMessage::to_llm_message)
            .collect())
    }
}

struct FilteringNotificationConverter {
    seen_roles: Arc<Mutex<Vec<String>>>,
    converted_roles: Arc<Mutex<Vec<String>>>,
}

impl AgentMessageConverter for FilteringNotificationConverter {
    fn convert_to_llm(&self, messages: &[AgentMessage]) -> Result<Vec<Message>, String> {
        self.seen_roles.lock().expect("mutex").extend(
            messages
                .iter()
                .filter_map(AgentMessage::role)
                .map(str::to_owned),
        );
        let llm_messages = messages
            .iter()
            .filter(|message| message.role() != Some("notification"))
            .filter_map(AgentMessage::to_llm_message)
            .collect::<Vec<_>>();
        self.converted_roles
            .lock()
            .expect("mutex")
            .extend(llm_messages.iter().map(llm_role_of).map(str::to_owned));
        Ok(llm_messages)
    }
}

struct CustomToUserConverter;

impl AgentMessageConverter for CustomToUserConverter {
    fn convert_to_llm(&self, messages: &[AgentMessage]) -> Result<Vec<Message>, String> {
        Ok(messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::Custom(value) if message.role() == Some("custom") => {
                    Some(Message::User(UserMessage {
                        content: UserContentValue::Plain(
                            value
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        ),
                        timestamp: value
                            .get("timestamp")
                            .and_then(Value::as_i64)
                            .unwrap_or_else(now_millis),
                    }))
                }
                _ => message.to_llm_message(),
            })
            .collect())
    }
}

struct QueuedAfterToolsProvider {
    executed: Arc<Mutex<Vec<String>>>,
    delivered: AtomicBool,
}

#[async_trait]
impl AgentQueuedMessageProvider for QueuedAfterToolsProvider {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        let tool_count = self.executed.lock().expect("mutex").len();
        if tool_count >= 2 && !self.delivered.swap(true, Ordering::SeqCst) {
            Ok(vec![AgentMessage::User(UserMessage::text("interrupt"))])
        } else {
            Ok(Vec::new())
        }
    }
}

struct OneShotMessageProvider {
    message: AgentMessage,
    delivered: AtomicBool,
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentQueuedMessageProvider for OneShotMessageProvider {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if self.delivered.swap(true, Ordering::SeqCst) {
            Ok(Vec::new())
        } else {
            Ok(vec![self.message.clone()])
        }
    }
}

struct EmptyCountingMessageProvider {
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentQueuedMessageProvider for EmptyCountingMessageProvider {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }
}

struct CountingNextTurnPreparer {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentNextTurnPreparer for CountingNextTurnPreparer {
    async fn prepare_next_turn(
        &self,
        _context: AgentNextTurnContext,
    ) -> Result<Option<AgentLoopTurnUpdate>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
}

struct CountingShouldStopAfterTurn {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl AgentShouldStopAfterTurn for CountingShouldStopAfterTurn {
    async fn should_stop_after_turn(&self, _context: AgentNextTurnContext) -> Result<bool, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    }
}

struct SecondPromptNextTurnPreparer {
    prepared: AtomicBool,
    seen: Arc<Mutex<Vec<(String, usize, usize)>>>,
}

#[async_trait]
impl AgentNextTurnPreparer for SecondPromptNextTurnPreparer {
    async fn prepare_next_turn(
        &self,
        context: AgentNextTurnContext,
    ) -> Result<Option<AgentLoopTurnUpdate>, String> {
        self.seen.lock().expect("mutex").push((
            context.context.system_prompt.clone(),
            context.tool_results.len(),
            context.new_messages.len(),
        ));
        if self.prepared.swap(true, Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(Some(AgentLoopTurnUpdate {
            context: Some(AgentContext {
                system_prompt: "second prompt".to_owned(),
                messages: context.context.messages.clone(),
                tools: context.context.tools.clone(),
            }),
            model: None,
            thinking_level: None,
            stream_options: None,
        }))
    }
}

struct StopAfterTurnHook {
    seen: Arc<Mutex<Vec<(Vec<String>, Vec<String>, Vec<String>)>>>,
}

#[async_trait]
impl AgentShouldStopAfterTurn for StopAfterTurnHook {
    async fn should_stop_after_turn(&self, context: AgentNextTurnContext) -> Result<bool, String> {
        self.seen.lock().expect("mutex").push((
            context
                .tool_results
                .iter()
                .map(|result| result.tool_call_id.clone())
                .collect(),
            context
                .context
                .messages
                .iter()
                .map(role_of)
                .map(str::to_owned)
                .collect(),
            context
                .new_messages
                .iter()
                .map(role_of)
                .map(str::to_owned)
                .collect(),
        ));
        Ok(true)
    }
}

#[tokio::test]
async fn agent_loop_emits_lifecycle_events_and_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Hi there!", Default::default()).into(),
    ]);

    let config = AgentLoopConfig::new(registration.get_model());
    let (messages, events) = agent_loop_prompt(context_with_model(&config.model), "Hello", config)
        .await
        .expect("loop");

    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0], AgentMessage::User(_)));
    assert_eq!(text_of(&messages[1]), Some("Hi there!"));
    let event_names: Vec<&str> = events.iter().map(event_name).collect();
    assert!(event_names.starts_with(&["agent_start", "turn_start", "message_start"]));
    assert!(event_names.contains(&"message_update"));
    assert!(event_names.contains(&"turn_end"));
    assert_eq!(event_names.last(), Some(&"agent_end"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_uses_stream_result_when_provider_ends_without_terminal_event() {
    let model = Model::faux("end-only-api", "end-only-provider", "end-only-model");
    let mut config = AgentLoopConfig::new(model.clone());
    config.stream_provider = Some(Arc::new(EndOnlyStreamProvider {
        response: faux_assistant_message("final from result", Default::default()),
    }));

    let (messages, events) = agent_loop_prompt(context_with_model(&model), "Hello", config)
        .await
        .expect("loop");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(text_of(&messages[1]), Some("final from result"));
    assert!(matches!(
        &messages[1],
        AgentMessage::Assistant(assistant)
            if assistant.stop_reason == StopReason::Stop && assistant.error_message.is_none()
    ));
    assert_eq!(
        events.iter().map(event_name).collect::<Vec<_>>(),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

#[tokio::test]
async fn agent_loop_filters_custom_messages_through_converter() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let provider_roles = Arc::new(Mutex::new(Vec::new()));
    let provider_roles_ref = provider_roles.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        provider_roles_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().map(llm_role_of).map(str::to_owned));
        faux_assistant_message("Response", Default::default())
    })]);

    let seen_roles = Arc::new(Mutex::new(Vec::new()));
    let converted_roles = Arc::new(Mutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: "You are helpful.".to_owned(),
        messages: vec![AgentMessage::custom(json!({
            "role": "notification",
            "text": "This is a notification",
            "timestamp": now_millis()
        }))],
        tools: Vec::new(),
    };
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.convert_to_llm = Some(Arc::new(FilteringNotificationConverter {
        seen_roles: seen_roles.clone(),
        converted_roles: converted_roles.clone(),
    }));

    let (messages, _events) = agent_loop_prompt(context, "Hello", config)
        .await
        .expect("loop");

    assert_eq!(
        *seen_roles.lock().expect("mutex"),
        vec!["notification".to_owned(), "user".to_owned()]
    );
    assert_eq!(
        *converted_roles.lock().expect("mutex"),
        vec!["user".to_owned()]
    );
    assert_eq!(
        *provider_roles.lock().expect("mutex"),
        vec!["user".to_owned()]
    );
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_executes_tool_calls_and_appends_results() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let config = AgentLoopConfig::new(registration.get_model());
    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["abc".to_owned()]);
    assert_eq!(messages.len(), 4);
    assert!(matches!(messages[1], AgentMessage::Assistant(_)));
    assert_eq!(text_of(&messages[2]), Some("echoed: abc"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let event_names: Vec<&str> = events.iter().map(event_name).collect();
    assert!(event_names.contains(&"tool_execution_start"));
    assert!(event_names.contains(&"tool_execution_end"));
    let tool_result_message_events: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageStart {
                message: AgentMessage::ToolResult(result),
            } if result.tool_call_id == "tool-1" => Some("message_start"),
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } if result.tool_call_id == "tool-1" => Some("message_end"),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_result_message_events,
        vec!["message_start", "message_end"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_emits_tool_execution_update_from_tool_callbacks() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "progress",
                json!({ "value": "original" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let update_observed_before_return = Arc::new(AtomicBool::new(false));
    let tool = AgentTool {
        definition: Tool {
            name: "progress".to_owned(),
            description: "Progress tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Progress".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(UpdatingExecutor {
            update_observed_before_return: update_observed_before_return.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(Arc::new(ReplacingToolCallHook));
    let (messages, events) = agent_loop_prompt(context, "run progress tool", config)
        .await
        .expect("loop");

    assert!(update_observed_before_return.load(Ordering::SeqCst));
    assert_eq!(text_of(&messages[2]), Some("final: hooked"));
    assert_eq!(text_of(&messages[3]), Some("done"));

    let tool_events = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolExecutionStart { .. }
                    | AgentEvent::ToolExecutionUpdate { .. }
                    | AgentEvent::ToolExecutionEnd { .. }
            )
        })
        .map(event_name)
        .collect::<Vec<_>>();
    assert_eq!(
        tool_events,
        vec![
            "tool_execution_start",
            "tool_execution_update",
            "tool_execution_end"
        ]
    );

    let update = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => Some((tool_call_id, tool_name, args, partial_result)),
        _ => None,
    });
    let Some((tool_call_id, tool_name, args, partial_result)) = update else {
        panic!("expected tool execution update");
    };
    assert_eq!(tool_call_id, "tool-1");
    assert_eq!(tool_name, "progress");
    assert_eq!(args.get("value").and_then(Value::as_str), Some("original"));
    assert_eq!(
        partial_result.content,
        vec![AgentToolResultContent::Text(TextContent::new(
            "partial: hooked"
        ))]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_resolves_dynamic_api_key_before_each_provider_request() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let responses = Arc::new(Mutex::new(VecDeque::from(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "again" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        ),
        faux_assistant_message("done", Default::default()),
    ])));
    let seen_options = Arc::new(Mutex::new(Vec::new()));
    let providers = Arc::new(Mutex::new(Vec::new()));
    let keys = Arc::new(Mutex::new(VecDeque::from(vec![
        Some("dynamic-key".to_owned()),
        None,
    ])));

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.stream_options.stream.api_key = Some("static-key".to_owned());
    config.stream_provider = Some(Arc::new(RecordingOptionsStreamProvider {
        seen_options: seen_options.clone(),
        responses,
    }));
    config.api_key_provider = Some(Arc::new(RecordingApiKeyProvider {
        keys,
        providers: providers.clone(),
    }));

    let (messages, _events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["again".to_owned()]);
    assert_eq!(text_of(&messages[2]), Some("echoed: again"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let seen = seen_options.lock().expect("mutex");
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].stream.api_key.as_deref(), Some("dynamic-key"));
    assert_eq!(seen[1].stream.api_key.as_deref(), Some("static-key"));
    assert_eq!(
        *providers.lock().expect("mutex"),
        vec![
            registration.get_model().provider.clone(),
            registration.get_model().provider.clone()
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_turns_missing_tools_into_error_tool_results_and_continues() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "missing",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let (messages, events) = agent_loop_prompt(
        context_with_model(&registration.get_model()),
        "run missing tool",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult", "assistant"]
    );
    assert_eq!(text_of(&messages[2]), Some("Tool missing not found"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
        _ => None,
    });
    assert_eq!(tool_end, Some(true));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_validates_tool_arguments_before_hooks_and_execution() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({}).as_object().cloned().unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let seen_hook = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(Arc::new(RecordingToolCallHook {
        seen: seen_hook.clone(),
    }));

    let (messages, events) = agent_loop_prompt(context, "run invalid tool", config)
        .await
        .expect("loop");

    assert!(executed.lock().expect("mutex").is_empty());
    assert!(seen_hook.lock().expect("mutex").is_empty());
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult", "assistant"]
    );
    let validation_error = text_of(&messages[2]).expect("validation error");
    assert!(validation_error.contains("Validation failed for tool \"echo\""));
    assert!(validation_error.contains("value: required property is missing"));
    assert!(validation_error.contains("Received arguments"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            result, is_error, ..
        } => Some((result, *is_error)),
        _ => None,
    });
    let Some((result, is_error)) = tool_end else {
        panic!("expected validation tool execution end");
    };
    assert!(is_error);
    assert!(matches!(
        result.content.first(),
        Some(AgentToolResultContent::Text(text)) if text.text == validation_error
    ));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_turns_tool_execution_errors_into_error_tool_results_and_continues() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "fail",
                json!({}).as_object().cloned().unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let tool = AgentTool {
        definition: Tool {
            name: "fail".to_owned(),
            description: "Failing tool".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        label: "Fail".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(FailingExecutor),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let (messages, events) = agent_loop_prompt(
        context,
        "run failing tool",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult", "assistant"]
    );
    assert_eq!(text_of(&messages[2]), Some("boom"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
        _ => None,
    });
    assert_eq!(tool_end, Some(true));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_turns_tool_result_hook_errors_into_error_tool_results_and_continues() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: Arc::new(Mutex::new(Vec::new())),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config
        .tool_result_hooks
        .push(Arc::new(FailingToolResultHook));
    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(text_of(&messages[2]), Some("hook exploded"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
        _ => None,
    });
    assert_eq!(tool_end, Some(true));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_injects_queued_messages_after_all_tool_results() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let second_request_user_texts = Arc::new(Mutex::new(Vec::new()));
    let second_request_user_texts_ref = second_request_user_texts.clone();
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_response_factory(move |context, _, _, _| {
            second_request_user_texts_ref.lock().expect("mutex").extend(
                context.messages.iter().filter_map(|message| match message {
                    Message::User(user) => match &user.content {
                        UserContentValue::Plain(text) => Some(text.clone()),
                        UserContentValue::Blocks(_) => None,
                    },
                    _ => None,
                }),
            );
            faux_assistant_message("done", Default::default())
        }),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_execution = ToolExecutionMode::Sequential;
    config.queued_message_provider = Some(Arc::new(QueuedAfterToolsProvider {
        executed: executed.clone(),
        delivered: AtomicBool::new(false),
    }));

    let (messages, events) = agent_loop_prompt(context, "run tools", config)
        .await
        .expect("loop");

    assert_eq!(
        *executed.lock().expect("mutex"),
        vec!["first".to_owned(), "second".to_owned()]
    );
    assert_eq!(messages.len(), 6);
    assert_eq!(text_of(&messages[2]), Some("echoed: first"));
    assert_eq!(text_of(&messages[3]), Some("echoed: second"));
    assert!(matches!(
        &messages[4],
        AgentMessage::User(user) if matches!(&user.content, UserContentValue::Plain(text) if text == "interrupt")
    ));
    assert_eq!(text_of(&messages[5]), Some("done"));
    assert!(
        second_request_user_texts
            .lock()
            .expect("mutex")
            .contains(&"interrupt".to_owned())
    );

    let message_start_markers = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageStart {
                message: AgentMessage::ToolResult(result),
            } => Some(format!("tool:{}", result.tool_call_id)),
            AgentEvent::MessageStart {
                message: AgentMessage::User(user),
            } => match &user.content {
                UserContentValue::Plain(text) => Some(format!("user:{text}")),
                UserContentValue::Blocks(_) => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let tool_1_index = message_start_markers
        .iter()
        .position(|marker| marker == "tool:tool-1")
        .expect("tool-1 result event");
    let tool_2_index = message_start_markers
        .iter()
        .position(|marker| marker == "tool:tool-2")
        .expect("tool-2 result event");
    let interrupt_index = message_start_markers
        .iter()
        .position(|marker| marker == "user:interrupt")
        .expect("queued user event");
    assert!(tool_1_index < interrupt_index);
    assert!(tool_2_index < interrupt_index);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_uses_prepare_next_turn_snapshot_before_continuing() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let second_request_prompts = Arc::new(Mutex::new(Vec::new()));
    let second_request_prompts_ref = second_request_prompts.clone();
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "hello" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_response_factory(move |context, _, _, _| {
            second_request_prompts_ref
                .lock()
                .expect("mutex")
                .push(context.system_prompt.clone().unwrap_or_default());
            faux_assistant_message("done", Default::default())
        }),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let context = AgentContext {
        system_prompt: "first prompt".to_owned(),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let seen_prepare = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.prepare_next_turn = Some(Arc::new(SecondPromptNextTurnPreparer {
        prepared: AtomicBool::new(false),
        seen: seen_prepare.clone(),
    }));

    let (messages, _) = agent_loop_prompt(context, "echo something", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["hello".to_owned()]);
    assert_eq!(text_of(&messages[2]), Some("echoed: hello"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    assert_eq!(
        *seen_prepare.lock().expect("mutex"),
        vec![
            ("first prompt".to_owned(), 1, 3),
            ("second prompt".to_owned(), 0, 4)
        ]
    );
    assert_eq!(
        *second_request_prompts.lock().expect("mutex"),
        vec!["second prompt".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_should_stop_after_current_turn_when_hook_returns_true() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "hello" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("should not run", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let seen_stop = Arc::new(Mutex::new(Vec::new()));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.should_stop_after_turn = Some(Arc::new(StopAfterTurnHook {
        seen: seen_stop.clone(),
    }));
    config.follow_up_message_provider = Some(Arc::new(OneShotMessageProvider {
        message: AgentMessage::User(UserMessage::text("follow up should stay queued")),
        delivered: AtomicBool::new(false),
        polls: follow_up_polls.clone(),
    }));

    let (messages, events) = agent_loop_prompt(context, "echo something", config)
        .await
        .expect("loop");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(*executed.lock().expect("mutex"), vec!["hello".to_owned()]);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(
        *seen_stop.lock().expect("mutex"),
        vec![(
            vec!["tool-1".to_owned()],
            vec![
                "user".to_owned(),
                "assistant".to_owned(),
                "toolResult".to_owned()
            ],
            vec![
                "user".to_owned(),
                "assistant".to_owned(),
                "toolResult".to_owned()
            ]
        )]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event_name(event) == "turn_end")
            .count(),
        1
    );
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_terminal_assistant_error_or_abort_skips_next_turn_hooks_and_queues() {
    for (stop_reason, error_message) in [
        (StopReason::Error, "provider failed"),
        (StopReason::Aborted, "Request was aborted"),
    ] {
        let registration = register_faux_provider(RegisterFauxProviderOptions::default());
        registration.set_responses(vec![
            faux_assistant_message(
                "terminal",
                FauxAssistantOptions {
                    stop_reason: Some(stop_reason.clone()),
                    error_message: Some(error_message.to_owned()),
                    ..Default::default()
                },
            )
            .into(),
            faux_assistant_message("should not run", Default::default()).into(),
        ]);

        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let should_stop_calls = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.prepare_next_turn = Some(Arc::new(CountingNextTurnPreparer {
            calls: prepare_calls.clone(),
        }));
        config.should_stop_after_turn = Some(Arc::new(CountingShouldStopAfterTurn {
            calls: should_stop_calls.clone(),
        }));
        config.queued_message_provider = Some(Arc::new(EmptyCountingMessageProvider {
            polls: steering_polls.clone(),
        }));
        config.follow_up_message_provider = Some(Arc::new(OneShotMessageProvider {
            message: AgentMessage::User(UserMessage::text("follow up should stay queued")),
            delivered: AtomicBool::new(false),
            polls: follow_up_polls.clone(),
        }));

        let (messages, events) = agent_loop_prompt(
            context_with_model(&registration.get_model()),
            "Initial",
            config,
        )
        .await
        .expect("loop");

        assert_eq!(registration.state().call_count(), 1);
        assert_eq!(
            messages.iter().map(role_of).collect::<Vec<_>>(),
            vec!["user", "assistant"]
        );
        let AgentMessage::Assistant(assistant) = messages.last().expect("assistant") else {
            panic!("assistant");
        };
        assert_eq!(assistant.stop_reason, stop_reason);
        assert_eq!(assistant.error_message.as_deref(), Some(error_message));
        let event_names = events.iter().map(event_name).collect::<Vec<_>>();
        assert_eq!(event_names.first(), Some(&"agent_start"));
        assert_eq!(event_names.last(), Some(&"agent_end"));
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "turn_start")
                .count(),
            1
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "turn_end")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    AgentEvent::MessageStart { message } => Some(role_of(message)),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec!["user", "assistant"]
        );
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert_eq!(should_stop_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            steering_polls.load(Ordering::SeqCst),
            1,
            "initial steering poll still happens before the terminal assistant response"
        );
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
        registration.unregister();
    }
}

#[tokio::test]
async fn agent_loop_processes_follow_up_messages_after_agent_would_stop() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let second_request_user_texts = Arc::new(Mutex::new(Vec::new()));
    let second_request_user_texts_ref = second_request_user_texts.clone();
    registration.set_responses(vec![
        faux_assistant_message("Processed initial", Default::default()).into(),
        faux_response_factory(move |context, _, _, _| {
            second_request_user_texts_ref.lock().expect("mutex").extend(
                context.messages.iter().filter_map(|message| match message {
                    Message::User(user) => match &user.content {
                        UserContentValue::Plain(text) => Some(text.clone()),
                        UserContentValue::Blocks(_) => None,
                    },
                    _ => None,
                }),
            );
            faux_assistant_message("Processed follow-up", Default::default())
        }),
    ]);

    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.follow_up_message_provider = Some(Arc::new(OneShotMessageProvider {
        message: AgentMessage::User(UserMessage::text("Queued follow-up")),
        delivered: AtomicBool::new(false),
        polls: follow_up_polls.clone(),
    }));

    let (messages, events) = agent_loop_prompt(
        context_with_model(&registration.get_model()),
        "Initial",
        config,
    )
    .await
    .expect("loop");

    assert_eq!(registration.state().call_count(), 2);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 2);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "user", "assistant"]
    );
    assert!(matches!(
        &messages[2],
        AgentMessage::User(user) if matches!(&user.content, UserContentValue::Plain(text) if text == "Queued follow-up")
    ));
    assert_eq!(text_of(&messages[3]), Some("Processed follow-up"));
    assert!(
        second_request_user_texts
            .lock()
            .expect("mutex")
            .contains(&"Queued follow-up".to_owned())
    );
    let message_end_roles = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => Some(role_of(message)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        message_end_roles,
        vec!["user", "assistant", "user", "assistant"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_injects_queued_messages_before_initial_provider_request() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_user_texts = Arc::new(Mutex::new(Vec::new()));
    let request_user_texts_ref = request_user_texts.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_user_texts_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().filter_map(|message| match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        faux_assistant_message("Processed", Default::default())
    })]);

    let polls = Arc::new(AtomicUsize::new(0));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.queued_message_provider = Some(Arc::new(OneShotMessageProvider {
        message: AgentMessage::User(UserMessage::text("Queued steering")),
        delivered: AtomicBool::new(false),
        polls: polls.clone(),
    }));

    let (messages, events) = agent_loop_prompt(
        context_with_model(&registration.get_model()),
        "Initial",
        config,
    )
    .await
    .expect("loop");

    assert_eq!(polls.load(Ordering::SeqCst), 2);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "user", "assistant"]
    );
    assert_eq!(
        *request_user_texts.lock().expect("mutex"),
        vec!["Initial".to_owned(), "Queued steering".to_owned()]
    );
    let message_end_roles = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => Some(role_of(message)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(message_end_roles, vec!["user", "user", "assistant"]);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_stops_after_tool_batch_when_all_results_terminate() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "hello", "terminate": true })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("should not run", Default::default()).into(),
    ]);

    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" },
                    "terminate": { "type": "boolean" }
                },
                "required": ["value", "terminate"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(ConditionalTerminateExecutor),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);

    let (messages, events) = agent_loop_prompt(
        context,
        "echo something",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event_name(event) == "turn_end")
            .count(),
        1
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_continues_after_parallel_tool_batch_when_not_all_results_terminate() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first", "terminate": true })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second", "terminate": false })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" },
                    "terminate": { "type": "boolean" }
                },
                "required": ["value", "terminate"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(ConditionalTerminateExecutor),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_execution = ToolExecutionMode::Parallel;

    let (messages, _) = agent_loop_prompt(context, "echo both", config)
        .await
        .expect("loop");

    assert_eq!(registration.state().call_count(), 2);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult", "toolResult", "assistant"]
    );
    assert_eq!(text_of(&messages[4]), Some("done"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_prepares_tool_arguments_before_execution() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "edit",
                json!({ "oldText": "before", "newText": "after" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "edit".to_owned(),
            description: "Edit tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string" },
                                "newText": { "type": "string" }
                            },
                            "required": ["oldText", "newText"]
                        }
                    }
                },
                "required": ["edits"]
            }),
        },
        label: "Edit".to_owned(),
        execution_mode: None,
        argument_preparer: Some(Arc::new(EditArgumentPreparer)),
        executor: Arc::new(EditExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let (messages, events) = agent_loop_prompt(
        context,
        "edit something",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert_eq!(
        *executed.lock().expect("mutex"),
        vec![("before".to_owned(), "after".to_owned())]
    );
    assert_eq!(text_of(&messages[2]), Some("edited 1"));
    let tool_start_args = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionStart { args, .. } => Some(args),
        _ => None,
    });
    assert_eq!(
        tool_start_args
            .and_then(|args| args.get("oldText"))
            .and_then(Value::as_str),
        Some("before")
    );
    assert!(tool_start_args.and_then(|args| args.get("edits")).is_none());
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_tool_call_hook_can_replace_arguments_before_execution() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "original" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(Arc::new(ReplacingToolCallHook));
    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["hooked".to_owned()]);
    assert_eq!(text_of(&messages[2]), Some("echoed: hooked"));
    let tool_start_args = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionStart { args, .. } => Some(args),
        _ => None,
    });
    assert_eq!(
        tool_start_args
            .and_then(|args| args.get("value"))
            .and_then(Value::as_str),
        Some("original")
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_tool_call_hook_can_block_execution_with_error_result() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "blocked" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(Arc::new(BlockingToolCallHook));
    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert!(executed.lock().expect("mutex").is_empty());
    assert_eq!(text_of(&messages[2]), Some("blocked by policy"));
    assert_eq!(text_of(&messages[3]), Some("done"));
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            result,
            is_error,
            ..
        } if tool_call_id == "tool-1" => Some((result, *is_error)),
        _ => None,
    });
    let Some((result, is_error)) = tool_end else {
        panic!("expected blocked tool execution end");
    };
    assert!(is_error);
    assert!(matches!(
        result.content.first(),
        Some(AgentToolResultContent::Text(text)) if text.text == "blocked by policy"
    ));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_executes_hook_replaced_args_without_schema_revalidation() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "original" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(RecordingValueExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config
        .tool_call_hooks
        .push(Arc::new(NumberReplacingToolCallHook));
    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(
        *executed.lock().expect("mutex"),
        vec![json!({ "value": 123 })]
    );
    assert_eq!(text_of(&messages[2]), Some("echoed: 123"));
    let tool_start_args = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionStart { args, .. } => Some(args),
        _ => None,
    });
    assert_eq!(
        tool_start_args
            .and_then(|args| args.get("value"))
            .and_then(Value::as_str),
        Some("original")
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_parallel_tool_execution_emits_completion_order_and_persists_source_order() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let first_can_finish = Arc::new(tokio::sync::Notify::new());
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(ParallelEchoExecutor {
            first_can_finish: first_can_finish.clone(),
            first_resolved: first_resolved.clone(),
            parallel_observed: parallel_observed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);

    let (messages, events) = agent_loop_prompt(
        context,
        "run tools",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert!(parallel_observed.load(Ordering::SeqCst));
    let tool_execution_end_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_execution_end_ids, vec!["tool-2", "tool-1"]);

    let tool_result_message_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_result_message_ids, vec!["tool-1", "tool-2"]);

    let turn_tool_result_ids = events
        .iter()
        .flat_map(|event| match event {
            AgentEvent::TurnEnd { tool_results, .. } => tool_results
                .iter()
                .map(|result| result.tool_call_id.as_str())
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();
    assert_eq!(turn_tool_result_ids, vec!["tool-1", "tool-2"]);
    assert_eq!(text_of(&messages[2]), Some("echoed: first"));
    assert_eq!(text_of(&messages[3]), Some("echoed: second"));
    assert_eq!(text_of(&messages[4]), Some("done"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_forces_sequential_execution_when_tool_requires_it() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "slow",
                    json!({ "value": "first" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "slow",
                    json!({ "value": "second" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let tool = AgentTool {
        definition: Tool {
            name: "slow".to_owned(),
            description: "Slow tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Slow".to_owned(),
        execution_mode: Some(ToolExecutionMode::Sequential),
        argument_preparer: None,
        executor: Arc::new(SequentialProbeExecutor {
            first_resolved: first_resolved.clone(),
            parallel_observed: parallel_observed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_execution = ToolExecutionMode::Parallel;

    let (messages, events) = agent_loop_prompt(context, "run tools", config)
        .await
        .expect("loop");

    assert!(!parallel_observed.load(Ordering::SeqCst));
    let tool_execution_end_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_execution_end_ids, vec!["tool-1", "tool-2"]);
    assert_eq!(text_of(&messages[2]), Some("echoed: first"));
    assert_eq!(text_of(&messages[3]), Some("echoed: second"));
    assert_eq!(text_of(&messages[4]), Some("done"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_sequential_tool_execution_emits_each_result_before_next_start() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_execution = ToolExecutionMode::Sequential;

    let (_messages, events) = agent_loop_prompt(context, "run tools", config)
        .await
        .expect("loop");

    assert_eq!(
        *executed.lock().expect("mutex"),
        vec!["first".to_owned(), "second".to_owned()]
    );
    let sequence = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                Some(format!("start:{tool_call_id}"))
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                Some(format!("end:{tool_call_id}"))
            }
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } => Some(format!("message:{}", result.tool_call_id)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequence,
        vec![
            "start:tool-1",
            "end:tool-1",
            "message:tool-1",
            "start:tool-2",
            "end:tool-2",
            "message:tool-2",
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_forces_sequential_execution_when_any_tool_requires_it() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "slow",
                    json!({ "value": "a" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "fast",
                    json!({ "value": "b" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let order = Arc::new(Mutex::new(Vec::new()));
    let slow_tool = AgentTool {
        definition: Tool {
            name: "slow".to_owned(),
            description: "Slow tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Slow".to_owned(),
        execution_mode: Some(ToolExecutionMode::Sequential),
        argument_preparer: None,
        executor: Arc::new(OrderedDelayExecutor {
            prefix: "slow",
            order: order.clone(),
            delay_ms: 20,
        }),
    };
    let fast_tool = AgentTool {
        definition: Tool {
            name: "fast".to_owned(),
            description: "Fast tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Fast".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(OrderedDelayExecutor {
            prefix: "fast",
            order: order.clone(),
            delay_ms: 0,
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(slow_tool);
    context.tools.push(fast_tool);

    let (messages, events) = agent_loop_prompt(
        context,
        "run tools",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert_eq!(
        *order.lock().expect("mutex"),
        vec![
            "slow:start:a".to_owned(),
            "slow:end:a".to_owned(),
            "fast:b".to_owned()
        ]
    );
    let tool_result_message_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_result_message_ids, vec!["tool-1", "tool-2"]);
    assert_eq!(text_of(&messages[2]), Some("slow: a"));
    assert_eq!(text_of(&messages[3]), Some("fast: b"));
    assert_eq!(text_of(&messages[4]), Some("done"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_runs_parallel_tools_in_parallel_by_default() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let first_can_finish = Arc::new(tokio::sync::Notify::new());
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: Some(ToolExecutionMode::Parallel),
        argument_preparer: None,
        executor: Arc::new(ParallelEchoExecutor {
            first_can_finish: first_can_finish.clone(),
            first_resolved: first_resolved.clone(),
            parallel_observed: parallel_observed.clone(),
        }),
    };
    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);

    let (_messages, events) = agent_loop_prompt(
        context,
        "run tools",
        AgentLoopConfig::new(registration.get_model()),
    )
    .await
    .expect("loop");

    assert!(parallel_observed.load(Ordering::SeqCst));
    let tool_execution_end_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_execution_end_ids, vec!["tool-2", "tool-1"]);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_transforms_context_before_converting_to_llm() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_roles = Arc::new(Mutex::new(Vec::new()));
    let request_roles_ref = request_roles.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_roles_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().map(|message| match message {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
                Message::ToolResult(_) => "toolResult",
            }));
        faux_assistant_message("ok", Default::default())
    })]);

    let transformed_seen = Arc::new(Mutex::new(Vec::new()));
    let converted_seen = Arc::new(Mutex::new(Vec::new()));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: vec![
            AgentMessage::User(UserMessage::text("old message 1")),
            AgentMessage::Assistant(faux_assistant_message("old response 1", Default::default())),
            AgentMessage::User(UserMessage::text("old message 2")),
            AgentMessage::Assistant(faux_assistant_message("old response 2", Default::default())),
        ],
        tools: Vec::new(),
    };
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.transform_context = Some(Arc::new(TailTransformer {
        seen: transformed_seen.clone(),
    }));
    config.convert_to_llm = Some(Arc::new(RecordingConverter {
        seen_roles: converted_seen.clone(),
    }));

    let (messages, _) = agent_loop_prompt(context, "new message", config)
        .await
        .expect("loop");

    assert_eq!(text_of(&messages[1]), Some("ok"));
    assert_eq!(
        *transformed_seen.lock().expect("mutex"),
        vec![
            "old message 1".to_owned(),
            "old message 2".to_owned(),
            "new message".to_owned()
        ]
    );
    assert_eq!(
        *converted_seen.lock().expect("mutex"),
        vec!["assistant", "user"]
    );
    assert_eq!(
        *request_roles.lock().expect("mutex"),
        vec!["assistant", "user"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_runs_tool_call_and_tool_result_hooks() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let seen_calls = Arc::new(Mutex::new(Vec::new()));

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(Arc::new(RecordingToolCallHook {
        seen: seen_calls.clone(),
    }));
    config
        .tool_result_hooks
        .push(Arc::new(ReplacingToolResultHook));

    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["abc".to_owned()]);
    assert_eq!(
        *seen_calls.lock().expect("mutex"),
        vec![("tool-1".to_owned(), "echo".to_owned(), "abc".to_owned())]
    );
    assert_eq!(messages.len(), 3);
    assert_eq!(text_of(&messages[2]), Some("patched result"));
    let AgentMessage::ToolResult(tool_result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert_eq!(tool_result.details, Some(json!({ "patched": true })));
    let event_names: Vec<&str> = events.iter().map(event_name).collect();
    assert!(event_names.contains(&"tool_execution_start"));
    assert!(event_names.contains(&"tool_execution_end"));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_tool_hooks_receive_assistant_and_context_snapshot() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = Arc::new(InspectingToolHookContext { seen: seen.clone() });

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.tool_call_hooks.push(hook.clone());
    config.tool_result_hooks.push(hook);

    let (messages, _events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(*executed.lock().expect("mutex"), vec!["abc".to_owned()]);
    assert_eq!(text_of(&messages[2]), Some("echoed: abc"));
    assert_eq!(
        *seen.lock().expect("mutex"),
        vec!["before".to_owned(), "after".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_tool_result_hook_can_override_error_flag() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(FailingExecutor),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config
        .tool_result_hooks
        .push(Arc::new(ClearingToolErrorHook));

    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(text_of(&messages[2]), Some("recovered"));
    let AgentMessage::ToolResult(tool_result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert!(!tool_result.is_error);
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            is_error,
            ..
        } if tool_call_id == "tool-1" => Some(*is_error),
        _ => None,
    });
    assert_eq!(tool_end, Some(false));
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_tool_result_hook_can_terminate_tool_batch() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "hello" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("tool-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("should not run", Default::default()).into(),
    ]);

    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };

    let mut context = context_with_model(&registration.get_model());
    context.tools.push(tool);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config
        .tool_result_hooks
        .push(Arc::new(TerminatingToolResultHook));

    let (messages, events) = agent_loop_prompt(context, "run tool", config)
        .await
        .expect("loop");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(*executed.lock().expect("mutex"), vec!["hello".to_owned()]);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(text_of(&messages[2]), Some("echoed: hello"));
    let AgentMessage::ToolResult(tool_result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert_eq!(tool_result.details, None);
    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            result, is_error, ..
        } => Some((result, is_error)),
        _ => None,
    });
    let Some((result, is_error)) = tool_end else {
        panic!("expected tool execution end");
    };
    assert!(!is_error);
    assert_eq!(result.terminate, true);
    assert_eq!(
        result.content,
        vec![AgentToolResultContent::Text(TextContent::new(
            "echoed: hello"
        ))]
    );
    assert_eq!(result.details, None);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_continue_validates_context_tail() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let config = AgentLoopConfig::new(registration.get_model());

    let empty = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    assert_eq!(
        agent_loop_continue(empty, config.clone())
            .await
            .expect_err("empty context"),
        "Cannot continue: no messages in context"
    );

    let assistant_tail = AgentContext {
        system_prompt: String::new(),
        messages: vec![AgentMessage::Assistant(faux_assistant_message(
            "tail",
            Default::default(),
        ))],
        tools: Vec::new(),
    };
    assert!(
        agent_loop_continue(assistant_tail, config)
            .await
            .expect_err("assistant tail")
            .contains("assistant")
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_continue_from_existing_context_omits_existing_user_events() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Response", Default::default()).into(),
    ]);

    let context = AgentContext {
        system_prompt: "You are helpful.".to_owned(),
        messages: vec![AgentMessage::User(UserMessage::text("Hello"))],
        tools: Vec::new(),
    };
    let (messages, events) =
        agent_loop_continue(context, AgentLoopConfig::new(registration.get_model()))
            .await
            .expect("continue");

    assert_eq!(messages.len(), 1);
    assert!(matches!(messages[0], AgentMessage::Assistant(_)));
    assert_eq!(text_of(&messages[0]), Some("Response"));
    let ended_roles = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => Some(role_of(message)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ended_roles, vec!["assistant"]);
    registration.unregister();
}

#[tokio::test]
async fn agent_loop_continue_allows_custom_last_message_via_converter() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let provider_user_texts = Arc::new(Mutex::new(Vec::new()));
    let provider_user_texts_ref = provider_user_texts.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        provider_user_texts_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().filter_map(|message| match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        faux_assistant_message("Response to custom message", Default::default())
    })]);

    let context = AgentContext {
        system_prompt: "You are helpful.".to_owned(),
        messages: vec![AgentMessage::custom(json!({
            "role": "custom",
            "text": "Hook content",
            "timestamp": now_millis()
        }))],
        tools: Vec::new(),
    };
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.convert_to_llm = Some(Arc::new(CustomToUserConverter));

    let (messages, _events) = agent_loop_continue(context, config)
        .await
        .expect("continue");

    assert_eq!(
        *provider_user_texts.lock().expect("mutex"),
        vec!["Hook content".to_owned()]
    );
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["assistant"]
    );
    assert_eq!(text_of(&messages[0]), Some("Response to custom message"));
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_initializes_state_and_forwards_thinking_level() {
    let default_model = Model::faux("default-api", "default-provider", "default-model");
    let default_agent = Agent::new(AgentOptions::new(default_model.clone()));

    assert_eq!(default_agent.state().system_prompt, "");
    assert_eq!(default_agent.state().model, default_model);
    assert_eq!(default_agent.state().thinking_level, ThinkingLevel::Off);
    assert!(default_agent.state().tools.is_empty());
    assert!(default_agent.state().messages.is_empty());
    assert!(!default_agent.state().is_streaming);
    assert!(default_agent.state().streaming_message.is_none());
    assert!(default_agent.state().pending_tool_calls.is_empty());
    assert!(default_agent.state().error_message.is_none());

    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let observed_reasoning = Arc::new(Mutex::new(Vec::new()));
    let observed_reasoning_ref = observed_reasoning.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        observed_reasoning_ref
            .lock()
            .expect("mutex")
            .push(options.reasoning);
        faux_assistant_message("ok", Default::default())
    })]);

    let mut options = AgentOptions::new(registration.get_model());
    options.system_prompt = "You are helpful.".to_owned();
    options.thinking_level = ThinkingLevel::Low;
    options.messages = vec![AgentMessage::User(UserMessage::text("Existing"))];
    let agent = Agent::new(options);

    assert_eq!(agent.state().system_prompt, "You are helpful.");
    assert_eq!(agent.state().thinking_level, ThinkingLevel::Low);
    assert_eq!(agent.state().messages.len(), 1);

    let messages = agent.prompt("hello").await.expect("prompt");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(
        *observed_reasoning.lock().expect("mutex"),
        vec![Some(ThinkingLevel::Low)]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_basic_prompt_updates_state_with_response_text() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![faux_assistant_message("4", Default::default()).into()]);
    let mut options = AgentOptions::new(registration.get_model());
    options.system_prompt = "You are a helpful assistant. Keep your responses concise.".to_owned();
    options.thinking_level = ThinkingLevel::Off;
    options.tools = Vec::new();
    let agent = Agent::new(options);

    let messages = agent
        .prompt("What is 2+2? Answer with just the number.")
        .await
        .expect("prompt");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert!(!agent.state().is_streaming);
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .map(role_of)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(text_of(&agent.state().messages[1]), Some("4"));
    registration.unregister();
}

#[test]
fn agent_stateful_wrapper_state_mutators_update_fields_without_notifications() {
    let initial_model = Model::faux("state-api", "state-provider", "state-model");
    let agent = Agent::new(AgentOptions::new(initial_model));
    let seen_events = Arc::new(Mutex::new(Vec::new()));
    let seen_events_ref = seen_events.clone();
    let listener_id = agent.subscribe(move |event| {
        seen_events_ref
            .lock()
            .expect("mutex")
            .push(event_name(event).to_owned());
    });

    let next_model = Model::faux("state-api", "state-provider", "state-next-model");
    let tool = AgentTool {
        definition: Tool {
            name: "test".to_owned(),
            description: "test tool".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        label: "Test".to_owned(),
        execution_mode: Some(ToolExecutionMode::Parallel),
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: Arc::new(Mutex::new(Vec::new())),
        }),
    };
    let user_message = AgentMessage::User(UserMessage::text("Hello"));
    let assistant_message =
        AgentMessage::Assistant(faux_assistant_message("Hi", Default::default()));

    {
        let mut state = agent.state_mut();
        state.system_prompt = "Custom prompt".to_owned();
        state.model = next_model.clone();
        state.thinking_level = ThinkingLevel::High;
        state.tools = vec![tool.clone()];
        state.messages = vec![user_message.clone()];
        state.messages.push(assistant_message.clone());
    }

    {
        let state = agent.state();
        assert_eq!(state.system_prompt, "Custom prompt");
        assert_eq!(state.model, next_model);
        assert_eq!(state.thinking_level, ThinkingLevel::High);
        assert_eq!(state.tools.len(), 1);
        assert_eq!(state.tools[0].definition.name, "test");
        assert_eq!(state.messages, vec![user_message, assistant_message]);
    }
    assert!(seen_events.lock().expect("mutex").is_empty());

    agent.unsubscribe(listener_id);
    agent.state_mut().system_prompt = "Another prompt".to_owned();
    assert!(seen_events.lock().expect("mutex").is_empty());
}

#[tokio::test]
async fn agent_stateful_wrapper_preserves_thinking_content_blocks() {
    let mut model_def = FauxModelDefinition::new("faux-reasoning");
    model_def.reasoning = true;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![model_def],
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            vec![faux_thinking("step by step"), faux_text("4")],
            Default::default(),
        )
        .into(),
    ]);

    let mut options = AgentOptions::new(registration.get_model());
    options.thinking_level = ThinkingLevel::Low;
    let agent = Agent::new(options);

    agent.prompt("What is 2+2?").await.expect("prompt");

    let state = agent.state();
    let Some(AgentMessage::Assistant(assistant)) = state.messages.get(1) else {
        panic!("expected assistant message");
    };
    assert_eq!(
        assistant.content,
        vec![
            AssistantContent::Thinking(ThinkingContent::new("step by step")),
            AssistantContent::Text(TextContent::new("4")),
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_maintains_context_across_prompt_turns() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Nice to meet you, Alice.", Default::default()).into(),
        faux_response_factory(|context, _, _, _| {
            let has_alice = context.messages.iter().any(|message| {
                match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => text.contains("Alice"),
                    UserContentValue::Blocks(blocks) => blocks.iter().any(|block| {
                        matches!(block, UserContent::Text(text) if text.text.contains("Alice"))
                    }),
                },
                _ => false,
            }
            });
            faux_assistant_message(
                if has_alice {
                    "Your name is Alice."
                } else {
                    "I do not know your name."
                },
                Default::default(),
            )
        }),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent
        .prompt("My name is Alice.")
        .await
        .expect("first prompt");
    let second_turn = agent
        .prompt("What is my name?")
        .await
        .expect("second prompt");

    assert_eq!(agent.state().messages.len(), 4);
    assert_eq!(
        text_of(second_turn.last().expect("assistant")),
        Some("Your name is Alice.")
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_tracks_pending_tool_calls_during_events() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call(
                "echo",
                json!({ "value": "abc" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("calc-1".to_owned()),
            ),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = AgentTool {
        definition: Tool {
            name: "echo".to_owned(),
            description: "Echo tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        },
        label: "Echo".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(EchoExecutor {
            executed: executed.clone(),
        }),
    };
    let mut options = AgentOptions::new(registration.get_model());
    options.tools = vec![tool];
    let agent = Arc::new(Agent::new(options));
    let observations = Arc::new(Mutex::new(Vec::<(String, Vec<String>)>::new()));
    let observations_ref = observations.clone();
    let agent_ref = agent.clone();
    agent.subscribe(move |event| match event {
        AgentEvent::ToolExecutionStart { .. } => {
            observations_ref.lock().expect("mutex").push((
                "tool_execution_start".to_owned(),
                agent_ref
                    .state()
                    .pending_tool_calls
                    .iter()
                    .cloned()
                    .collect(),
            ));
        }
        AgentEvent::ToolExecutionEnd { .. } => {
            observations_ref.lock().expect("mutex").push((
                "tool_execution_end".to_owned(),
                agent_ref
                    .state()
                    .pending_tool_calls
                    .iter()
                    .cloned()
                    .collect(),
            ));
        }
        _ => {}
    });

    let messages = agent.prompt("run tool").await.expect("prompt");

    assert_eq!(*executed.lock().expect("mutex"), vec!["abc".to_owned()]);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "toolResult", "assistant"]
    );
    assert_eq!(text_of(&agent.state().messages[2]), Some("echoed: abc"));
    assert!(agent.state().pending_tool_calls.is_empty());
    assert_eq!(
        *observations.lock().expect("mutex"),
        vec![
            ("tool_execution_start".to_owned(), vec!["calc-1".to_owned()]),
            ("tool_execution_end".to_owned(), Vec::<String>::new())
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_emits_streaming_lifecycle_updates() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("1 2 3 4 5", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_ref = events.clone();
    agent.subscribe(move |event| {
        events_ref
            .lock()
            .expect("mutex")
            .push(event_name(event).to_owned());
    });

    agent.prompt("Count from 1 to 5.").await.expect("prompt");

    let events = events.lock().expect("mutex").clone();
    for required in [
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ] {
        assert!(events.contains(&required.to_owned()), "missing {required}");
    }
    let index_of = |name: &str| {
        events
            .iter()
            .position(|event| event == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    assert!(index_of("agent_start") < index_of("message_start"));
    assert!(index_of("message_start") < index_of("message_end"));
    assert!(index_of("message_end") < index_of("agent_end"));
    assert!(!agent.state().is_streaming);
    assert_eq!(agent.state().messages.len(), 2);
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_updates_state_and_notifies_subscribers() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_listener = seen.clone();
    let listener_id = agent.subscribe(move |event| {
        seen_for_listener
            .lock()
            .expect("mutex")
            .push(event_name(event).to_owned());
    });

    assert!(seen.lock().expect("mutex").is_empty());
    agent.state_mut().system_prompt = "Test prompt".to_owned();
    assert!(seen.lock().expect("mutex").is_empty());

    let messages = agent.prompt("hello").await.expect("prompt");
    assert_eq!(messages.len(), 2);
    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(text_of(&agent.state().messages[1]), Some("ok"));
    assert!(!agent.state().is_streaming);
    assert!(
        seen.lock()
            .expect("mutex")
            .contains(&"agent_end".to_owned())
    );

    let seen_before_unsubscribe = seen.lock().expect("mutex").len();
    agent.unsubscribe(listener_id);
    agent.state_mut().system_prompt = "Another prompt".to_owned();
    assert_eq!(seen.lock().expect("mutex").len(), seen_before_unsubscribe);
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_awaits_async_subscribers_before_finishing_run() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_started = Arc::new(AtomicBool::new(false));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let listener_started_ref = listener_started.clone();
    let listener_finished_ref = listener_finished.clone();
    let release_listener_ref = release_listener.clone();
    agent.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let listener_finished = listener_finished_ref.clone();
        let release_listener = release_listener_ref.clone();
        async move {
            if matches!(event, AgentEvent::AgentEnd { .. }) {
                listener_started.store(true, Ordering::SeqCst);
                release_listener.notified().await;
                listener_finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let prompt_task = tokio::spawn(async move {
        let messages = agent.prompt("hello").await.expect("prompt");
        (agent, messages)
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    assert!(listener_started.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(
        !prompt_task.is_finished(),
        "prompt should wait for async agent_end subscriber"
    );

    release_listener.notify_one();
    let (agent, messages) = prompt_task.await.expect("prompt task");
    assert!(listener_finished.load(Ordering::SeqCst));
    assert_eq!(messages.len(), 2);
    assert_eq!(agent.state().messages.len(), 2);
    assert!(!agent.state().is_streaming);
    assert!(agent.state().streaming_message.is_none());
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_wait_for_idle_waits_for_async_subscribers() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_started = Arc::new(AtomicBool::new(false));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let listener_started_notify = Arc::new(tokio::sync::Notify::new());
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let listener_started_ref = listener_started.clone();
    let listener_finished_ref = listener_finished.clone();
    let listener_started_notify_ref = listener_started_notify.clone();
    let release_listener_ref = release_listener.clone();
    agent.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let listener_finished = listener_finished_ref.clone();
        let listener_started_notify = listener_started_notify_ref.clone();
        let release_listener = release_listener_ref.clone();
        async move {
            if matches!(
                event,
                AgentEvent::MessageEnd {
                    message: AgentMessage::Assistant(_)
                }
            ) {
                listener_started.store(true, Ordering::SeqCst);
                listener_started_notify.notify_waiters();
                release_listener.notified().await;
                listener_finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let prompt_future = agent.prompt("hello");
    tokio::pin!(prompt_future);
    tokio::select! {
        result = &mut prompt_future => panic!("prompt resolved before async listener blocked: {result:?}"),
        _ = listener_started_notify.notified() => {}
    }
    assert!(listener_started.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(agent.state().is_streaming);

    let idle_future = agent.wait_for_idle();
    tokio::pin!(idle_future);
    tokio::select! {
        _ = &mut idle_future => panic!("wait_for_idle resolved before async listener finished"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
    }
    assert!(agent.state().is_streaming);

    release_listener.notify_one();
    let (messages, ()) = tokio::join!(prompt_future, idle_future);
    let messages = messages.expect("prompt");
    assert!(listener_finished.load(Ordering::SeqCst));
    assert_eq!(messages.len(), 2);
    assert!(!agent.state().is_streaming);
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_passes_active_abort_flag_to_subscribers() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("abort target", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observations_ref = observations.clone();
    agent.subscribe_with_abort_flag(move |event, abort_flag| {
        if matches!(event, AgentEvent::AgentStart) {
            observations_ref
                .lock()
                .expect("mutex")
                .push(abort_flag.load(Ordering::SeqCst));
            abort_flag.store(true, Ordering::SeqCst);
            observations_ref
                .lock()
                .expect("mutex")
                .push(abort_flag.load(Ordering::SeqCst));
        }
    });

    let messages = tokio::time::timeout(std::time::Duration::from_secs(1), agent.prompt("hello"))
        .await
        .expect("prompt should finish after abort")
        .expect("prompt");
    assert_eq!(
        *observations.lock().expect("mutex"),
        vec![false, true],
        "subscriber should receive the live abort flag before the run starts provider IO"
    );
    let AgentMessage::Assistant(assistant) = messages.last().expect("assistant") else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert!(!agent.state().is_streaming);
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_reduces_state_before_subscribers_run() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observations_ref = observations.clone();
    agent.subscribe(move |event| match event {
        AgentEvent::MessageStart { message } => {
            observations_ref
                .lock()
                .expect("mutex")
                .push(format!("start:{}", role_of(message)));
        }
        AgentEvent::MessageEnd { message } => {
            observations_ref
                .lock()
                .expect("mutex")
                .push(format!("end:{}", role_of(message)));
        }
        AgentEvent::AgentEnd { .. } => {
            observations_ref
                .lock()
                .expect("mutex")
                .push("agent_end".to_owned());
        }
        _ => {}
    });

    agent.prompt("hello").await.expect("prompt");

    assert_eq!(
        *observations.lock().expect("mutex"),
        vec![
            "start:user".to_owned(),
            "end:user".to_owned(),
            "start:assistant".to_owned(),
            "end:assistant".to_owned(),
            "agent_end".to_owned(),
        ]
    );
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .map(role_of)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_rejects_prompt_and_continue_while_streaming() {
    let agent = Agent::new(AgentOptions::new(Model::faux(
        "busy-api",
        "busy-provider",
        "busy-model",
    )));
    agent.state_mut().is_streaming = true;

    let prompt_error = agent
        .prompt("Second message")
        .await
        .expect_err("prompt busy");
    assert_eq!(
        prompt_error,
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    );

    let continue_error = agent.continue_run().await.expect_err("continue busy");
    assert_eq!(
        continue_error,
        "Agent is already processing. Wait for completion before continuing."
    );
    assert!(agent.state().error_message.is_none());
}

#[tokio::test]
async fn agent_stateful_wrapper_validates_continue_tail_before_loop() {
    let agent = Agent::new(AgentOptions::new(Model::faux(
        "continue-api",
        "continue-provider",
        "continue-model",
    )));

    let empty_error = agent.continue_run().await.expect_err("empty continue");
    assert_eq!(empty_error, "No messages to continue from");

    agent.state_mut().messages = vec![
        AgentMessage::User(UserMessage::text("Initial")),
        AgentMessage::Assistant(faux_assistant_message(
            "Initial response",
            Default::default(),
        )),
    ];

    let assistant_tail_error = agent.continue_run().await.expect_err("assistant tail");
    assert_eq!(
        assistant_tail_error,
        "Cannot continue from message role: assistant"
    );
    assert!(!agent.state().is_streaming);
    assert!(agent.state().error_message.is_none());
}

#[tokio::test]
async fn agent_stateful_wrapper_forwards_session_id_to_provider_options() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let observed_sessions = Arc::new(Mutex::new(Vec::new()));
    let observed_sessions_ref = observed_sessions.clone();
    let observed_sessions_ref_2 = observed_sessions.clone();
    registration.set_responses(vec![
        faux_response_factory(move |_, options, _, _| {
            observed_sessions_ref
                .lock()
                .expect("mutex")
                .push(options.stream.session_id.clone());
            faux_assistant_message("ok", Default::default())
        }),
        faux_response_factory(move |_, options, _, _| {
            observed_sessions_ref_2
                .lock()
                .expect("mutex")
                .push(options.stream.session_id.clone());
            faux_assistant_message("ok again", Default::default())
        }),
    ]);

    let mut options = AgentOptions::new(registration.get_model());
    options.stream_options.stream.session_id = Some("session-abc".to_owned());
    let mut agent = Agent::new(options);

    agent.prompt("hello").await.expect("first prompt");
    agent.state_mut().messages.clear();
    agent.set_session_id(Some("session-def".to_owned()));
    assert_eq!(agent.session_id(), Some("session-def"));
    agent.prompt("hello again").await.expect("second prompt");

    assert_eq!(
        *observed_sessions.lock().expect("mutex"),
        vec![
            Some("session-abc".to_owned()),
            Some("session-def".to_owned())
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_prompt_with_images_builds_multimodal_user_message() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let observed_blocks = Arc::new(Mutex::new(None));
    let observed_blocks_ref = observed_blocks.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        let blocks = context.messages.first().and_then(|message| match message {
            Message::User(user) => match &user.content {
                UserContentValue::Blocks(blocks) => Some(blocks.clone()),
                UserContentValue::Plain(_) => None,
            },
            _ => None,
        });
        *observed_blocks_ref.lock().expect("mutex") = blocks;
        faux_assistant_message("saw image", Default::default())
    })]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let image = ImageContent {
        data: "data:image/png;base64,AAAA".to_owned(),
        mime_type: "image/png".to_owned(),
    };

    let messages = agent
        .prompt_with_images("Describe this", vec![image.clone()])
        .await
        .expect("prompt with images");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    let observed = observed_blocks
        .lock()
        .expect("mutex")
        .clone()
        .expect("multimodal blocks");
    assert_eq!(observed.len(), 2);
    assert!(matches!(
        &observed[0],
        UserContent::Text(text) if text.text == "Describe this"
    ));
    assert!(matches!(&observed[1], UserContent::Image(seen) if seen == &image));
    registration.unregister();
}

#[tokio::test]
async fn agent_stateful_wrapper_persists_provider_start_failures_as_error_messages() {
    let model = Model::faux(
        "missing-agent-test-api",
        "missing-provider",
        "missing-model",
    );
    let agent = Agent::new(AgentOptions::new(model));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_listener = seen.clone();
    agent.subscribe(move |event| {
        seen_for_listener
            .lock()
            .expect("mutex")
            .push(event_name(event).to_owned());
    });

    let messages = agent.prompt("hello").await.expect("prompt resolves");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    let AgentMessage::Assistant(assistant) = messages.last().expect("assistant") else {
        panic!("assistant");
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(text_of(messages.last().expect("assistant")), Some(""));
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("No API provider registered for api: missing-agent-test-api")
    );
    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(
        agent.state().error_message.as_deref(),
        Some("No API provider registered for api: missing-agent-test-api")
    );
    assert!(!agent.state().is_streaming);
    assert_eq!(
        *seen.lock().expect("mutex"),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

#[tokio::test]
async fn agent_abort_handle_cancels_active_provider_stream() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
    ]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.abort();
    assert!(!agent.state().is_streaming);

    let abort_handle = agent.abort_handle();
    let abort_task = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        abort_handle.store(true, Ordering::SeqCst);
    });

    let messages = agent.prompt("hello").await.expect("prompt resolves");
    abort_task.await.expect("abort task");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    let AgentMessage::Assistant(assistant) = messages.last().expect("assistant") else {
        panic!("assistant");
    };
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("Request was aborted")
    );
    assert_eq!(
        agent.state().error_message.as_deref(),
        Some("Request was aborted")
    );
    assert!(!agent.state().is_streaming);
    registration.unregister();
}

#[tokio::test]
async fn agent_queues_steering_and_follow_up_without_mutating_state() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent.steer(Message::User(UserMessage::text("Steering message")));
    agent.follow_up(Message::User(UserMessage::text("Follow-up message")));

    assert!(agent.state().messages.is_empty());
    registration.unregister();
}

#[tokio::test]
async fn agent_has_queued_messages_tracks_steering_follow_up_and_clears() {
    let agent = Agent::new(AgentOptions::new(Model::faux(
        "queue-api",
        "queue-provider",
        "queue-model",
    )));

    assert!(!agent.has_queued_messages());

    agent.steer(Message::User(UserMessage::text("Steering message")));
    assert!(agent.has_queued_messages());

    agent.clear_steering_queue();
    assert!(!agent.has_queued_messages());

    agent.follow_up(Message::User(UserMessage::text("Follow-up message")));
    assert!(agent.has_queued_messages());

    agent.abort();
    assert!(
        agent.has_queued_messages(),
        "low-level Agent::abort matches Pi Agent.abort by cancelling the active run without clearing queued messages"
    );
    agent.clear_follow_up_queue();
    assert!(!agent.has_queued_messages());

    agent.steer(Message::User(UserMessage::text("Steering message")));
    agent.follow_up(Message::User(UserMessage::text("Follow-up message")));
    assert!(agent.has_queued_messages());

    agent.clear_all_queues();
    assert!(!agent.has_queued_messages());
}

#[tokio::test]
async fn agent_prompt_injects_queued_steering_before_first_provider_request() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_user_texts = Arc::new(Mutex::new(Vec::new()));
    let request_user_texts_ref = request_user_texts.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_user_texts_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().filter_map(|message| match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        faux_assistant_message("Processed", Default::default())
    })]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.steer(Message::User(UserMessage::text("Queued steering")));

    let messages = agent.prompt("Initial").await.expect("prompt");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "user", "assistant"]
    );
    assert_eq!(
        *request_user_texts.lock().expect("mutex"),
        vec!["Initial".to_owned(), "Queued steering".to_owned()]
    );
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .map(role_of)
            .collect::<Vec<_>>(),
        vec!["user", "user", "assistant"]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_clear_queues_prevents_queued_messages_from_running() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_user_texts = Arc::new(Mutex::new(Vec::new()));
    let request_user_texts_ref = request_user_texts.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_user_texts_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().filter_map(|message| match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        faux_assistant_message("Processed", Default::default())
    })]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.steer(Message::User(UserMessage::text("Steering should clear")));
    agent.follow_up(Message::User(UserMessage::text("Follow-up should clear")));
    agent.clear_all_queues();

    let messages = agent.prompt("Initial").await.expect("prompt");

    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(
        *request_user_texts.lock().expect("mutex"),
        vec!["Initial".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_reset_clears_state_and_queued_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Processed", Default::default()).into(),
    ]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.state_mut().messages = vec![AgentMessage::User(UserMessage::text("Before reset"))];
    agent.state_mut().is_streaming = true;
    agent.state_mut().streaming_message = Some(AgentMessage::User(UserMessage::text("streaming")));
    agent
        .state_mut()
        .pending_tool_calls
        .insert("tool-1".to_owned());
    agent.state_mut().error_message = Some("boom".to_owned());
    agent.steer(Message::User(UserMessage::text("Steering should clear")));

    agent.reset();
    let messages = agent.prompt("After reset").await.expect("prompt");

    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert!(!agent.state().is_streaming);
    assert!(agent.state().streaming_message.is_none());
    assert!(agent.state().pending_tool_calls.is_empty());
    assert!(agent.state().error_message.is_none());
    registration.unregister();
}

#[tokio::test]
async fn agent_reset_does_not_finish_active_run_or_allow_concurrent_prompt() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("first", Default::default()).into(),
        faux_assistant_message("second", Default::default()).into(),
    ]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_started = Arc::new(tokio::sync::Notify::new());
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let did_block = Arc::new(AtomicBool::new(false));
    let listener_started_ref = listener_started.clone();
    let release_listener_ref = release_listener.clone();
    let did_block_ref = did_block.clone();
    agent.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let release_listener = release_listener_ref.clone();
        let did_block = did_block_ref.clone();
        async move {
            if matches!(
                event,
                AgentEvent::MessageEnd {
                    message: AgentMessage::Assistant(_)
                }
            ) && !did_block.swap(true, Ordering::SeqCst)
            {
                listener_started.notify_waiters();
                release_listener.notified().await;
            }
        }
    });

    let first_prompt = agent.prompt("first");
    tokio::pin!(first_prompt);
    tokio::select! {
        result = &mut first_prompt => panic!("prompt resolved before listener blocked: {result:?}"),
        _ = listener_started.notified() => {}
    }
    assert!(agent.state().is_streaming);

    agent.reset();
    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.is_empty());

    let prompt_error = agent
        .prompt("concurrent")
        .await
        .expect_err("reset must not clear active run guard");
    assert_eq!(
        prompt_error,
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    );

    let idle = agent.wait_for_idle();
    tokio::pin!(idle);
    tokio::select! {
        _ = &mut idle => panic!("wait_for_idle resolved while the pre-reset run was still active"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
    }

    release_listener.notify_one();
    let (first_messages, ()) = tokio::join!(first_prompt, idle);
    assert_eq!(
        first_messages
            .expect("first prompt")
            .iter()
            .map(role_of)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.is_empty());

    let second_messages = agent.prompt("after reset").await.expect("second prompt");
    assert_eq!(
        second_messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(agent.state().messages.len(), 2);
    registration.unregister();
}

#[tokio::test]
async fn agent_continue_from_user_tail_gets_assistant_response() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let observed_user_texts = Arc::new(Mutex::new(Vec::new()));
    let observed_user_texts_ref = observed_user_texts.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        observed_user_texts_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().filter_map(|message| match message {
                Message::User(user) => match &user.content {
                    UserContentValue::Plain(text) => Some(text.clone()),
                    UserContentValue::Blocks(_) => None,
                },
                _ => None,
            }));
        faux_assistant_message("HELLO WORLD", Default::default())
    })]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    {
        let mut state = agent.state_mut();
        state.system_prompt =
            "You are a helpful assistant. Follow instructions exactly.".to_owned();
        state.thinking_level = ThinkingLevel::Off;
        state.tools = Vec::new();
        state.messages = vec![AgentMessage::User(UserMessage::text(
            "Say exactly: HELLO WORLD",
        ))];
    }

    let messages = agent.continue_run().await.expect("continue");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(
        *observed_user_texts.lock().expect("mutex"),
        vec!["Say exactly: HELLO WORLD".to_owned()]
    );
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["assistant"]
    );
    assert_eq!(text_of(&messages[0]), Some("HELLO WORLD"));
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .map(role_of)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(
        text_of(agent.state().messages.last().expect("assistant")),
        Some("HELLO WORLD")
    );
    assert!(!agent.state().is_streaming);
    registration.unregister();
}

#[tokio::test]
async fn agent_continue_from_tool_result_tail_gets_assistant_response() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let observed_roles = Arc::new(Mutex::new(Vec::new()));
    let observed_tool_result = Arc::new(Mutex::new(None));
    let observed_roles_ref = observed_roles.clone();
    let observed_tool_result_ref = observed_tool_result.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        observed_roles_ref
            .lock()
            .expect("mutex")
            .extend(context.messages.iter().map(llm_role_of).map(str::to_owned));
        *observed_tool_result_ref.lock().expect("mutex") =
            context.messages.last().and_then(|message| match message {
                Message::ToolResult(result) => Some((
                    result.tool_call_id.clone(),
                    result.tool_name.clone(),
                    llm_text_of(message).unwrap_or_default().to_owned(),
                )),
                _ => None,
            });
        faux_assistant_message("The answer is 8.", Default::default())
    })]);

    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.state_mut().messages = vec![
        AgentMessage::User(UserMessage::text("What is 5 + 3?")),
        AgentMessage::Assistant(faux_assistant_message(
            vec![
                AssistantContent::text("Let me calculate that."),
                faux_tool_call(
                    "calculate",
                    json!({ "expression": "5 + 3" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("calc-1".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )),
        AgentMessage::ToolResult(ToolResultMessage {
            tool_call_id: "calc-1".to_owned(),
            tool_name: "calculate".to_owned(),
            content: vec![ToolResultContent::text("5 + 3 = 8")],
            details: None,
            is_error: false,
            timestamp: now_millis(),
        }),
    ];

    let messages = agent.continue_run().await.expect("continue");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(
        *observed_roles.lock().expect("mutex"),
        vec![
            "user".to_owned(),
            "assistant".to_owned(),
            "toolResult".to_owned()
        ]
    );
    assert_eq!(
        *observed_tool_result.lock().expect("mutex"),
        Some((
            "calc-1".to_owned(),
            "calculate".to_owned(),
            "5 + 3 = 8".to_owned()
        ))
    );
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["assistant"]
    );
    assert_eq!(agent.state().messages.len(), 4);
    assert_eq!(
        text_of(agent.state().messages.last().expect("assistant")),
        Some("The answer is 8.")
    );
    assert!(!agent.state().is_streaming);
    registration.unregister();
}

#[tokio::test]
async fn agent_continue_from_assistant_tail_processes_queued_follow_up() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Processed", Default::default()).into(),
    ]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.state_mut().messages = vec![
        AgentMessage::User(UserMessage::text("Initial")),
        AgentMessage::Assistant(faux_assistant_message(
            "Initial response",
            Default::default(),
        )),
    ];
    agent.follow_up(Message::User(UserMessage::text("Queued follow-up")));

    let messages = agent.continue_run().await.expect("continue");

    assert_eq!(registration.state().call_count(), 1);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|message| role_of(message))
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert!(matches!(
        &agent.state().messages[2],
        AgentMessage::User(user) if matches!(&user.content, UserContentValue::Plain(text) if text == "Queued follow-up")
    ));
    assert_eq!(text_of(&agent.state().messages[3]), Some("Processed"));
    registration.unregister();
}

#[tokio::test]
async fn agent_continue_from_assistant_tail_drains_steering_one_at_a_time() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("Processed 1", Default::default()).into(),
        faux_assistant_message("Processed 2", Default::default()).into(),
    ]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    agent.state_mut().messages = vec![
        AgentMessage::User(UserMessage::text("Initial")),
        AgentMessage::Assistant(faux_assistant_message(
            "Initial response",
            Default::default(),
        )),
    ];
    agent.steer(Message::User(UserMessage::text("Steering 1")));
    agent.steer(Message::User(UserMessage::text("Steering 2")));

    let messages = agent.continue_run().await.expect("continue");

    assert_eq!(registration.state().call_count(), 2);
    assert_eq!(
        messages.iter().map(role_of).collect::<Vec<_>>(),
        vec!["user", "assistant", "user", "assistant"]
    );
    assert_eq!(
        agent
            .state()
            .messages
            .iter()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|message| role_of(message))
            .collect::<Vec<_>>(),
        vec!["user", "assistant", "user", "assistant"]
    );
    assert_eq!(
        text_of(agent.state().messages.last().expect("last")),
        Some("Processed 2")
    );
    registration.unregister();
}

#[test]
fn harness_utilities_cover_utf8_truncation_compaction_and_uuid() {
    let sample = "alpha\nβeta\n終";
    assert_eq!(utf8_byte_len(sample), sample.len());
    assert!(truncate_head_utf8(sample, 8).truncated);
    assert!(truncate_tail_utf8(sample, 8).truncated);

    let messages = vec![Message::User(UserMessage::text("x".repeat(400)))];
    assert!(should_compact(
        &messages,
        &CompactionSettings {
            max_context_tokens: 10,
            threshold_percent: 50
        }
    ));

    let id = uuidv7();
    assert_eq!(id.len(), 36);
    assert_eq!(id.chars().nth(14), Some('7'));
    assert!(matches!(id.chars().nth(19), Some('8' | '9' | 'a' | 'b')));
    assert_eq!(id.chars().filter(|character| *character == '-').count(), 4);
}
