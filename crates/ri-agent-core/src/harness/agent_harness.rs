use crate::{
    agent_loop::{AgentLoopConfig, agent_loop_prompt_messages},
    harness::{
        CustomMessageContent, LocalExecutionEnv, PromptTemplate, Session, SessionMessage, Skill,
        format_prompt_template_invocation, format_skill_invocation,
    },
    types::{
        AgentContext, AgentEvent, AgentEventSink, AgentLoopTurnUpdate, AgentMessage,
        AgentNextTurnContext, AgentNextTurnPreparer, AgentTool, AgentToolCallHook,
        AgentToolCallHookContext, AgentToolResult, AgentToolResultContent, AgentToolResultHook,
        AgentToolResultHookContext, QueueMode, ToolExecutionMode,
    },
};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use ri_llm_provider::{
    AssistantMessage, CacheRetention, ImageContent, Message, Model, ProviderPayloadHook,
    SimpleStreamOptions, ThinkingLevel, Transport, UserContent, UserContentValue, UserMessage,
    now_millis,
};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Display,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use thiserror::Error;
use tokio::sync::Notify;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentHarnessPhase {
    Idle,
    Turn,
    Compaction,
    BranchSummary,
    Retry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentHarnessErrorCode {
    Busy,
    InvalidArgument,
    InvalidState,
    Session,
    Unknown,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct AgentHarnessError {
    pub code: AgentHarnessErrorCode,
    pub message: String,
}

impl AgentHarnessError {
    pub fn new(code: AgentHarnessErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn session(error: impl Display) -> Self {
        Self::new(AgentHarnessErrorCode::Session, error.to_string())
    }

    fn unknown(error: impl Into<String>) -> Self {
        Self::new(AgentHarnessErrorCode::Unknown, error)
    }
}

#[derive(Debug, Clone)]
pub struct QueueUpdateEvent {
    pub steer: Vec<AgentMessage>,
    pub follow_up: Vec<AgentMessage>,
    pub next_turn: Vec<AgentMessage>,
}

#[derive(Debug, Clone)]
pub struct AbortResult {
    pub cleared_steer: Vec<AgentMessage>,
    pub cleared_follow_up: Vec<AgentMessage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentHarnessResources {
    pub skills: Vec<Skill>,
    pub prompt_templates: Vec<PromptTemplate>,
}

#[derive(Clone)]
pub struct SystemPromptContext {
    pub env: LocalExecutionEnv,
    pub session: Session,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub active_tools: Vec<AgentTool>,
    pub resources: AgentHarnessResources,
}

pub type SystemPromptProvider =
    Arc<dyn Fn(SystemPromptContext) -> Result<String, AgentHarnessError> + Send + Sync>;

#[derive(Debug, Clone)]
pub struct ResourcesUpdateEvent {
    pub resources: AgentHarnessResources,
    pub previous_resources: AgentHarnessResources,
}

#[derive(Debug, Clone)]
pub struct ModelSelectEvent {
    pub model: Model,
    pub previous_model: Model,
}

#[derive(Debug, Clone)]
pub struct ThinkingLevelSelectEvent {
    pub level: ThinkingLevel,
    pub previous_level: ThinkingLevel,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderAuth {
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
}

pub type ProviderAuthProvider =
    Arc<dyn Fn(&Model) -> Result<ProviderAuth, AgentHarnessError> + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionPatch<T> {
    Set(T),
    Clear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderMapPatch {
    Replace(BTreeMap<String, String>),
    Merge(BTreeMap<String, Option<String>>),
    Clear,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataMapPatch {
    Replace(Map<String, Value>),
    Merge(BTreeMap<String, Option<Value>>),
    Clear,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentHarnessStreamOptionsPatch {
    pub transport: Option<OptionPatch<Transport>>,
    pub timeout_ms: Option<OptionPatch<u64>>,
    pub max_retries: Option<OptionPatch<u32>>,
    pub max_retry_delay_ms: Option<OptionPatch<u64>>,
    pub cache_retention: Option<OptionPatch<CacheRetention>>,
    pub headers: Option<HeaderMapPatch>,
    pub metadata: Option<MetadataMapPatch>,
}

#[derive(Debug, Clone)]
pub struct BeforeProviderRequestEvent {
    pub model: Model,
    pub session_id: String,
    pub stream_options: SimpleStreamOptions,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeforeProviderRequestResult {
    pub stream_options: Option<AgentHarnessStreamOptionsPatch>,
}

pub type BeforeProviderRequestHook = Arc<
    dyn Fn(
            BeforeProviderRequestEvent,
        ) -> Result<Option<BeforeProviderRequestResult>, AgentHarnessError>
        + Send
        + Sync,
>;

#[derive(Debug, Clone)]
pub struct BeforeProviderPayloadEvent {
    pub model: Model,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeforeProviderPayloadResult {
    pub payload: Value,
}

pub type BeforeProviderPayloadHook = Arc<
    dyn Fn(
            BeforeProviderPayloadEvent,
        ) -> Result<Option<BeforeProviderPayloadResult>, AgentHarnessError>
        + Send
        + Sync,
>;

#[derive(Debug, Clone)]
pub enum AgentHarnessEvent {
    Agent(AgentEvent),
    QueueUpdate(QueueUpdateEvent),
    Abort(AbortResult),
    ResourcesUpdate(ResourcesUpdateEvent),
    ModelSelect(ModelSelectEvent),
    ThinkingLevelSelect(ThinkingLevelSelectEvent),
    SavePoint { had_pending_mutations: bool },
    Settled { next_turn_count: usize },
}

pub type AgentHarnessListener = Arc<dyn Fn(&AgentHarnessEvent) + Send + Sync>;
type AgentHarnessAsyncListener =
    Arc<dyn Fn(AgentHarnessEvent) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone)]
enum AgentHarnessListenerEntry {
    Sync(AgentHarnessListener),
    Async(AgentHarnessAsyncListener),
}
pub type BeforeAgentStartHook = Arc<
    dyn Fn(BeforeAgentStartEvent) -> Result<Option<BeforeAgentStartResult>, AgentHarnessError>
        + Send
        + Sync,
>;
pub type ContextHook =
    Arc<dyn Fn(ContextEvent) -> Result<Option<ContextResult>, AgentHarnessError> + Send + Sync>;
pub type ToolCallHook =
    Arc<dyn Fn(ToolCallEvent) -> Result<Option<ToolCallResult>, AgentHarnessError> + Send + Sync>;
pub type ToolResultHook = Arc<
    dyn Fn(ToolResultEvent) -> Result<Option<ToolResultPatch>, AgentHarnessError> + Send + Sync,
>;

#[derive(Debug, Clone)]
pub struct BeforeAgentStartEvent {
    pub prompt: String,
    pub images: Vec<ImageContent>,
    pub system_prompt: String,
    pub resources: AgentHarnessResources,
}

#[derive(Debug, Clone, Default)]
pub struct BeforeAgentStartResult {
    pub messages: Option<Vec<AgentMessage>>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContextEvent {
    pub messages: Vec<AgentMessage>,
}

#[derive(Debug, Clone)]
pub struct ContextResult {
    pub messages: Vec<AgentMessage>,
}

#[derive(Debug, Clone)]
pub struct ToolCallEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallResult {
    pub input: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ToolResultEvent {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: Value,
    pub content: Vec<AgentToolResultContent>,
    pub details: Option<Value>,
    pub terminate: bool,
    pub is_error: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ToolResultPatch {
    pub content: Option<Vec<AgentToolResultContent>>,
    pub details: Option<Value>,
    pub terminate: Option<bool>,
}

#[derive(Clone)]
pub struct AgentHarnessOptions {
    pub env: LocalExecutionEnv,
    pub session: Session,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub system_prompt: String,
    pub system_prompt_provider: Option<SystemPromptProvider>,
    pub stream_options: SimpleStreamOptions,
    pub get_api_key_and_headers: Option<ProviderAuthProvider>,
    pub resources: AgentHarnessResources,
    pub tools: Vec<AgentTool>,
    pub active_tool_names: Option<Vec<String>>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub tool_execution: ToolExecutionMode,
    pub max_turns: usize,
}

impl AgentHarnessOptions {
    pub fn new(env: LocalExecutionEnv, session: Session, model: Model) -> Self {
        Self {
            env,
            session,
            model,
            thinking_level: ThinkingLevel::Off,
            system_prompt: String::new(),
            system_prompt_provider: None,
            stream_options: SimpleStreamOptions::default(),
            get_api_key_and_headers: None,
            resources: AgentHarnessResources::default(),
            tools: Vec::new(),
            active_tool_names: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
            max_turns: 16,
        }
    }
}

#[derive(Debug)]
struct HarnessMessageQueue {
    inner: Mutex<HarnessMessageQueueInner>,
}

#[derive(Debug)]
struct HarnessMessageQueueInner {
    mode: QueueMode,
    messages: VecDeque<AgentMessage>,
}

impl HarnessMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            inner: Mutex::new(HarnessMessageQueueInner {
                mode,
                messages: VecDeque::new(),
            }),
        }
    }

    fn enqueue(&self, message: impl Into<AgentMessage>) {
        self.inner.lock().messages.push_back(message.into());
    }

    fn mode(&self) -> QueueMode {
        self.inner.lock().mode
    }

    fn set_mode(&self, mode: QueueMode) {
        self.inner.lock().mode = mode;
    }

    fn drain_now(&self) -> Vec<AgentMessage> {
        let mut inner = self.inner.lock();
        match inner.mode {
            QueueMode::All => inner.messages.drain(..).collect(),
            QueueMode::OneAtATime => inner.messages.pop_front().into_iter().collect(),
        }
    }

    fn drain_all(&self) -> Vec<AgentMessage> {
        self.inner.lock().messages.drain(..).collect()
    }

    fn snapshot(&self) -> Vec<AgentMessage> {
        self.inner.lock().messages.iter().cloned().collect()
    }
}

struct HarnessQueuedMessageProvider {
    queue: Arc<HarnessMessageQueue>,
    emit_queue_update: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>,
}

#[async_trait]
impl crate::types::AgentQueuedMessageProvider for HarnessQueuedMessageProvider {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        let messages = self.queue.drain_now();
        if !messages.is_empty() {
            (self.emit_queue_update)().await;
        }
        Ok(messages)
    }
}

#[derive(Debug, Clone)]
enum HarnessSessionWrite {
    Message {
        message: Message,
    },
    ModelChange {
        provider: String,
        model_id: String,
    },
    ThinkingLevelChange {
        thinking_level: String,
    },
    Custom {
        custom_type: String,
        data: Option<Value>,
    },
    CustomMessage {
        custom_type: String,
        content: CustomMessageContent,
        display: bool,
        details: Option<Value>,
    },
    Label {
        target_id: String,
        label: Option<String>,
    },
    SessionName {
        name: String,
    },
}

struct HarnessNextTurnPreparer {
    env: LocalExecutionEnv,
    session: Session,
    model: Arc<Mutex<Model>>,
    thinking_level: Arc<Mutex<ThinkingLevel>>,
    system_prompt: Arc<Mutex<String>>,
    system_prompt_provider: Option<SystemPromptProvider>,
    stream_options: Arc<Mutex<SimpleStreamOptions>>,
    get_api_key_and_headers: Option<ProviderAuthProvider>,
    provider_request_hooks: Arc<Mutex<BTreeMap<u64, BeforeProviderRequestHook>>>,
    provider_payload_hooks: Arc<Mutex<BTreeMap<u64, BeforeProviderPayloadHook>>>,
    abort_flag: Arc<AtomicBool>,
    resources: Arc<Mutex<AgentHarnessResources>>,
    tools: Arc<Mutex<Vec<AgentTool>>>,
    active_tool_names: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentNextTurnPreparer for HarnessNextTurnPreparer {
    async fn prepare_next_turn(
        &self,
        context: AgentNextTurnContext,
    ) -> Result<Option<AgentLoopTurnUpdate>, String> {
        let model = self.model.lock().clone();
        let thinking_level = *self.thinking_level.lock();
        let resources = self.resources.lock().clone();
        let tools = self.tools.lock().clone();
        let active_tool_names = self.active_tool_names.lock().clone();
        let active_tools = active_tools_from(&tools, &active_tool_names);
        let stream_options = resolve_provider_stream_options(
            &model,
            thinking_level,
            &self.session,
            self.stream_options.lock().clone(),
            self.get_api_key_and_headers.as_ref(),
            &self.provider_request_hooks,
            &self.provider_payload_hooks,
            self.abort_flag.clone(),
        )
        .map_err(|error| error.message)?;
        let system_prompt = resolve_system_prompt_from_parts(
            self.system_prompt.lock().clone(),
            self.system_prompt_provider.as_ref(),
            SystemPromptContext {
                env: self.env.clone(),
                session: self.session.clone(),
                model: model.clone(),
                thinking_level,
                active_tools: active_tools.clone(),
                resources,
            },
        )
        .map_err(|error| error.message)?;
        Ok(Some(AgentLoopTurnUpdate {
            context: Some(AgentContext {
                system_prompt,
                messages: context.context.messages,
                tools: active_tools,
            }),
            model: Some(model),
            thinking_level: Some(thinking_level),
            stream_options: Some(stream_options),
        }))
    }
}

struct HarnessEventSink {
    listeners: Arc<Mutex<BTreeMap<u64, AgentHarnessListenerEntry>>>,
}

#[async_trait]
impl AgentEventSink for HarnessEventSink {
    async fn on_event(&self, event: &AgentEvent) {
        emit_to_async(&self.listeners, AgentHarnessEvent::Agent(event.clone())).await;
    }
}

pub struct AgentHarness {
    pub env: LocalExecutionEnv,
    session: Session,
    model: Arc<Mutex<Model>>,
    thinking_level: Arc<Mutex<ThinkingLevel>>,
    system_prompt: Arc<Mutex<String>>,
    system_prompt_provider: Option<SystemPromptProvider>,
    stream_options: Arc<Mutex<SimpleStreamOptions>>,
    get_api_key_and_headers: Option<ProviderAuthProvider>,
    resources: Arc<Mutex<AgentHarnessResources>>,
    tools: Arc<Mutex<Vec<AgentTool>>>,
    active_tool_names: Arc<Mutex<Vec<String>>>,
    steering_queue: Arc<HarnessMessageQueue>,
    follow_up_queue: Arc<HarnessMessageQueue>,
    next_turn_queue: Arc<HarnessMessageQueue>,
    phase: Mutex<AgentHarnessPhase>,
    listeners: Arc<Mutex<BTreeMap<u64, AgentHarnessListenerEntry>>>,
    before_agent_start_hooks: Mutex<BTreeMap<u64, BeforeAgentStartHook>>,
    context_hooks: Arc<Mutex<BTreeMap<u64, ContextHook>>>,
    tool_call_hooks: Arc<Mutex<BTreeMap<u64, ToolCallHook>>>,
    tool_result_hooks: Arc<Mutex<BTreeMap<u64, ToolResultHook>>>,
    provider_request_hooks: Arc<Mutex<BTreeMap<u64, BeforeProviderRequestHook>>>,
    provider_payload_hooks: Arc<Mutex<BTreeMap<u64, BeforeProviderPayloadHook>>>,
    pending_session_writes: Arc<Mutex<VecDeque<HarnessSessionWrite>>>,
    next_listener_id: AtomicU64,
    next_hook_id: AtomicU64,
    abort_flag: Arc<AtomicBool>,
    idle_notify: Notify,
    tool_execution: Mutex<ToolExecutionMode>,
    max_turns: Mutex<usize>,
}

impl AgentHarness {
    pub fn new(options: AgentHarnessOptions) -> Self {
        let active_tool_names = options.active_tool_names.unwrap_or_else(|| {
            options
                .tools
                .iter()
                .map(|tool| tool.definition.name.clone())
                .collect()
        });
        Self {
            env: options.env,
            session: options.session,
            model: Arc::new(Mutex::new(options.model)),
            thinking_level: Arc::new(Mutex::new(options.thinking_level)),
            system_prompt: Arc::new(Mutex::new(options.system_prompt)),
            system_prompt_provider: options.system_prompt_provider,
            stream_options: Arc::new(Mutex::new(options.stream_options)),
            get_api_key_and_headers: options.get_api_key_and_headers,
            resources: Arc::new(Mutex::new(options.resources)),
            tools: Arc::new(Mutex::new(options.tools)),
            active_tool_names: Arc::new(Mutex::new(active_tool_names)),
            steering_queue: Arc::new(HarnessMessageQueue::new(options.steering_mode)),
            follow_up_queue: Arc::new(HarnessMessageQueue::new(options.follow_up_mode)),
            next_turn_queue: Arc::new(HarnessMessageQueue::new(QueueMode::All)),
            phase: Mutex::new(AgentHarnessPhase::Idle),
            listeners: Arc::new(Mutex::new(BTreeMap::new())),
            before_agent_start_hooks: Mutex::new(BTreeMap::new()),
            context_hooks: Arc::new(Mutex::new(BTreeMap::new())),
            tool_call_hooks: Arc::new(Mutex::new(BTreeMap::new())),
            tool_result_hooks: Arc::new(Mutex::new(BTreeMap::new())),
            provider_request_hooks: Arc::new(Mutex::new(BTreeMap::new())),
            provider_payload_hooks: Arc::new(Mutex::new(BTreeMap::new())),
            pending_session_writes: Arc::new(Mutex::new(VecDeque::new())),
            next_listener_id: AtomicU64::new(1),
            next_hook_id: AtomicU64::new(1),
            abort_flag: Arc::new(AtomicBool::new(false)),
            idle_notify: Notify::new(),
            tool_execution: Mutex::new(options.tool_execution),
            max_turns: Mutex::new(options.max_turns),
        }
    }

    pub fn session(&self) -> Session {
        self.session.clone()
    }

    pub fn phase(&self) -> AgentHarnessPhase {
        *self.phase.lock()
    }

    pub fn get_model(&self) -> Model {
        self.model.lock().clone()
    }

    pub fn set_model(&self, model: Model) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::ModelChange {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
        })?;
        let previous_model = self.get_model();
        *self.model.lock() = model.clone();
        self.emit(AgentHarnessEvent::ModelSelect(ModelSelectEvent {
            model,
            previous_model,
        }));
        Ok(())
    }

    pub fn get_thinking_level(&self) -> ThinkingLevel {
        *self.thinking_level.lock()
    }

    pub fn set_thinking_level(
        &self,
        thinking_level: ThinkingLevel,
    ) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::ThinkingLevelChange {
            thinking_level: thinking_level_name(thinking_level).to_owned(),
        })?;
        let previous_level = self.get_thinking_level();
        *self.thinking_level.lock() = thinking_level;
        self.emit(AgentHarnessEvent::ThinkingLevelSelect(
            ThinkingLevelSelectEvent {
                level: thinking_level,
                previous_level,
            },
        ));
        Ok(())
    }

    pub fn get_stream_options(&self) -> SimpleStreamOptions {
        self.stream_options.lock().clone()
    }

    pub fn set_stream_options(&self, stream_options: SimpleStreamOptions) {
        *self.stream_options.lock() = stream_options;
    }

    pub fn append_message(&self, message: Message) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::Message { message })
    }

    pub fn append_custom_entry(
        &self,
        custom_type: impl Into<String>,
        data: Option<Value>,
    ) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::Custom {
            custom_type: custom_type.into(),
            data,
        })
    }

    pub fn append_custom_message(
        &self,
        custom_type: impl Into<String>,
        content: CustomMessageContent,
        display: bool,
        details: Option<Value>,
    ) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::CustomMessage {
            custom_type: custom_type.into(),
            content,
            display,
            details,
        })
    }

    pub fn append_label(
        &self,
        target_id: impl Into<String>,
        label: Option<String>,
    ) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::Label {
            target_id: target_id.into(),
            label,
        })
    }

    pub fn append_session_name(&self, name: impl Into<String>) -> Result<(), AgentHarnessError> {
        self.write_or_queue_session_write(HarnessSessionWrite::SessionName { name: name.into() })
    }

    pub fn get_resources(&self) -> AgentHarnessResources {
        self.resources.lock().clone()
    }

    pub fn set_resources(&self, resources: AgentHarnessResources) {
        let previous_resources = {
            let mut current = self.resources.lock();
            let previous = current.clone();
            *current = resources.clone();
            previous
        };
        self.emit(AgentHarnessEvent::ResourcesUpdate(ResourcesUpdateEvent {
            resources,
            previous_resources,
        }));
    }

    pub fn get_tools(&self) -> Vec<AgentTool> {
        self.tools.lock().clone()
    }

    pub fn get_active_tool_names(&self) -> Vec<String> {
        self.active_tool_names.lock().clone()
    }

    pub fn get_active_tools(&self) -> Vec<AgentTool> {
        let tools = self.tools.lock().clone();
        let active_tool_names = self.active_tool_names.lock().clone();
        active_tools_from(&tools, &active_tool_names)
    }

    pub fn set_active_tools(&self, tool_names: Vec<String>) -> Result<(), AgentHarnessError> {
        let tools = self.tools.lock();
        validate_tool_names(&tool_names, &tools)?;
        drop(tools);
        *self.active_tool_names.lock() = tool_names;
        Ok(())
    }

    pub fn set_tools(
        &self,
        tools: Vec<AgentTool>,
        active_tool_names: Option<Vec<String>>,
    ) -> Result<(), AgentHarnessError> {
        let next_active_tool_names =
            active_tool_names.unwrap_or_else(|| self.active_tool_names.lock().clone());
        validate_tool_names(&next_active_tool_names, &tools)?;
        *self.tools.lock() = tools;
        *self.active_tool_names.lock() = next_active_tool_names;
        Ok(())
    }

    pub fn get_steering_mode(&self) -> QueueMode {
        self.steering_queue.mode()
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.set_mode(mode);
    }

    pub fn get_follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.mode()
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.set_mode(mode);
    }

    pub fn subscribe(&self, listener: impl Fn(&AgentHarnessEvent) + Send + Sync + 'static) -> u64 {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners
            .lock()
            .insert(id, AgentHarnessListenerEntry::Sync(Arc::new(listener)));
        id
    }

    pub fn subscribe_async<F, Fut>(&self, listener: F) -> u64
    where
        F: Fn(AgentHarnessEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners.lock().insert(
            id,
            AgentHarnessListenerEntry::Async(Arc::new(move |event| Box::pin(listener(event)))),
        );
        id
    }

    pub fn unsubscribe(&self, id: u64) {
        self.listeners.lock().remove(&id);
    }

    pub fn on_before_agent_start(
        &self,
        hook: impl Fn(
            BeforeAgentStartEvent,
        ) -> Result<Option<BeforeAgentStartResult>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.before_agent_start_hooks
            .lock()
            .insert(id, Arc::new(hook));
        id
    }

    pub fn remove_before_agent_start_hook(&self, id: u64) {
        self.before_agent_start_hooks.lock().remove(&id);
    }

    pub fn on_context(
        &self,
        hook: impl Fn(ContextEvent) -> Result<Option<ContextResult>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.context_hooks.lock().insert(id, Arc::new(hook));
        id
    }

    pub fn remove_context_hook(&self, id: u64) {
        self.context_hooks.lock().remove(&id);
    }

    pub fn on_tool_call(
        &self,
        hook: impl Fn(ToolCallEvent) -> Result<Option<ToolCallResult>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.tool_call_hooks.lock().insert(id, Arc::new(hook));
        id
    }

    pub fn remove_tool_call_hook(&self, id: u64) {
        self.tool_call_hooks.lock().remove(&id);
    }

    pub fn on_tool_result(
        &self,
        hook: impl Fn(ToolResultEvent) -> Result<Option<ToolResultPatch>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.tool_result_hooks.lock().insert(id, Arc::new(hook));
        id
    }

    pub fn remove_tool_result_hook(&self, id: u64) {
        self.tool_result_hooks.lock().remove(&id);
    }

    pub fn on_before_provider_request(
        &self,
        hook: impl Fn(
            BeforeProviderRequestEvent,
        ) -> Result<Option<BeforeProviderRequestResult>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.provider_request_hooks
            .lock()
            .insert(id, Arc::new(hook));
        id
    }

    pub fn remove_before_provider_request_hook(&self, id: u64) {
        self.provider_request_hooks.lock().remove(&id);
    }

    pub fn on_before_provider_payload(
        &self,
        hook: impl Fn(
            BeforeProviderPayloadEvent,
        ) -> Result<Option<BeforeProviderPayloadResult>, AgentHarnessError>
        + Send
        + Sync
        + 'static,
    ) -> u64 {
        let id = self.next_hook_id.fetch_add(1, Ordering::SeqCst);
        self.provider_payload_hooks
            .lock()
            .insert(id, Arc::new(hook));
        id
    }

    pub fn remove_before_provider_payload_hook(&self, id: u64) {
        self.provider_payload_hooks.lock().remove(&id);
    }

    pub fn steer(&self, text: impl Into<String>) -> Result<(), AgentHarnessError> {
        self.steer_message(user_message(text))
    }

    pub fn steer_message(&self, message: Message) -> Result<(), AgentHarnessError> {
        self.require_running("Cannot steer while idle")?;
        self.steering_queue.enqueue(message);
        self.emit_queue_update();
        Ok(())
    }

    pub fn follow_up(&self, text: impl Into<String>) -> Result<(), AgentHarnessError> {
        self.follow_up_message(user_message(text))
    }

    pub fn follow_up_message(&self, message: Message) -> Result<(), AgentHarnessError> {
        self.require_running("Cannot follow up while idle")?;
        self.follow_up_queue.enqueue(message);
        self.emit_queue_update();
        Ok(())
    }

    pub fn next_turn(&self, text: impl Into<String>) {
        self.next_turn_message(user_message(text));
    }

    pub fn next_turn_message(&self, message: Message) {
        self.next_turn_queue.enqueue(message);
        self.emit_queue_update();
    }

    pub fn abort(&self) -> AbortResult {
        self.abort_flag.store(true, Ordering::SeqCst);
        let result = AbortResult {
            cleared_steer: self.steering_queue.drain_all(),
            cleared_follow_up: self.follow_up_queue.drain_all(),
        };
        self.emit_queue_update();
        self.emit(AgentHarnessEvent::Abort(result.clone()));
        result
    }

    pub async fn wait_for_idle(&self) {
        loop {
            if self.phase() == AgentHarnessPhase::Idle {
                return;
            }
            self.idle_notify.notified().await;
        }
    }

    pub async fn prompt(
        &self,
        text: impl Into<String>,
    ) -> Result<AssistantMessage, AgentHarnessError> {
        self.prompt_with_images(text, Vec::new()).await
    }

    pub async fn prompt_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> Result<AssistantMessage, AgentHarnessError> {
        self.start_turn()?;
        let result = self.prompt_inner(text.into(), images).await;
        self.finish_turn();
        result
    }

    pub async fn skill(
        &self,
        name: impl AsRef<str>,
        additional_instructions: Option<&str>,
    ) -> Result<AssistantMessage, AgentHarnessError> {
        self.start_turn()?;
        let result = async {
            let skill = {
                let resources = self.resources.lock();
                resources
                    .skills
                    .iter()
                    .find(|skill| skill.name == name.as_ref())
                    .cloned()
                    .ok_or_else(|| {
                        AgentHarnessError::new(
                            AgentHarnessErrorCode::InvalidArgument,
                            format!("Unknown skill: {}", name.as_ref()),
                        )
                    })?
            };
            self.prompt_inner(
                format_skill_invocation(&skill, additional_instructions),
                Vec::new(),
            )
            .await
        }
        .await;
        self.finish_turn();
        result
    }

    pub async fn prompt_from_template(
        &self,
        name: impl AsRef<str>,
        args: &[String],
    ) -> Result<AssistantMessage, AgentHarnessError> {
        self.start_turn()?;
        let result = async {
            let template = {
                let resources = self.resources.lock();
                resources
                    .prompt_templates
                    .iter()
                    .find(|template| template.name == name.as_ref())
                    .cloned()
                    .ok_or_else(|| {
                        AgentHarnessError::new(
                            AgentHarnessErrorCode::InvalidArgument,
                            format!("Unknown prompt template: {}", name.as_ref()),
                        )
                    })?
            };
            self.prompt_inner(
                format_prompt_template_invocation(&template, args),
                Vec::new(),
            )
            .await
        }
        .await;
        self.finish_turn();
        result
    }

    async fn prompt_inner(
        &self,
        text: String,
        images: Vec<ImageContent>,
    ) -> Result<AssistantMessage, AgentHarnessError> {
        self.abort_flag.store(false, Ordering::SeqCst);
        let session_context = self
            .session
            .build_context()
            .map_err(AgentHarnessError::session)?;
        let context_messages = session_context
            .messages
            .into_iter()
            .filter_map(llm_session_message)
            .map(AgentMessage::from)
            .collect();
        let mut prompt_messages = self.next_turn_queue.drain_all();
        if !prompt_messages.is_empty() {
            self.emit_queue_update_async().await;
        }
        let resources = self.get_resources();
        let system_prompt = self.resolve_system_prompt()?;
        let before_result = self.emit_before_agent_start(BeforeAgentStartEvent {
            prompt: text.clone(),
            images: images.clone(),
            system_prompt: system_prompt.clone(),
            resources,
        })?;
        prompt_messages.push(user_message_with_images(text, images).into());
        if let Some(messages) = before_result
            .as_ref()
            .and_then(|result| result.messages.clone())
        {
            prompt_messages.extend(messages);
        }

        let model = self.get_model();
        let thinking_level = self.get_thinking_level();
        let stream_options = resolve_provider_stream_options(
            &model,
            thinking_level,
            &self.session,
            self.stream_options.lock().clone(),
            self.get_api_key_and_headers.as_ref(),
            &self.provider_request_hooks,
            &self.provider_payload_hooks,
            self.abort_flag.clone(),
        )?;
        let config = AgentLoopConfig {
            model: model.clone(),
            stream_options,
            tool_call_hooks: vec![Arc::new(HarnessToolCallHook {
                hooks: self.tool_call_hooks.clone(),
            })],
            tool_result_hooks: vec![Arc::new(HarnessToolResultHook {
                hooks: self.tool_result_hooks.clone(),
            })],
            transform_context: Some(Arc::new(HarnessContextTransformer {
                hooks: self.context_hooks.clone(),
            })),
            convert_to_llm: None,
            prepare_next_turn: Some(Arc::new(HarnessNextTurnPreparer {
                env: self.env.clone(),
                session: self.session.clone(),
                model: self.model.clone(),
                thinking_level: self.thinking_level.clone(),
                system_prompt: self.system_prompt.clone(),
                system_prompt_provider: self.system_prompt_provider.clone(),
                stream_options: self.stream_options.clone(),
                get_api_key_and_headers: self.get_api_key_and_headers.clone(),
                provider_request_hooks: self.provider_request_hooks.clone(),
                provider_payload_hooks: self.provider_payload_hooks.clone(),
                abort_flag: self.abort_flag.clone(),
                resources: self.resources.clone(),
                tools: self.tools.clone(),
                active_tool_names: self.active_tool_names.clone(),
            })),
            should_stop_after_turn: None,
            queued_message_provider: Some(Arc::new(HarnessQueuedMessageProvider {
                queue: self.steering_queue.clone(),
                emit_queue_update: self.queue_update_emitter_async(),
            })),
            follow_up_message_provider: Some(Arc::new(HarnessQueuedMessageProvider {
                queue: self.follow_up_queue.clone(),
                emit_queue_update: self.queue_update_emitter_async(),
            })),
            event_sink: Some(Arc::new(HarnessEventSink {
                listeners: self.listeners.clone(),
            })),
            skip_initial_queued_message_poll: false,
            tool_execution: *self.tool_execution.lock(),
            max_turns: *self.max_turns.lock(),
        };
        let context = AgentContext {
            system_prompt: before_result
                .and_then(|result| result.system_prompt)
                .unwrap_or(system_prompt),
            messages: context_messages,
            tools: self.get_active_tools(),
        };
        let (messages, events) = agent_loop_prompt_messages(context, prompt_messages, config)
            .await
            .map_err(AgentHarnessError::unknown)?;
        let mut session = self.session.clone();
        for message in &messages {
            if let Some(message) = message.to_llm_message() {
                session
                    .append_message(message)
                    .map_err(AgentHarnessError::session)?;
            }
        }
        let had_pending_mutations = !self.pending_session_writes.lock().is_empty();
        self.flush_pending_session_writes()?;
        self.emit_async(AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        })
        .await;
        drop(events);
        self.flush_pending_session_writes()?;
        let assistant = messages
            .iter()
            .rev()
            .find_map(|message| match message {
                AgentMessage::Assistant(assistant) => Some(assistant.clone()),
                _ => None,
            })
            .ok_or_else(|| {
                AgentHarnessError::new(
                    AgentHarnessErrorCode::InvalidState,
                    "AgentHarness prompt completed without an assistant message",
                )
            })?;
        self.emit_async(AgentHarnessEvent::Settled {
            next_turn_count: self.next_turn_queue.snapshot().len(),
        })
        .await;
        Ok(assistant)
    }

    fn start_turn(&self) -> Result<(), AgentHarnessError> {
        let mut phase = self.phase.lock();
        if *phase != AgentHarnessPhase::Idle {
            return Err(AgentHarnessError::new(
                AgentHarnessErrorCode::Busy,
                "AgentHarness is busy",
            ));
        }
        *phase = AgentHarnessPhase::Turn;
        Ok(())
    }

    fn finish_turn(&self) {
        *self.phase.lock() = AgentHarnessPhase::Idle;
        self.idle_notify.notify_waiters();
    }

    fn require_running(&self, message: &'static str) -> Result<(), AgentHarnessError> {
        if self.phase() == AgentHarnessPhase::Idle {
            return Err(AgentHarnessError::new(
                AgentHarnessErrorCode::InvalidState,
                message,
            ));
        }
        Ok(())
    }

    fn emit_before_agent_start(
        &self,
        event: BeforeAgentStartEvent,
    ) -> Result<Option<BeforeAgentStartResult>, AgentHarnessError> {
        let hooks: Vec<BeforeAgentStartHook> = self
            .before_agent_start_hooks
            .lock()
            .values()
            .cloned()
            .collect();
        let mut last_result = None;
        for hook in hooks {
            if let Some(result) = hook(event.clone())? {
                last_result = Some(result);
            }
        }
        Ok(last_result)
    }

    fn queue_update_emitter_async(&self) -> Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync> {
        let listeners = self.listeners.clone();
        let steering_queue = self.steering_queue.clone();
        let follow_up_queue = self.follow_up_queue.clone();
        let next_turn_queue = self.next_turn_queue.clone();
        Arc::new(move || {
            let listeners = listeners.clone();
            let event = AgentHarnessEvent::QueueUpdate(QueueUpdateEvent {
                steer: steering_queue.snapshot(),
                follow_up: follow_up_queue.snapshot(),
                next_turn: next_turn_queue.snapshot(),
            });
            Box::pin(async move {
                emit_to_async(&listeners, event).await;
            })
        })
    }

    fn emit_queue_update(&self) {
        self.emit(AgentHarnessEvent::QueueUpdate(QueueUpdateEvent {
            steer: self.steering_queue.snapshot(),
            follow_up: self.follow_up_queue.snapshot(),
            next_turn: self.next_turn_queue.snapshot(),
        }));
    }

    async fn emit_queue_update_async(&self) {
        self.emit_async(AgentHarnessEvent::QueueUpdate(QueueUpdateEvent {
            steer: self.steering_queue.snapshot(),
            follow_up: self.follow_up_queue.snapshot(),
            next_turn: self.next_turn_queue.snapshot(),
        }))
        .await;
    }

    fn write_or_queue_session_write(
        &self,
        write: HarnessSessionWrite,
    ) -> Result<(), AgentHarnessError> {
        if self.phase() == AgentHarnessPhase::Idle {
            self.apply_session_write(write)
        } else {
            self.pending_session_writes.lock().push_back(write);
            Ok(())
        }
    }

    fn flush_pending_session_writes(&self) -> Result<(), AgentHarnessError> {
        flush_pending_session_writes_for(&self.session, &self.pending_session_writes)
    }

    fn apply_session_write(&self, write: HarnessSessionWrite) -> Result<(), AgentHarnessError> {
        apply_session_write_for(&self.session, write)
    }

    fn resolve_system_prompt(&self) -> Result<String, AgentHarnessError> {
        resolve_system_prompt_from_parts(
            self.system_prompt.lock().clone(),
            self.system_prompt_provider.as_ref(),
            SystemPromptContext {
                env: self.env.clone(),
                session: self.session.clone(),
                model: self.get_model(),
                thinking_level: self.get_thinking_level(),
                active_tools: self.get_active_tools(),
                resources: self.get_resources(),
            },
        )
    }

    fn emit(&self, event: AgentHarnessEvent) {
        emit_to(&self.listeners, &event);
    }

    async fn emit_async(&self, event: AgentHarnessEvent) {
        emit_to_async(&self.listeners, event).await;
    }
}

fn resolve_system_prompt_from_parts(
    fallback: String,
    provider: Option<&SystemPromptProvider>,
    context: SystemPromptContext,
) -> Result<String, AgentHarnessError> {
    match provider {
        Some(provider) => provider(context),
        None => Ok(fallback),
    }
}

struct HarnessProviderPayloadHook {
    hooks: Arc<Mutex<BTreeMap<u64, BeforeProviderPayloadHook>>>,
}

impl ProviderPayloadHook for HarnessProviderPayloadHook {
    fn on_payload(&self, model: &Model, payload: Value) -> Result<Value, String> {
        let hooks: Vec<BeforeProviderPayloadHook> = self.hooks.lock().values().cloned().collect();
        let mut current = payload;
        for hook in hooks {
            if let Some(result) = hook(BeforeProviderPayloadEvent {
                model: model.clone(),
                payload: current.clone(),
            })
            .map_err(|error| error.to_string())?
            {
                current = result.payload;
            }
        }
        Ok(current)
    }
}

fn resolve_provider_stream_options(
    model: &Model,
    thinking_level: ThinkingLevel,
    session: &Session,
    base_stream_options: SimpleStreamOptions,
    get_api_key_and_headers: Option<&ProviderAuthProvider>,
    provider_request_hooks: &Arc<Mutex<BTreeMap<u64, BeforeProviderRequestHook>>>,
    provider_payload_hooks: &Arc<Mutex<BTreeMap<u64, BeforeProviderPayloadHook>>>,
    abort_flag: Arc<AtomicBool>,
) -> Result<SimpleStreamOptions, AgentHarnessError> {
    let mut options = base_stream_options;
    options.reasoning = (thinking_level != ThinkingLevel::Off).then_some(thinking_level);
    if options.stream.session_id.is_none() {
        options.stream.session_id = Some(session.metadata_id());
    }
    if let Some(provider) = get_api_key_and_headers {
        let auth = provider(model)?;
        options.stream.api_key = auth.api_key;
        options.stream.headers.extend(auth.headers);
    }

    let hooks: Vec<BeforeProviderRequestHook> =
        provider_request_hooks.lock().values().cloned().collect();
    for hook in hooks {
        if let Some(result) = hook(BeforeProviderRequestEvent {
            model: model.clone(),
            session_id: options.stream.session_id.clone().unwrap_or_default(),
            stream_options: options.clone(),
        })? && let Some(patch) = result.stream_options
        {
            apply_stream_options_patch(&mut options, patch);
        }
    }

    if !provider_payload_hooks.lock().is_empty() {
        options
            .payload_hooks
            .push(Arc::new(HarnessProviderPayloadHook {
                hooks: provider_payload_hooks.clone(),
            }));
    }
    options.stream.abort_flag = Some(abort_flag);
    Ok(options)
}

fn apply_stream_options_patch(
    options: &mut SimpleStreamOptions,
    patch: AgentHarnessStreamOptionsPatch,
) {
    if let Some(patch) = patch.transport {
        apply_option_patch(&mut options.stream.transport, patch);
    }
    if let Some(patch) = patch.timeout_ms {
        apply_option_patch(&mut options.stream.timeout_ms, patch);
    }
    if let Some(patch) = patch.max_retries {
        apply_option_patch(&mut options.stream.max_retries, patch);
    }
    if let Some(patch) = patch.max_retry_delay_ms {
        apply_option_patch(&mut options.stream.max_retry_delay_ms, patch);
    }
    if let Some(patch) = patch.cache_retention {
        apply_option_patch(&mut options.stream.cache_retention, patch);
    }
    if let Some(patch) = patch.headers {
        apply_header_patch(&mut options.stream.headers, patch);
    }
    if let Some(patch) = patch.metadata {
        apply_metadata_patch(&mut options.stream.metadata, patch);
    }
}

fn apply_option_patch<T>(target: &mut Option<T>, patch: OptionPatch<T>) {
    match patch {
        OptionPatch::Set(value) => *target = Some(value),
        OptionPatch::Clear => *target = None,
    }
}

fn apply_header_patch(target: &mut BTreeMap<String, String>, patch: HeaderMapPatch) {
    match patch {
        HeaderMapPatch::Replace(headers) => *target = headers,
        HeaderMapPatch::Clear => target.clear(),
        HeaderMapPatch::Merge(headers) => {
            for (key, value) in headers {
                if let Some(value) = value {
                    target.insert(key, value);
                } else {
                    target.remove(&key);
                }
            }
        }
    }
}

fn apply_metadata_patch(target: &mut Map<String, Value>, patch: MetadataMapPatch) {
    match patch {
        MetadataMapPatch::Replace(metadata) => *target = metadata,
        MetadataMapPatch::Clear => target.clear(),
        MetadataMapPatch::Merge(metadata) => {
            for (key, value) in metadata {
                if let Some(value) = value {
                    target.insert(key, value);
                } else {
                    target.remove(&key);
                }
            }
        }
    }
}

fn flush_pending_session_writes_for(
    session: &Session,
    pending_session_writes: &Mutex<VecDeque<HarnessSessionWrite>>,
) -> Result<(), AgentHarnessError> {
    while let Some(write) = pending_session_writes.lock().pop_front() {
        if let Err(error) = apply_session_write_for(session, write.clone()) {
            pending_session_writes.lock().push_front(write);
            return Err(error);
        }
    }
    Ok(())
}

fn apply_session_write_for(
    session: &Session,
    write: HarnessSessionWrite,
) -> Result<(), AgentHarnessError> {
    let mut session = session.clone();
    match write {
        HarnessSessionWrite::Message { message } => session
            .append_message(message)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::ModelChange { provider, model_id } => session
            .append_model_change(provider, model_id)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::ThinkingLevelChange { thinking_level } => session
            .append_thinking_level_change(thinking_level)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::Custom { custom_type, data } => session
            .append_custom_entry(custom_type, data)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::CustomMessage {
            custom_type,
            content,
            display,
            details,
        } => session
            .append_custom_message_entry(custom_type, content, display, details)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::Label { target_id, label } => session
            .append_label(target_id, label)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
        HarnessSessionWrite::SessionName { name } => session
            .append_session_name(name)
            .map(|_| ())
            .map_err(AgentHarnessError::session),
    }
}

fn active_tools_from(tools: &[AgentTool], active_tool_names: &[String]) -> Vec<AgentTool> {
    active_tool_names
        .iter()
        .filter_map(|name| {
            tools
                .iter()
                .find(|tool| tool.definition.name == *name)
                .cloned()
        })
        .collect()
}

fn thinking_level_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
}

fn validate_tool_names(
    tool_names: &[String],
    tools: &[AgentTool],
) -> Result<(), AgentHarnessError> {
    let missing = tool_names
        .iter()
        .filter(|name| {
            !tools
                .iter()
                .any(|tool| tool.definition.name == name.as_str())
        })
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(AgentHarnessError::new(
            AgentHarnessErrorCode::InvalidArgument,
            format!("Unknown tool(s): {}", missing.join(", ")),
        ));
    }
    Ok(())
}

struct HarnessToolCallHook {
    hooks: Arc<Mutex<BTreeMap<u64, ToolCallHook>>>,
}

#[async_trait]
impl AgentToolCallHook for HarnessToolCallHook {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<Value>, String> {
        let hooks: Vec<ToolCallHook> = self.hooks.lock().values().cloned().collect();
        if hooks.is_empty() {
            return Ok(None);
        }
        let mut current_input = context.input;
        let mut replacement = None;
        for hook in hooks {
            if let Some(result) = hook(ToolCallEvent {
                tool_call_id: context.tool_call_id.clone(),
                tool_name: context.tool_name.clone(),
                input: current_input.clone(),
            })
            .map_err(|error| error.to_string())?
            {
                if let Some(input) = result.input {
                    current_input = input;
                    replacement = Some(current_input.clone());
                }
            }
        }
        Ok(replacement)
    }
}

struct HarnessToolResultHook {
    hooks: Arc<Mutex<BTreeMap<u64, ToolResultHook>>>,
}

#[async_trait]
impl AgentToolResultHook for HarnessToolResultHook {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResult>, String> {
        let hooks: Vec<ToolResultHook> = self.hooks.lock().values().cloned().collect();
        if hooks.is_empty() {
            return Ok(None);
        }
        let mut result = context.result;
        let mut patched = false;
        for hook in hooks {
            if let Some(patch) = hook(ToolResultEvent {
                tool_call_id: context.tool_call_id.clone(),
                tool_name: context.tool_name.clone(),
                input: context.input.clone(),
                content: result.content.clone(),
                details: result.details.clone(),
                terminate: result.terminate,
                is_error: false,
            })
            .map_err(|error| error.to_string())?
            {
                if let Some(content) = patch.content {
                    result.content = content;
                    patched = true;
                }
                if let Some(details) = patch.details {
                    result.details = Some(details);
                    patched = true;
                }
                if let Some(terminate) = patch.terminate {
                    result.terminate = terminate;
                    patched = true;
                }
            }
        }
        Ok(patched.then_some(result))
    }
}

struct HarnessContextTransformer {
    hooks: Arc<Mutex<BTreeMap<u64, ContextHook>>>,
}

#[async_trait]
impl crate::types::AgentContextTransformer for HarnessContextTransformer {
    async fn transform_context(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, String> {
        let hooks: Vec<ContextHook> = self.hooks.lock().values().cloned().collect();
        if hooks.is_empty() {
            return Ok(messages);
        }
        let mut current = messages;
        for hook in hooks {
            if let Some(result) = hook(ContextEvent {
                messages: current.clone(),
            })
            .map_err(|error| error.to_string())?
            {
                current = result.messages;
            }
        }
        Ok(current)
    }
}

fn emit_to(listeners: &Mutex<BTreeMap<u64, AgentHarnessListenerEntry>>, event: &AgentHarnessEvent) {
    let listeners: Vec<AgentHarnessListenerEntry> = listeners.lock().values().cloned().collect();
    for listener in listeners {
        match listener {
            AgentHarnessListenerEntry::Sync(listener) => listener(event),
            AgentHarnessListenerEntry::Async(listener) => {
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    std::mem::drop(handle.spawn(listener(event.clone())));
                }
            }
        }
    }
}

async fn emit_to_async(
    listeners: &Mutex<BTreeMap<u64, AgentHarnessListenerEntry>>,
    event: AgentHarnessEvent,
) {
    let listeners: Vec<AgentHarnessListenerEntry> = listeners.lock().values().cloned().collect();
    for listener in listeners {
        match listener {
            AgentHarnessListenerEntry::Sync(listener) => listener(&event),
            AgentHarnessListenerEntry::Async(listener) => listener(event.clone()).await,
        }
    }
}

fn user_message(text: impl Into<String>) -> Message {
    user_message_with_images(text.into(), Vec::new())
}

fn user_message_with_images(text: String, images: Vec<ImageContent>) -> Message {
    if images.is_empty() {
        return Message::User(UserMessage::text(text));
    }
    let mut content = vec![UserContent::Text(ri_llm_provider::TextContent::new(text))];
    content.extend(images.into_iter().map(UserContent::Image));
    Message::User(UserMessage {
        content: UserContentValue::Blocks(content),
        timestamp: now_millis(),
    })
}

fn llm_session_message(message: SessionMessage) -> Option<Message> {
    match message {
        SessionMessage::Llm { message } => Some(message),
        _ => None,
    }
}
