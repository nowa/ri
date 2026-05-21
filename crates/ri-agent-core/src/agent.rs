use crate::{
    agent_loop::{
        AgentLoopConfig, agent_loop_continue, agent_loop_prompt, agent_loop_prompt_messages,
    },
    types::{
        AgentApiKeyProvider, AgentContext, AgentContextTransformer, AgentEvent, AgentEventSink,
        AgentMessage, AgentMessageConverter, AgentNextTurnPreparer, AgentQueuedMessageProvider,
        AgentShouldStopAfterTurn, AgentState, AgentStreamProvider, AgentTool, AgentToolCallHook,
        AgentToolResultHook, QueueMode, ToolExecutionMode,
    },
};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::{Mutex, MutexGuard};
use ri_llm_provider::{
    ImageContent, Message, Model, SimpleStreamOptions, TextContent, ThinkingLevel, UserContent,
    UserContentValue, UserMessage, now_millis,
};
use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use tokio::sync::Notify;

type SyncListener = Arc<dyn Fn(&AgentEvent, Arc<AtomicBool>) + Send + Sync>;
type AsyncListener =
    Arc<dyn Fn(AgentEvent, Arc<AtomicBool>) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone)]
enum Listener {
    Sync(SyncListener),
    Async(AsyncListener),
}

#[derive(Debug)]
struct PendingMessageQueue {
    inner: Mutex<PendingMessageQueueInner>,
}

#[derive(Debug)]
struct PendingMessageQueueInner {
    mode: QueueMode,
    messages: VecDeque<AgentMessage>,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            inner: Mutex::new(PendingMessageQueueInner {
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

    fn has_items(&self) -> bool {
        !self.inner.lock().messages.is_empty()
    }

    fn clear(&self) {
        self.inner.lock().messages.clear();
    }
}

#[async_trait]
impl AgentQueuedMessageProvider for PendingMessageQueue {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        Ok(self.drain_now())
    }
}

struct QueuedMessageProviderChain {
    queue: Arc<PendingMessageQueue>,
    extra: Option<Arc<dyn AgentQueuedMessageProvider>>,
}

#[async_trait]
impl AgentQueuedMessageProvider for QueuedMessageProviderChain {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String> {
        let mut messages = self.queue.drain_now();
        if let Some(extra) = &self.extra {
            messages.extend(extra.get_queued_messages().await?);
        }
        Ok(messages)
    }
}

struct AgentRuntimeEventSink {
    state: Arc<Mutex<AgentState>>,
    listeners: Arc<Mutex<BTreeMap<u64, Listener>>>,
    abort_flag: Arc<AtomicBool>,
}

#[async_trait]
impl AgentEventSink for AgentRuntimeEventSink {
    async fn on_event(&self, event: &AgentEvent) {
        {
            let mut state = self.state.lock();
            reduce_agent_state(&mut state, event);
        }

        let listeners: Vec<Listener> = self.listeners.lock().values().cloned().collect();
        for listener in listeners {
            match listener {
                Listener::Sync(listener) => listener(event, self.abort_flag.clone()),
                Listener::Async(listener) => listener(event.clone(), self.abort_flag.clone()).await,
            }
        }
    }
}

fn reduce_agent_state(state: &mut AgentState, event: &AgentEvent) {
    match event {
        AgentEvent::AgentStart | AgentEvent::TurnStart => {}
        AgentEvent::AgentEnd { .. } => {
            state.streaming_message = None;
        }
        AgentEvent::TurnEnd { message, .. } => {
            if let AgentMessage::Assistant(assistant) = message
                && let Some(error) = &assistant.error_message
            {
                state.error_message = Some(error.clone());
            }
        }
        AgentEvent::MessageStart { message } | AgentEvent::MessageUpdate { message, .. } => {
            state.streaming_message = Some(message.clone());
        }
        AgentEvent::MessageEnd { message } => {
            state.streaming_message = None;
            state.messages.push(message.clone());
        }
        AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
            state.pending_tool_calls.insert(tool_call_id.clone());
        }
        AgentEvent::ToolExecutionUpdate { .. } => {}
        AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
            state.pending_tool_calls.remove(tool_call_id);
        }
    }
}

#[derive(Clone)]
pub struct AgentOptions {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
    pub stream_options: SimpleStreamOptions,
    pub tool_call_hooks: Vec<Arc<dyn AgentToolCallHook>>,
    pub tool_result_hooks: Vec<Arc<dyn AgentToolResultHook>>,
    pub transform_context: Option<Arc<dyn AgentContextTransformer>>,
    pub convert_to_llm: Option<Arc<dyn AgentMessageConverter>>,
    pub stream_provider: Option<Arc<dyn AgentStreamProvider>>,
    pub api_key_provider: Option<Arc<dyn AgentApiKeyProvider>>,
    pub prepare_next_turn: Option<Arc<dyn AgentNextTurnPreparer>>,
    pub should_stop_after_turn: Option<Arc<dyn AgentShouldStopAfterTurn>>,
    pub queued_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    pub follow_up_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    pub tool_execution: ToolExecutionMode,
    pub max_turns: usize,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
}

impl AgentOptions {
    pub fn new(model: Model) -> Self {
        Self {
            system_prompt: String::new(),
            model,
            thinking_level: ThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            stream_options: SimpleStreamOptions::default(),
            tool_call_hooks: Vec::new(),
            tool_result_hooks: Vec::new(),
            transform_context: None,
            convert_to_llm: None,
            stream_provider: None,
            api_key_provider: None,
            prepare_next_turn: None,
            should_stop_after_turn: None,
            queued_message_provider: None,
            follow_up_message_provider: None,
            tool_execution: ToolExecutionMode::Parallel,
            max_turns: 16,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
        }
    }
}

pub struct Agent {
    state: Arc<Mutex<AgentState>>,
    stream_options: SimpleStreamOptions,
    tool_call_hooks: Vec<Arc<dyn AgentToolCallHook>>,
    tool_result_hooks: Vec<Arc<dyn AgentToolResultHook>>,
    transform_context: Option<Arc<dyn AgentContextTransformer>>,
    convert_to_llm: Option<Arc<dyn AgentMessageConverter>>,
    stream_provider: Option<Arc<dyn AgentStreamProvider>>,
    api_key_provider: Option<Arc<dyn AgentApiKeyProvider>>,
    prepare_next_turn: Option<Arc<dyn AgentNextTurnPreparer>>,
    should_stop_after_turn: Option<Arc<dyn AgentShouldStopAfterTurn>>,
    queued_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    follow_up_message_provider: Option<Arc<dyn AgentQueuedMessageProvider>>,
    steering_queue: Arc<PendingMessageQueue>,
    follow_up_queue: Arc<PendingMessageQueue>,
    tool_execution: ToolExecutionMode,
    max_turns: usize,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    listeners: Arc<Mutex<BTreeMap<u64, Listener>>>,
    next_listener_id: AtomicU64,
    abort_flag: Arc<AtomicBool>,
    idle_notify: Notify,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let mut state = AgentState::new(options.model);
        state.system_prompt = options.system_prompt;
        state.thinking_level = options.thinking_level;
        state.tools = options.tools;
        state.messages = options.messages;
        let steering_queue = Arc::new(PendingMessageQueue::new(options.steering_mode));
        let follow_up_queue = Arc::new(PendingMessageQueue::new(options.follow_up_mode));
        Self {
            state: Arc::new(Mutex::new(state)),
            stream_options: options.stream_options,
            tool_call_hooks: options.tool_call_hooks,
            tool_result_hooks: options.tool_result_hooks,
            transform_context: options.transform_context,
            convert_to_llm: options.convert_to_llm,
            stream_provider: options.stream_provider,
            api_key_provider: options.api_key_provider,
            prepare_next_turn: options.prepare_next_turn,
            should_stop_after_turn: options.should_stop_after_turn,
            queued_message_provider: options.queued_message_provider,
            follow_up_message_provider: options.follow_up_message_provider,
            steering_queue,
            follow_up_queue,
            tool_execution: options.tool_execution,
            max_turns: options.max_turns,
            steering_mode: options.steering_mode,
            follow_up_mode: options.follow_up_mode,
            listeners: Arc::new(Mutex::new(BTreeMap::new())),
            next_listener_id: AtomicU64::new(1),
            abort_flag: Arc::new(AtomicBool::new(false)),
            idle_notify: Notify::new(),
        }
    }

    pub fn state(&self) -> MutexGuard<'_, AgentState> {
        self.state.lock()
    }

    pub fn state_mut(&self) -> MutexGuard<'_, AgentState> {
        self.state.lock()
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.mode()
    }

    pub fn set_steering_mode(&mut self, mode: QueueMode) {
        self.steering_mode = mode;
        self.steering_queue.set_mode(mode);
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.mode()
    }

    pub fn set_follow_up_mode(&mut self, mode: QueueMode) {
        self.follow_up_mode = mode;
        self.follow_up_queue.set_mode(mode);
    }

    pub fn session_id(&self) -> Option<&str> {
        self.stream_options.stream.session_id.as_deref()
    }

    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.stream_options.stream.session_id = session_id;
    }

    pub fn steer(&self, message: impl Into<AgentMessage>) {
        self.steering_queue.enqueue(message);
    }

    pub fn follow_up(&self, message: impl Into<AgentMessage>) {
        self.follow_up_queue.enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.has_items() || self.follow_up_queue.has_items()
    }

    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.messages.clear();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.error_message = None;
        drop(state);
        self.abort_flag.store(false, Ordering::SeqCst);
        self.clear_all_queues();
        self.idle_notify.notify_waiters();
    }

    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::SeqCst);
        self.clear_all_queues();
    }

    pub fn abort_handle(&self) -> Arc<AtomicBool> {
        self.abort_flag.clone()
    }

    pub fn subscribe(&self, listener: impl Fn(&AgentEvent) + Send + Sync + 'static) -> u64 {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners.lock().insert(
            id,
            Listener::Sync(Arc::new(move |event, _abort_flag| listener(event))),
        );
        id
    }

    pub fn subscribe_with_abort_flag(
        &self,
        listener: impl Fn(&AgentEvent, Arc<AtomicBool>) + Send + Sync + 'static,
    ) -> u64 {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners
            .lock()
            .insert(id, Listener::Sync(Arc::new(listener)));
        id
    }

    pub fn subscribe_async<F, Fut>(&self, listener: F) -> u64
    where
        F: Fn(AgentEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners.lock().insert(
            id,
            Listener::Async(Arc::new(move |event, _abort_flag| {
                Box::pin(listener(event))
            })),
        );
        id
    }

    pub fn subscribe_async_with_abort_flag<F, Fut>(&self, listener: F) -> u64
    where
        F: Fn(AgentEvent, Arc<AtomicBool>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.listeners.lock().insert(
            id,
            Listener::Async(Arc::new(move |event, abort_flag| {
                Box::pin(listener(event, abort_flag))
            })),
        );
        id
    }

    pub fn unsubscribe(&self, id: u64) {
        self.listeners.lock().remove(&id);
    }

    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.idle_notify.notified();
            if !self.state.lock().is_streaming {
                return;
            }
            notified.await;
        }
    }

    pub async fn prompt(&self, prompt: impl Into<String>) -> Result<Vec<AgentMessage>, String> {
        if self.state.lock().is_streaming {
            return Err(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
                    .to_owned(),
            );
        }
        self.run_with_skip(
            |context, config| agent_loop_prompt(context, prompt.into(), config),
            false,
        )
        .await
    }

    pub async fn prompt_with_images(
        &self,
        prompt: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> Result<Vec<AgentMessage>, String> {
        if images.is_empty() {
            return self.prompt(prompt).await;
        }

        let mut content = vec![UserContent::Text(TextContent::new(prompt.into()))];
        content.extend(images.into_iter().map(UserContent::Image));
        self.prompt_messages(vec![Message::User(UserMessage {
            content: UserContentValue::Blocks(content),
            timestamp: now_millis(),
        })])
        .await
    }

    pub async fn continue_run(&self) -> Result<Vec<AgentMessage>, String> {
        if self.state.lock().is_streaming {
            return Err(
                "Agent is already processing. Wait for completion before continuing.".to_owned(),
            );
        }
        if self.state.lock().messages.is_empty() {
            return Err("No messages to continue from".to_owned());
        }
        let assistant_tail = {
            let state = self.state.lock();
            matches!(state.messages.last(), Some(AgentMessage::Assistant(_)))
        };
        if assistant_tail {
            let steering_messages = self.steering_queue.drain_now();
            if !steering_messages.is_empty() {
                return self
                    .prompt_messages_with_options(steering_messages, true)
                    .await;
            }
            let follow_up_messages = self.follow_up_queue.drain_now();
            if !follow_up_messages.is_empty() {
                return self.prompt_messages(follow_up_messages).await;
            }
            return Err("Cannot continue from message role: assistant".to_owned());
        }
        self.run_with_skip(agent_loop_continue, false).await
    }

    pub async fn prompt_messages<M>(&self, messages: Vec<M>) -> Result<Vec<AgentMessage>, String>
    where
        M: Into<AgentMessage>,
    {
        if self.state.lock().is_streaming {
            return Err(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
                    .to_owned(),
            );
        }
        self.prompt_messages_with_options(messages, false).await
    }

    async fn prompt_messages_with_options<M>(
        &self,
        messages: Vec<M>,
        skip_initial_queued_message_poll: bool,
    ) -> Result<Vec<AgentMessage>, String>
    where
        M: Into<AgentMessage>,
    {
        self.run_with_skip(
            |context, config| agent_loop_prompt_messages(context, messages, config),
            skip_initial_queued_message_poll,
        )
        .await
    }

    async fn run_with_skip<F, Fut>(
        &self,
        f: F,
        skip_initial_queued_message_poll: bool,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnOnce(AgentContext, AgentLoopConfig) -> Fut,
        Fut: std::future::Future<Output = Result<(Vec<AgentMessage>, Vec<AgentEvent>), String>>,
    {
        let (context, model, thinking_level) = {
            let mut state = self.state.lock();
            if state.is_streaming {
                return Err("Agent is already streaming".to_owned());
            }

            state.is_streaming = true;
            state.error_message = None;
            (
                AgentContext {
                    system_prompt: state.system_prompt.clone(),
                    messages: state.messages.clone(),
                    tools: state.tools.clone(),
                },
                state.model.clone(),
                state.thinking_level,
            )
        };

        self.abort_flag.store(false, Ordering::SeqCst);

        let mut stream_options = self.stream_options.clone();
        stream_options.reasoning = (thinking_level != ThinkingLevel::Off).then_some(thinking_level);
        stream_options.stream.abort_flag = Some(self.abort_flag.clone());

        let event_sink = AgentRuntimeEventSink {
            state: self.state.clone(),
            listeners: self.listeners.clone(),
            abort_flag: self.abort_flag.clone(),
        };

        let config = AgentLoopConfig {
            model,
            stream_options,
            tool_call_hooks: self.tool_call_hooks.clone(),
            tool_result_hooks: self.tool_result_hooks.clone(),
            transform_context: self.transform_context.clone(),
            convert_to_llm: self.convert_to_llm.clone(),
            stream_provider: self.stream_provider.clone(),
            api_key_provider: self.api_key_provider.clone(),
            prepare_next_turn: self.prepare_next_turn.clone(),
            should_stop_after_turn: self.should_stop_after_turn.clone(),
            queued_message_provider: Some(Arc::new(QueuedMessageProviderChain {
                queue: self.steering_queue.clone(),
                extra: self.queued_message_provider.clone(),
            })),
            follow_up_message_provider: Some(Arc::new(QueuedMessageProviderChain {
                queue: self.follow_up_queue.clone(),
                extra: self.follow_up_message_provider.clone(),
            })),
            event_sink: Some(Arc::new(event_sink)),
            skip_initial_queued_message_poll,
            tool_execution: self.tool_execution,
            max_turns: self.max_turns,
        };

        let outcome = f(context, config).await;
        match outcome {
            Ok((messages, _events)) => {
                self.finish_run();
                Ok(messages)
            }
            Err(error) => {
                self.state.lock().error_message = Some(error.clone());
                self.finish_run();
                Err(error)
            }
        }
    }

    fn finish_run(&self) {
        let mut state = self.state.lock();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        drop(state);
        self.idle_notify.notify_waiters();
    }
}
