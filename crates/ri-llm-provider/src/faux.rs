use crate::{
    api_registry::{
        ApiProvider, ProviderError, ensure_model_api, register_api_provider,
        unregister_api_providers,
    },
    event_stream::{
        AssistantMessageEventSender, AssistantMessageEventStream, assistant_message_event_stream,
    },
    types::{
        AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
        ImageContent, Message, Model, ModelCost, SimpleStreamOptions, StopReason, StreamOptions,
        TextContent, ThinkingContent, ToolCall, ToolResultContent, ToolResultMessage, Usage,
        now_millis,
    },
};
use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::{
    any::Any,
    collections::{BTreeMap, VecDeque},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: bool,
    pub input: Vec<crate::types::InputKind>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
}

impl FauxModelDefinition {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            reasoning: false,
            input: vec![
                crate::types::InputKind::Text,
                crate::types::InputKind::Image,
            ],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FauxAssistantOptions {
    pub stop_reason: Option<StopReason>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<i64>,
}

pub enum FauxContent {
    Text(String),
    Block(AssistantContent),
    Blocks(Vec<AssistantContent>),
}

impl From<&str> for FauxContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for FauxContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<AssistantContent> for FauxContent {
    fn from(value: AssistantContent) -> Self {
        Self::Block(value)
    }
}

impl From<Vec<AssistantContent>> for FauxContent {
    fn from(value: Vec<AssistantContent>) -> Self {
        Self::Blocks(value)
    }
}

pub fn faux_text(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(TextContent::new(text))
}

pub fn faux_thinking(thinking: impl Into<String>) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent::new(thinking))
}

pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: Map<String, Value>,
    id: Option<String>,
) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.unwrap_or_else(|| random_id("tool")),
        name: name.into(),
        arguments,
        thought_signature: None,
    })
}

pub fn faux_assistant_message(
    content: impl Into<FauxContent>,
    options: FauxAssistantOptions,
) -> AssistantMessage {
    AssistantMessage {
        content: normalize_content(content.into()),
        api: DEFAULT_API.to_owned(),
        provider: DEFAULT_PROVIDER.to_owned(),
        model: DEFAULT_MODEL_ID.to_owned(),
        response_model: None,
        response_id: options.response_id,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        error_message: options.error_message,
        timestamp: options.timestamp.unwrap_or_else(now_millis),
    }
}

fn normalize_content(content: FauxContent) -> Vec<AssistantContent> {
    match content {
        FauxContent::Text(text) => vec![faux_text(text)],
        FauxContent::Block(block) => vec![block],
        FauxContent::Blocks(blocks) => blocks,
    }
}

type FauxResponseFactory = Arc<
    dyn Fn(&Context, &SimpleStreamOptions, FauxState, &Model) -> AssistantMessage + Send + Sync,
>;
type FauxResponseFuture = Pin<Box<dyn Future<Output = AssistantMessage> + Send>>;
type FauxAsyncResponseFactory =
    Arc<dyn Fn(Context, SimpleStreamOptions, FauxState, Model) -> FauxResponseFuture + Send + Sync>;

#[derive(Clone)]
pub enum FauxResponseStep {
    Message(AssistantMessage),
    Factory(FauxResponseFactory),
    AsyncFactory(FauxAsyncResponseFactory),
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(value: AssistantMessage) -> Self {
        Self::Message(value)
    }
}

pub fn faux_response_factory(
    factory: impl Fn(&Context, &SimpleStreamOptions, FauxState, &Model) -> AssistantMessage
    + Send
    + Sync
    + 'static,
) -> FauxResponseStep {
    FauxResponseStep::Factory(Arc::new(factory))
}

pub fn faux_async_response_factory<F, Fut>(factory: F) -> FauxResponseStep
where
    F: Fn(Context, SimpleStreamOptions, FauxState, Model) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = AssistantMessage> + Send + 'static,
{
    FauxResponseStep::AsyncFactory(Arc::new(move |context, options, state, model| {
        Box::pin(factory(context, options, state, model))
    }))
}

#[derive(Debug, Clone)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    pub models: Vec<FauxModelDefinition>,
    pub tokens_per_second: Option<f64>,
    pub token_size: Option<TokenSize>,
}

impl Default for RegisterFauxProviderOptions {
    fn default() -> Self {
        Self {
            api: None,
            provider: None,
            models: Vec::new(),
            tokens_per_second: None,
            token_size: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TokenSize {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Clone)]
pub struct FauxState {
    call_count: Arc<AtomicUsize>,
}

impl FauxState {
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

pub struct FauxProviderRegistration {
    api: String,
    provider: Arc<FauxProvider>,
    source_id: String,
    pub models: Vec<Model>,
}

impl FauxProviderRegistration {
    pub fn api(&self) -> &str {
        &self.api
    }

    pub fn state(&self) -> FauxState {
        FauxState {
            call_count: self.provider.call_count.clone(),
        }
    }

    pub fn get_model(&self) -> Model {
        self.models[0].clone()
    }

    pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    pub fn set_responses(&self, responses: impl IntoIterator<Item = FauxResponseStep>) {
        *self.provider.pending_responses.lock() = responses.into_iter().collect();
    }

    pub fn append_responses(&self, responses: impl IntoIterator<Item = FauxResponseStep>) {
        self.provider.pending_responses.lock().extend(responses);
    }

    pub fn pending_response_count(&self) -> usize {
        self.provider.pending_responses.lock().len()
    }

    pub fn unregister(&self) {
        unregister_api_providers(&self.source_id);
    }
}

struct FauxProvider {
    api: String,
    provider: String,
    pending_responses: Mutex<VecDeque<FauxResponseStep>>,
    call_count: Arc<AtomicUsize>,
    prompt_cache: Arc<Mutex<BTreeMap<String, String>>>,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
}

impl ApiProvider for FauxProvider {
    fn api(&self) -> &str {
        &self.api
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_with_simple_options(
            model,
            context,
            SimpleStreamOptions {
                stream: options,
                ..Default::default()
            },
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_with_simple_options(model, context, options)
    }
}

impl FauxProvider {
    fn stream_with_simple_options(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, &self.api)?;

        let (sender, stream) = assistant_message_event_stream();
        let step = self.pending_responses.lock().pop_front();
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let api = self.api.clone();
        let provider = self.provider.clone();
        let model = model.clone();
        let call_count = self.call_count.clone();
        let min_token_size = self.min_token_size;
        let max_token_size = self.max_token_size;
        let tokens_per_second = self.tokens_per_second;
        let prompt_cache = self.prompt_cache.clone();
        let abort_flag = options.stream.abort_flag.clone();

        tokio::spawn(async move {
            let message = match step {
                Some(FauxResponseStep::Message(message)) => Ok(message),
                Some(FauxResponseStep::Factory(factory)) => {
                    let state = FauxState { call_count };
                    std::panic::catch_unwind(AssertUnwindSafe(|| {
                        factory(&context, &options, state, &model)
                    }))
                    .map_err(|panic| panic_payload_to_string(panic.as_ref()))
                }
                Some(FauxResponseStep::AsyncFactory(factory)) => {
                    let state = FauxState { call_count };
                    let future = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        factory(context.clone(), options.clone(), state, model.clone())
                    }))
                    .map_err(|panic| panic_payload_to_string(panic.as_ref()));
                    match future {
                        Ok(future) => AssertUnwindSafe(future)
                            .catch_unwind()
                            .await
                            .map_err(|panic| panic_payload_to_string(panic.as_ref())),
                        Err(error) => Err(error),
                    }
                }
                None => Ok(create_error_message(
                    "No more faux responses queued",
                    &api,
                    &provider,
                    &model.id,
                )),
            };

            let message = match message {
                Ok(message) => message,
                Err(error) => {
                    let error = with_usage_estimate(
                        create_error_message(error, &api, &provider, &model.id),
                        &context,
                        &options.stream,
                        &prompt_cache,
                    );
                    sender.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: error.clone(),
                    });
                    sender.end(error);
                    return;
                }
            };

            let mut message = clone_message(message, &api, &provider, &model.id);
            message = with_usage_estimate(message, &context, &options.stream, &prompt_cache);
            stream_with_deltas(
                sender,
                message,
                min_token_size,
                max_token_size,
                tokens_per_second,
                abort_flag,
            )
            .await;
        });

        Ok(stream)
    }
}

pub fn register_faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderRegistration {
    let api = options.api.unwrap_or_else(|| random_id(DEFAULT_API));
    let provider_name = options
        .provider
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_owned());
    let source_id = random_id("faux-provider");
    let token_size = options.token_size.unwrap_or(TokenSize {
        min: DEFAULT_MIN_TOKEN_SIZE,
        max: DEFAULT_MAX_TOKEN_SIZE,
    });
    let min_token_size = token_size.min.max(1).min(token_size.max.max(1));
    let max_token_size = token_size.max.max(min_token_size);

    let definitions = if options.models.is_empty() {
        vec![FauxModelDefinition {
            id: DEFAULT_MODEL_ID.to_owned(),
            name: Some(DEFAULT_MODEL_NAME.to_owned()),
            reasoning: false,
            input: vec![
                crate::types::InputKind::Text,
                crate::types::InputKind::Image,
            ],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
        }]
    } else {
        options.models
    };

    let models: Vec<Model> = definitions
        .into_iter()
        .map(|definition| Model {
            id: definition.id.clone(),
            name: definition.name.unwrap_or(definition.id),
            api: api.clone(),
            provider: provider_name.clone(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            reasoning: definition.reasoning,
            thinking_level_map: BTreeMap::new(),
            input: definition.input,
            cost: definition.cost,
            context_window: definition.context_window,
            max_tokens: definition.max_tokens,
            headers: BTreeMap::new(),
            compat: None,
        })
        .collect();

    let provider = Arc::new(FauxProvider {
        api: api.clone(),
        provider: provider_name,
        pending_responses: Mutex::new(VecDeque::new()),
        call_count: Arc::new(AtomicUsize::new(0)),
        prompt_cache: Arc::new(Mutex::new(BTreeMap::new())),
        min_token_size,
        max_token_size,
        tokens_per_second: options.tokens_per_second,
    });

    register_api_provider(provider.clone(), Some(source_id.clone()));

    FauxProviderRegistration {
        api,
        provider,
        source_id,
        models,
    }
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Faux response factory panicked".to_owned()
    }
}

fn estimate_tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(4) as u64
}

fn random_id(prefix: &str) -> String {
    let sequence = ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{prefix}:{}:{sequence}", now_millis())
}

fn content_to_text(content: &crate::types::UserContentValue) -> String {
    match content {
        crate::types::UserContentValue::Plain(text) => text.clone(),
        crate::types::UserContentValue::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                crate::types::UserContent::Text(text) => text.text.clone(),
                crate::types::UserContent::Image(image) => {
                    format!("[image:{}:{}]", image.mime_type, image.data.len())
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn assistant_content_to_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(text) => text.text.clone(),
            AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContent::ToolCall(tool_call) => {
                format!(
                    "{}:{}",
                    tool_call.name,
                    serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_owned())
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
    let mut lines = vec![message.tool_name.clone()];
    lines.extend(message.content.iter().map(|block| match block {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Image(ImageContent { data, mime_type }) => {
            format!("[image:{mime_type}:{}]", data.len())
        }
    }));
    lines.join("\n")
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::User(message) => content_to_text(&message.content),
        Message::Assistant(message) => assistant_content_to_text(&message.content),
        Message::ToolResult(message) => tool_result_to_text(message),
    }
}

fn serialize_context(context: &Context) -> String {
    let mut parts = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        parts.push(format!("system:{system_prompt}"));
    }
    parts.extend(
        context
            .messages
            .iter()
            .map(|message| format!("{}:{}", message_role(message), message_to_text(message))),
    );
    if !context.tools.is_empty() {
        parts.push(format!(
            "tools:{}",
            serde_json::to_string(&context.tools).unwrap_or_default()
        ));
    }
    parts.join("\n\n")
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn common_prefix_length(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(left, right)| left == right)
        .map(|(ch, _)| ch.len_utf8())
        .sum()
}

fn with_usage_estimate(
    mut message: AssistantMessage,
    context: &Context,
    options: &StreamOptions,
    prompt_cache: &Mutex<BTreeMap<String, String>>,
) -> AssistantMessage {
    let prompt_text = serialize_context(context);
    let prompt_tokens = estimate_tokens(&prompt_text);
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;

    if let Some(session_id) = &options.session_id {
        if options.cache_retention != Some(CacheRetention::None) {
            let mut prompt_cache = prompt_cache.lock();
            if let Some(previous_prompt) = prompt_cache.get(session_id) {
                let cached_chars = common_prefix_length(previous_prompt, &prompt_text);
                cache_read = estimate_tokens(&previous_prompt[..cached_chars]);
                cache_write = estimate_tokens(&prompt_text[cached_chars..]);
                input = prompt_tokens.saturating_sub(cache_read);
            } else {
                cache_write = prompt_tokens;
            }
            prompt_cache.insert(session_id.clone(), prompt_text);
        }
    }

    message.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: Default::default(),
    };
    message
}

fn clone_message(
    mut message: AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    message.api = api.to_owned();
    message.provider = provider.to_owned();
    message.model = model_id.to_owned();
    if message.timestamp == 0 {
        message.timestamp = now_millis();
    }
    message
}

fn create_error_message(
    error: impl Into<String>,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: api.to_owned(),
        provider: provider.to_owned(),
        model: model_id.to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Error,
        error_message: Some(error.into()),
        timestamp: now_millis(),
    }
}

fn chunks_for_text(text: &str, min_token_size: usize, max_token_size: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let char_size = min_token_size.max(max_token_size).max(1) * 4;
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if current.chars().count() >= char_size {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    let Some(tokens_per_second) = tokens_per_second else {
        tokio::task::yield_now().await;
        return;
    };
    if tokens_per_second <= 0.0 {
        tokio::task::yield_now().await;
        return;
    }
    let delay = Duration::from_secs_f64(estimate_tokens(chunk) as f64 / tokens_per_second);
    tokio::time::sleep(delay).await;
}

async fn stream_with_deltas(
    sender: AssistantMessageEventSender,
    message: AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    abort_flag: Option<Arc<AtomicBool>>,
) {
    let mut partial = AssistantMessage {
        content: Vec::new(),
        ..message.clone()
    };

    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
        return;
    }

    sender.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    for (index, block) in message.content.iter().cloned().enumerate() {
        match block {
            AssistantContent::Thinking(thinking) => {
                partial
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent::new("")));
                sender.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                for chunk in chunks_for_text(&thinking.thinking, min_token_size, max_token_size) {
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    if let AssistantContent::Thinking(partial_thinking) =
                        &mut partial.content[index]
                    {
                        partial_thinking.thinking.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: thinking.thinking,
                    partial: partial.clone(),
                });
            }
            AssistantContent::Text(text) => {
                partial
                    .content
                    .push(AssistantContent::Text(TextContent::new("")));
                sender.push(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                for chunk in chunks_for_text(&text.text, min_token_size, max_token_size) {
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    if let AssistantContent::Text(partial_text) = &mut partial.content[index] {
                        partial_text.text.push_str(&chunk);
                    }
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: text.text,
                    partial: partial.clone(),
                });
            }
            AssistantContent::ToolCall(tool_call) => {
                partial.content.push(AssistantContent::ToolCall(ToolCall {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: Map::new(),
                    thought_signature: tool_call.thought_signature.clone(),
                }));
                sender.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index,
                    partial: partial.clone(),
                });
                let arguments =
                    serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_owned());
                for chunk in chunks_for_text(&arguments, min_token_size, max_token_size) {
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    schedule_chunk(&chunk, tokens_per_second).await;
                    if push_abort_if_requested(&sender, abort_flag.as_deref(), &partial) {
                        return;
                    }
                    sender.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: index,
                        delta: chunk,
                        partial: partial.clone(),
                    });
                }
                if let AssistantContent::ToolCall(partial_tool_call) = &mut partial.content[index] {
                    partial_tool_call.arguments = tool_call.arguments.clone();
                }
                sender.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index,
                    tool_call,
                    partial: partial.clone(),
                });
            }
        }
    }

    match message.stop_reason {
        StopReason::Error | StopReason::Aborted => {
            sender.push(AssistantMessageEvent::Error {
                reason: message.stop_reason,
                error: message.clone(),
            });
            sender.end(message);
        }
        StopReason::Stop | StopReason::Length | StopReason::ToolUse => {
            sender.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message: message.clone(),
            });
            sender.end(message);
        }
    }
}

fn push_abort_if_requested(
    sender: &AssistantMessageEventSender,
    abort_flag: Option<&AtomicBool>,
    partial: &AssistantMessage,
) -> bool {
    if !abort_flag.is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst)) {
        return false;
    }

    let mut error = partial.clone();
    error.stop_reason = StopReason::Aborted;
    error.error_message = Some("Request was aborted".to_owned());
    error.timestamp = now_millis();
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Aborted,
        error: error.clone(),
    });
    sender.end(error);
    true
}
