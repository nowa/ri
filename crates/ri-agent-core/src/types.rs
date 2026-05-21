use async_trait::async_trait;
use futures::future::BoxFuture;
use ri_llm_provider::{
    AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Context, ImageContent,
    Message, Model, SimpleStreamOptions, TextContent, ThinkingLevel, Tool, ToolCall,
    ToolResultMessage, UserMessage,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug, Clone, PartialEq)]
pub enum AgentMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    Custom(Value),
}

impl AgentMessage {
    pub fn custom(value: impl Into<Value>) -> Self {
        Self::Custom(value.into())
    }

    pub fn into_llm_message(self) -> Option<Message> {
        match self {
            Self::User(message) => Some(Message::User(message)),
            Self::Assistant(message) => Some(Message::Assistant(message)),
            Self::ToolResult(message) => Some(Message::ToolResult(message)),
            Self::Custom(_) => None,
        }
    }

    pub fn to_llm_message(&self) -> Option<Message> {
        self.clone().into_llm_message()
    }

    pub fn role(&self) -> Option<&str> {
        match self {
            Self::User(_) => Some("user"),
            Self::Assistant(_) => Some("assistant"),
            Self::ToolResult(_) => Some("toolResult"),
            Self::Custom(value) => value.get("role").and_then(Value::as_str),
        }
    }

    pub fn is_assistant(&self) -> bool {
        matches!(self, Self::Assistant(_))
    }
}

impl From<Message> for AgentMessage {
    fn from(value: Message) -> Self {
        match value {
            Message::User(message) => Self::User(message),
            Message::Assistant(message) => Self::Assistant(message),
            Message::ToolResult(message) => Self::ToolResult(message),
        }
    }
}

impl From<UserMessage> for AgentMessage {
    fn from(value: UserMessage) -> Self {
        Self::User(value)
    }
}

impl From<AssistantMessage> for AgentMessage {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ToolResultMessage> for AgentMessage {
    fn from(value: ToolResultMessage) -> Self {
        Self::ToolResult(value)
    }
}

impl TryFrom<AgentMessage> for Message {
    type Error = AgentMessage;

    fn try_from(value: AgentMessage) -> Result<Self, Self::Error> {
        value.clone().into_llm_message().ok_or(value)
    }
}

impl Serialize for AgentMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::User(message) => Message::User(message.clone()).serialize(serializer),
            Self::Assistant(message) => Message::Assistant(message.clone()).serialize(serializer),
            Self::ToolResult(message) => Message::ToolResult(message.clone()).serialize(serializer),
            Self::Custom(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for AgentMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match serde_json::from_value::<Message>(value.clone()) {
            Ok(message) => Ok(message.into()),
            Err(_) => Ok(Self::Custom(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    All,
    OneAtATime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult {
    pub content: Vec<AgentToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminate: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AgentToolResultContent {
    Text(TextContent),
    Image(ImageContent),
}

impl AgentToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![AgentToolResultContent::Text(TextContent::new(text))],
            details: None,
            terminate: false,
        }
    }
}

#[async_trait]
pub trait AgentToolExecutor: Send + Sync {
    async fn execute(&self, tool_call_id: &str, params: Value) -> Result<AgentToolResult, String>;

    async fn execute_with_updates(
        &self,
        tool_call_id: &str,
        params: Value,
        on_update: AgentToolUpdateCallback,
    ) -> Result<AgentToolResult, String> {
        let _ = on_update;
        self.execute(tool_call_id, params).await
    }
}

pub type AgentToolUpdateCallback =
    Arc<dyn Fn(AgentToolResult) -> BoxFuture<'static, ()> + Send + Sync + 'static>;

#[derive(Debug, Clone)]
pub struct AgentToolCallHookContext {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: Value,
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolCallHookResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub block: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl AgentToolCallHookResult {
    pub fn replace_input(input: Value) -> Self {
        Self {
            input: Some(input),
            block: false,
            reason: None,
        }
    }

    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            input: None,
            block: true,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentToolResultHookContext {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResultHookResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AgentToolResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl AgentToolResultHookResult {
    pub fn replace_result(result: AgentToolResult) -> Self {
        Self {
            result: Some(result),
            is_error: None,
        }
    }

    pub fn set_is_error(is_error: bool) -> Self {
        Self {
            result: None,
            is_error: Some(is_error),
        }
    }
}

#[async_trait]
pub trait AgentToolCallHook: Send + Sync {
    async fn on_tool_call(
        &self,
        context: AgentToolCallHookContext,
    ) -> Result<Option<AgentToolCallHookResult>, String>;
}

#[async_trait]
pub trait AgentToolResultHook: Send + Sync {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String>;
}

#[async_trait]
pub trait AgentContextTransformer: Send + Sync {
    async fn transform_context(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, String>;
}

pub trait AgentMessageConverter: Send + Sync {
    fn convert_to_llm(&self, messages: &[AgentMessage]) -> Result<Vec<Message>, String>;
}

pub trait AgentStreamProvider: Send + Sync {
    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, String>;
}

#[async_trait]
pub trait AgentApiKeyProvider: Send + Sync {
    async fn get_api_key(&self, provider: &str) -> Result<Option<String>, String>;
}

#[async_trait]
pub trait AgentQueuedMessageProvider: Send + Sync {
    async fn get_queued_messages(&self) -> Result<Vec<AgentMessage>, String>;
}

#[derive(Debug, Clone)]
pub struct AgentNextTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: AgentContext,
    pub new_messages: Vec<AgentMessage>,
}

#[derive(Debug, Clone)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
    pub stream_options: Option<SimpleStreamOptions>,
}

#[async_trait]
pub trait AgentNextTurnPreparer: Send + Sync {
    async fn prepare_next_turn(
        &self,
        context: AgentNextTurnContext,
    ) -> Result<Option<AgentLoopTurnUpdate>, String>;
}

#[async_trait]
pub trait AgentShouldStopAfterTurn: Send + Sync {
    async fn should_stop_after_turn(&self, context: AgentNextTurnContext) -> Result<bool, String>;
}

pub trait AgentToolArgumentPreparer: Send + Sync {
    fn prepare_arguments(&self, args: Value) -> Result<Value, String>;
}

pub struct AgentTool {
    pub definition: Tool,
    pub label: String,
    pub execution_mode: Option<ToolExecutionMode>,
    pub argument_preparer: Option<Arc<dyn AgentToolArgumentPreparer>>,
    pub executor: Arc<dyn AgentToolExecutor>,
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("definition", &self.definition)
            .field("label", &self.label)
            .field("execution_mode", &self.execution_mode)
            .field(
                "argument_preparer",
                &self
                    .argument_preparer
                    .as_ref()
                    .map(|_| "AgentToolArgumentPreparer"),
            )
            .finish_non_exhaustive()
    }
}

impl Clone for AgentTool {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            label: self.label.clone(),
            execution_mode: self.execution_mode,
            argument_preparer: self.argument_preparer.clone(),
            executor: self.executor.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<AgentTool>,
}

impl AgentContext {
    pub fn to_llm_context(&self) -> Context {
        Context {
            system_prompt: (!self.system_prompt.is_empty()).then_some(self.system_prompt.clone()),
            messages: self
                .messages
                .iter()
                .filter_map(AgentMessage::to_llm_message)
                .collect(),
            tools: self
                .tools
                .iter()
                .map(|tool| tool.definition.clone())
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

#[async_trait]
pub trait AgentEventSink: Send + Sync {
    async fn on_event(&self, event: &AgentEvent);
}

#[derive(Debug, Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<AgentTool>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: BTreeSet<String>,
    pub error_message: Option<String>,
}

impl AgentState {
    pub fn new(model: Model) -> Self {
        Self {
            system_prompt: String::new(),
            model,
            thinking_level: ThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: BTreeSet::new(),
            error_message: None,
        }
    }
}

pub fn assistant_tool_calls(message: &AssistantMessage) -> impl Iterator<Item = &ToolCall> {
    message.content.iter().filter_map(|content| match content {
        ri_llm_provider::AssistantContent::ToolCall(tool_call) => Some(tool_call),
        _ => None,
    })
}
