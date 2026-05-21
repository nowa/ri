use parking_lot::Mutex;
use ri_llm_provider::{Message, UserContent, now_millis};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{Display, Formatter},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionErrorCode {
    NotFound,
    InvalidSession,
    InvalidEntry,
    InvalidForkTarget,
    Storage,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionError {
    pub code: SessionErrorCode,
    pub message: String,
}

impl SessionError {
    pub fn new(code: SessionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl Display for SessionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SessionError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    pub id: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlSessionMetadata {
    pub id: String,
    pub created_at: String,
    pub cwd: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    Text(String),
    Blocks(Vec<UserContent>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashExecutionMessage {
    pub command: String,
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    pub exclude_from_context: bool,
}

impl BashExecutionMessage {
    pub fn new(command: impl Into<String>, output: impl Into<String>, timestamp: i64) -> Self {
        Self {
            command: command.into(),
            output: output.into(),
            exit_code: None,
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp,
            exclude_from_context: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BashExecutionMessageWire {
    role: String,
    command: String,
    output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
    cancelled: bool,
    truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_output_path: Option<String>,
    timestamp: i64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    exclude_from_context: bool,
}

impl From<BashExecutionMessage> for BashExecutionMessageWire {
    fn from(message: BashExecutionMessage) -> Self {
        Self {
            role: "bashExecution".to_owned(),
            command: message.command,
            output: message.output,
            exit_code: message.exit_code,
            cancelled: message.cancelled,
            truncated: message.truncated,
            full_output_path: message.full_output_path,
            timestamp: message.timestamp,
            exclude_from_context: message.exclude_from_context,
        }
    }
}

impl TryFrom<BashExecutionMessageWire> for BashExecutionMessage {
    type Error = String;

    fn try_from(wire: BashExecutionMessageWire) -> Result<Self, Self::Error> {
        if wire.role != "bashExecution" {
            return Err(format!(
                "expected bashExecution message role, got {}",
                wire.role
            ));
        }
        Ok(Self {
            command: wire.command,
            output: wire.output,
            exit_code: wire.exit_code,
            cancelled: wire.cancelled,
            truncated: wire.truncated,
            full_output_path: wire.full_output_path,
            timestamp: wire.timestamp,
            exclude_from_context: wire.exclude_from_context,
        })
    }
}

impl Serialize for BashExecutionMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        BashExecutionMessageWire::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BashExecutionMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        BashExecutionMessageWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionEntryMessage {
    Llm(Message),
    BashExecution(BashExecutionMessage),
}

impl SessionEntryMessage {
    pub fn as_llm_message(&self) -> Option<&Message> {
        match self {
            Self::Llm(message) => Some(message),
            Self::BashExecution(_) => None,
        }
    }

    pub fn into_llm_message(self) -> Option<Message> {
        match self {
            Self::Llm(message) => Some(message),
            Self::BashExecution(_) => None,
        }
    }

    pub fn role(&self) -> &'static str {
        match self {
            Self::Llm(Message::User(_)) => "user",
            Self::Llm(Message::Assistant(_)) => "assistant",
            Self::Llm(Message::ToolResult(_)) => "toolResult",
            Self::BashExecution(_) => "bashExecution",
        }
    }
}

impl From<Message> for SessionEntryMessage {
    fn from(message: Message) -> Self {
        Self::Llm(message)
    }
}

impl From<BashExecutionMessage> for SessionEntryMessage {
    fn from(message: BashExecutionMessage) -> Self {
        Self::BashExecution(message)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum SessionMessage {
    #[serde(rename = "llm")]
    Llm {
        message: Message,
    },
    BashExecution(BashExecutionMessage),
    Custom {
        custom_type: String,
        content: CustomMessageContent,
        display: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        timestamp: i64,
    },
    BranchSummary {
        summary: String,
        from_id: String,
        timestamp: i64,
    },
    CompactionSummary {
        summary: String,
        tokens_before: u64,
        timestamp: i64,
    },
}

impl SessionMessage {
    pub fn role(&self) -> &'static str {
        match self {
            Self::Llm {
                message: Message::User(_),
            } => "user",
            Self::Llm {
                message: Message::Assistant(_),
            } => "assistant",
            Self::Llm {
                message: Message::ToolResult(_),
            } => "toolResult",
            Self::BashExecution(_) => "bashExecution",
            Self::Custom { .. } => "custom",
            Self::BranchSummary { .. } => "branchSummary",
            Self::CompactionSummary { .. } => "compactionSummary",
        }
    }
}

impl From<Message> for SessionMessage {
    fn from(message: Message) -> Self {
        Self::Llm { message }
    }
}

impl From<BashExecutionMessage> for SessionMessage {
    fn from(message: BashExecutionMessage) -> Self {
        Self::BashExecution(message)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionTreeEntry {
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        message: SessionEntryMessage,
    },
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
    },
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
    Compaction {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        #[serde(rename = "firstKeptEntryId")]
        first_kept_entry_id: String,
        #[serde(rename = "tokensBefore")]
        tokens_before: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "fromId")]
        from_id: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    Custom {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    CustomMessage {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        content: CustomMessageContent,
        display: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },
    Label {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    SessionInfo {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Leaf {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: Option<String>,
    },
}

impl SessionTreeEntry {
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::ThinkingLevelChange { id, .. }
            | Self::ModelChange { id, .. }
            | Self::Compaction { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::Custom { id, .. }
            | Self::CustomMessage { id, .. }
            | Self::Label { id, .. }
            | Self::SessionInfo { id, .. }
            | Self::Leaf { id, .. } => id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::ThinkingLevelChange { parent_id, .. }
            | Self::ModelChange { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::Custom { parent_id, .. }
            | Self::CustomMessage { parent_id, .. }
            | Self::Label { parent_id, .. }
            | Self::SessionInfo { parent_id, .. }
            | Self::Leaf { parent_id, .. } => parent_id.as_deref(),
        }
    }

    pub fn entry_type(&self) -> &'static str {
        match self {
            Self::Message { .. } => "message",
            Self::ThinkingLevelChange { .. } => "thinking_level_change",
            Self::ModelChange { .. } => "model_change",
            Self::Compaction { .. } => "compaction",
            Self::BranchSummary { .. } => "branch_summary",
            Self::Custom { .. } => "custom",
            Self::CustomMessage { .. } => "custom_message",
            Self::Label { .. } => "label",
            Self::SessionInfo { .. } => "session_info",
            Self::Leaf { .. } => "leaf",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionContext {
    pub messages: Vec<SessionMessage>,
    pub thinking_level: String,
    pub model: Option<SessionModelSelection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionModelSelection {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone)]
pub struct InMemorySessionStorage {
    metadata: SessionMetadata,
    entries: Vec<SessionTreeEntry>,
    by_id: BTreeMap<String, SessionTreeEntry>,
    labels_by_id: BTreeMap<String, String>,
    leaf_id: Option<String>,
}

impl InMemorySessionStorage {
    pub fn new() -> Self {
        Self::with_options(None, None)
    }

    pub fn with_options(
        entries: Option<Vec<SessionTreeEntry>>,
        metadata: Option<SessionMetadata>,
    ) -> Self {
        let entries = entries.unwrap_or_default();
        let by_id = entries
            .iter()
            .map(|entry| (entry.id().to_owned(), entry.clone()))
            .collect();
        let labels_by_id = build_labels_by_id(&entries);
        let mut leaf_id = None;
        for entry in &entries {
            leaf_id = leaf_id_after_entry(entry);
        }
        Self {
            metadata: metadata.unwrap_or_else(|| SessionMetadata {
                id: uuidv7(),
                created_at: now_iso(),
            }),
            entries,
            by_id,
            labels_by_id,
            leaf_id,
        }
    }

    pub fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        validate_existing_leaf(&self.by_id, self.leaf_id.as_deref())?;
        Ok(self.leaf_id.clone())
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        validate_leaf(&self.by_id, leaf_id.as_deref())?;
        let entry = SessionTreeEntry::Leaf {
            id: generate_entry_id(&self.by_id),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso(),
            target_id: leaf_id.clone(),
        };
        self.entries.push(entry.clone());
        self.by_id.insert(entry.id().to_owned(), entry);
        self.leaf_id = leaf_id;
        Ok(())
    }

    pub fn create_entry_id(&self) -> String {
        generate_entry_id(&self.by_id)
    }

    pub fn append_entry(&mut self, entry: SessionTreeEntry) {
        update_label_cache(&mut self.labels_by_id, &entry);
        self.leaf_id = leaf_id_after_entry(&entry);
        self.by_id.insert(entry.id().to_owned(), entry.clone());
        self.entries.push(entry);
    }

    pub fn get_entry(&self, id: &str) -> Option<&SessionTreeEntry> {
        self.by_id.get(id)
    }

    pub fn find_entries(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.entry_type() == entry_type)
            .cloned()
            .collect()
    }

    pub fn label(&self, id: &str) -> Option<&str> {
        self.labels_by_id.get(id).map(String::as_str)
    }

    pub fn path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        get_path_to_root(&self.by_id, leaf_id)
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        self.entries.clone()
    }
}

impl Default for InMemorySessionStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct JsonlSessionStorage {
    file_path: PathBuf,
    metadata: JsonlSessionMetadata,
    entries: Vec<SessionTreeEntry>,
    by_id: BTreeMap<String, SessionTreeEntry>,
    labels_by_id: BTreeMap<String, String>,
    leaf_id: Option<String>,
}

impl JsonlSessionStorage {
    pub fn create(
        file_path: impl AsRef<Path>,
        cwd: impl Into<String>,
        session_id: impl Into<String>,
        parent_session_path: Option<String>,
    ) -> Result<Self, SessionError> {
        let file_path = file_path.as_ref().to_path_buf();
        let header = SessionHeader {
            entry_type: "session".to_owned(),
            version: 3,
            id: session_id.into(),
            timestamp: now_iso(),
            cwd: cwd.into(),
            parent_session: parent_session_path,
        };
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(storage_error)?;
        }
        let header_line = serde_json::to_string(&header).map_err(storage_error)?;
        fs::write(&file_path, format!("{header_line}\n")).map_err(storage_error)?;
        Ok(Self::from_loaded(file_path, header, Vec::new(), None))
    }

    pub fn open(file_path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let file_path = file_path.as_ref().to_path_buf();
        let (header, entries, leaf_id) = load_jsonl_storage(&file_path)?;
        Ok(Self::from_loaded(file_path, header, entries, leaf_id))
    }

    fn from_loaded(
        file_path: PathBuf,
        header: SessionHeader,
        entries: Vec<SessionTreeEntry>,
        leaf_id: Option<String>,
    ) -> Self {
        let by_id = entries
            .iter()
            .map(|entry| (entry.id().to_owned(), entry.clone()))
            .collect();
        let labels_by_id = build_labels_by_id(&entries);
        let metadata = JsonlSessionMetadata {
            id: header.id,
            created_at: header.timestamp,
            cwd: header.cwd,
            path: file_path.to_string_lossy().into_owned(),
            parent_session_path: header.parent_session,
        };
        Self {
            file_path,
            metadata,
            entries,
            by_id,
            labels_by_id,
            leaf_id,
        }
    }

    pub fn metadata(&self) -> &JsonlSessionMetadata {
        &self.metadata
    }

    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        validate_existing_leaf(&self.by_id, self.leaf_id.as_deref())?;
        Ok(self.leaf_id.clone())
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        validate_leaf(&self.by_id, leaf_id.as_deref())?;
        let entry = SessionTreeEntry::Leaf {
            id: generate_entry_id(&self.by_id),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso(),
            target_id: leaf_id.clone(),
        };
        self.append_entry(entry)?;
        self.leaf_id = leaf_id;
        Ok(())
    }

    pub fn create_entry_id(&self) -> String {
        generate_entry_id(&self.by_id)
    }

    pub fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let line = serde_json::to_string(&entry).map_err(storage_error)?;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .and_then(|mut file| {
                use std::io::Write;
                writeln!(file, "{line}")
            })
            .map_err(storage_error)?;
        update_label_cache(&mut self.labels_by_id, &entry);
        self.leaf_id = leaf_id_after_entry(&entry);
        self.by_id.insert(entry.id().to_owned(), entry.clone());
        self.entries.push(entry);
        Ok(())
    }

    pub fn get_entry(&self, id: &str) -> Option<&SessionTreeEntry> {
        self.by_id.get(id)
    }

    pub fn find_entries(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.entry_type() == entry_type)
            .cloned()
            .collect()
    }

    pub fn label(&self, id: &str) -> Option<&str> {
        self.labels_by_id.get(id).map(String::as_str)
    }

    pub fn path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        get_path_to_root(&self.by_id, leaf_id)
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        self.entries.clone()
    }
}

#[derive(Debug, Clone)]
pub enum SessionStorageKind {
    InMemory(InMemorySessionStorage),
    Jsonl(JsonlSessionStorage),
}

impl SessionStorageKind {
    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        match self {
            Self::InMemory(storage) => storage.leaf_id(),
            Self::Jsonl(storage) => storage.leaf_id(),
        }
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        match self {
            Self::InMemory(storage) => storage.set_leaf_id(leaf_id),
            Self::Jsonl(storage) => storage.set_leaf_id(leaf_id),
        }
    }

    pub fn create_entry_id(&self) -> String {
        match self {
            Self::InMemory(storage) => storage.create_entry_id(),
            Self::Jsonl(storage) => storage.create_entry_id(),
        }
    }

    pub fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        match self {
            Self::InMemory(storage) => {
                storage.append_entry(entry);
                Ok(())
            }
            Self::Jsonl(storage) => storage.append_entry(entry),
        }
    }

    pub fn get_entry(&self, id: &str) -> Option<&SessionTreeEntry> {
        match self {
            Self::InMemory(storage) => storage.get_entry(id),
            Self::Jsonl(storage) => storage.get_entry(id),
        }
    }

    pub fn find_entries(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        match self {
            Self::InMemory(storage) => storage.find_entries(entry_type),
            Self::Jsonl(storage) => storage.find_entries(entry_type),
        }
    }

    pub fn label(&self, id: &str) -> Option<&str> {
        match self {
            Self::InMemory(storage) => storage.label(id),
            Self::Jsonl(storage) => storage.label(id),
        }
    }

    pub fn path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        match self {
            Self::InMemory(storage) => storage.path_to_root(leaf_id),
            Self::Jsonl(storage) => storage.path_to_root(leaf_id),
        }
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        match self {
            Self::InMemory(storage) => storage.entries(),
            Self::Jsonl(storage) => storage.entries(),
        }
    }
}

impl From<InMemorySessionStorage> for SessionStorageKind {
    fn from(value: InMemorySessionStorage) -> Self {
        Self::InMemory(value)
    }
}

impl From<JsonlSessionStorage> for SessionStorageKind {
    fn from(value: JsonlSessionStorage) -> Self {
        Self::Jsonl(value)
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    storage: Arc<Mutex<SessionStorageKind>>,
}

impl Session {
    pub fn new(storage: impl Into<SessionStorageKind>) -> Self {
        Self {
            storage: Arc::new(Mutex::new(storage.into())),
        }
    }

    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        self.storage.lock().leaf_id()
    }

    pub fn metadata_id(&self) -> String {
        match &*self.storage.lock() {
            SessionStorageKind::InMemory(storage) => storage.metadata().id.clone(),
            SessionStorageKind::Jsonl(storage) => storage.metadata().id.clone(),
        }
    }

    pub fn in_memory_metadata(&self) -> Option<SessionMetadata> {
        match &*self.storage.lock() {
            SessionStorageKind::InMemory(storage) => Some(storage.metadata().clone()),
            SessionStorageKind::Jsonl(_) => None,
        }
    }

    pub fn jsonl_metadata(&self) -> Option<JsonlSessionMetadata> {
        match &*self.storage.lock() {
            SessionStorageKind::InMemory(_) => None,
            SessionStorageKind::Jsonl(storage) => Some(storage.metadata().clone()),
        }
    }

    pub fn get_entry(&self, id: &str) -> Option<SessionTreeEntry> {
        self.storage.lock().get_entry(id).cloned()
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        self.storage.lock().entries()
    }

    pub fn branch(&self, from_id: Option<&str>) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let owned_leaf;
        let leaf_id = match from_id {
            Some(id) => Some(id),
            None => {
                owned_leaf = self.storage.lock().leaf_id()?;
                owned_leaf.as_deref()
            }
        };
        self.storage.lock().path_to_root(leaf_id)
    }

    pub fn build_context(&self) -> Result<SessionContext, SessionError> {
        build_session_context(&self.branch(None)?)
    }

    pub fn label(&self, id: &str) -> Option<String> {
        self.storage.lock().label(id).map(str::to_owned)
    }

    pub fn session_name(&self) -> Option<String> {
        self.storage
            .lock()
            .find_entries("session_info")
            .into_iter()
            .filter_map(|entry| match entry {
                SessionTreeEntry::SessionInfo { name, .. } => name,
                _ => None,
            })
            .last()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty())
    }

    pub fn append_message(&mut self, message: Message) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::Message {
            id,
            parent_id,
            timestamp: now_iso(),
            message: message.into(),
        })
    }

    pub fn append_bash_execution(
        &mut self,
        message: BashExecutionMessage,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::Message {
            id,
            parent_id,
            timestamp: now_iso(),
            message: message.into(),
        })
    }

    pub fn append_thinking_level_change(
        &mut self,
        thinking_level: impl Into<String>,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::ThinkingLevelChange {
            id,
            parent_id,
            timestamp: now_iso(),
            thinking_level: thinking_level.into(),
        })
    }

    pub fn append_model_change(
        &mut self,
        provider: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::ModelChange {
            id,
            parent_id,
            timestamp: now_iso(),
            provider: provider.into(),
            model_id: model_id.into(),
        })
    }

    pub fn append_compaction(
        &mut self,
        summary: impl Into<String>,
        first_kept_entry_id: impl Into<String>,
        tokens_before: u64,
    ) -> Result<String, SessionError> {
        self.append_compaction_with_details(summary, first_kept_entry_id, tokens_before, None, None)
    }

    pub fn append_compaction_with_details(
        &mut self,
        summary: impl Into<String>,
        first_kept_entry_id: impl Into<String>,
        tokens_before: u64,
        details: Option<Value>,
        from_hook: Option<bool>,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::Compaction {
            id,
            parent_id,
            timestamp: now_iso(),
            summary: summary.into(),
            first_kept_entry_id: first_kept_entry_id.into(),
            tokens_before,
            details,
            from_hook,
        })
    }

    pub fn append_custom_entry(
        &mut self,
        custom_type: impl Into<String>,
        data: Option<Value>,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::Custom {
            id,
            parent_id,
            timestamp: now_iso(),
            custom_type: custom_type.into(),
            data,
        })
    }

    pub fn append_custom_message_entry(
        &mut self,
        custom_type: impl Into<String>,
        content: CustomMessageContent,
        display: bool,
        details: Option<Value>,
    ) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::CustomMessage {
            id,
            parent_id,
            timestamp: now_iso(),
            custom_type: custom_type.into(),
            content,
            display,
            details,
        })
    }

    pub fn append_label(
        &mut self,
        target_id: impl Into<String>,
        label: Option<String>,
    ) -> Result<String, SessionError> {
        let target_id = target_id.into();
        if self.storage.lock().get_entry(&target_id).is_none() {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {target_id} not found"),
            ));
        }
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::Label {
            id,
            parent_id,
            timestamp: now_iso(),
            target_id,
            label,
        })
    }

    pub fn append_session_name(&mut self, name: impl Into<String>) -> Result<String, SessionError> {
        let id = self.storage.lock().create_entry_id();
        let parent_id = self.storage.lock().leaf_id()?;
        self.append_entry(SessionTreeEntry::SessionInfo {
            id,
            parent_id,
            timestamp: now_iso(),
            name: Some(name.into().trim().to_owned()),
        })
    }

    pub fn move_to(
        &mut self,
        entry_id: Option<String>,
        summary: Option<BranchMoveSummary>,
    ) -> Result<Option<String>, SessionError> {
        if let Some(entry_id) = &entry_id {
            if self.storage.lock().get_entry(entry_id).is_none() {
                return Err(SessionError::new(
                    SessionErrorCode::NotFound,
                    format!("Entry {entry_id} not found"),
                ));
            }
        }
        self.storage.lock().set_leaf_id(entry_id.clone())?;
        let Some(summary) = summary else {
            return Ok(None);
        };
        let id = self.storage.lock().create_entry_id();
        let from_id = entry_id.unwrap_or_else(|| "root".to_owned());
        self.append_entry(SessionTreeEntry::BranchSummary {
            id,
            parent_id: (from_id != "root").then_some(from_id.clone()),
            timestamp: now_iso(),
            from_id,
            summary: summary.summary,
            details: summary.details,
            from_hook: summary.from_hook,
        })
        .map(Some)
    }

    fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<String, SessionError> {
        let id = entry.id().to_owned();
        self.storage.lock().append_entry(entry)?;
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchMoveSummary {
    pub summary: String,
    pub details: Option<Value>,
    pub from_hook: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct InMemorySessionRepo {
    sessions: BTreeMap<String, Session>,
}

impl InMemorySessionRepo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&mut self, id: Option<String>) -> Session {
        let metadata = SessionMetadata {
            id: id.unwrap_or_else(uuidv7),
            created_at: now_iso(),
        };
        let session = Session::new(InMemorySessionStorage::with_options(
            None,
            Some(metadata.clone()),
        ));
        self.sessions.insert(metadata.id, session.clone());
        session
    }

    pub fn open(&self, metadata: &SessionMetadata) -> Result<Session, SessionError> {
        self.sessions.get(&metadata.id).cloned().ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {}", metadata.id),
            )
        })
    }

    pub fn list(&self) -> Vec<SessionMetadata> {
        self.sessions
            .values()
            .filter_map(|session| session.in_memory_metadata())
            .collect()
    }

    pub fn delete(&mut self, metadata: &SessionMetadata) {
        self.sessions.remove(&metadata.id);
    }

    pub fn fork(
        &mut self,
        source_metadata: &SessionMetadata,
        options: SessionForkOptions,
    ) -> Result<Session, SessionError> {
        let source = self.open(source_metadata)?;
        let entries = entries_to_fork(&source.storage.lock(), &options)?;
        let metadata = SessionMetadata {
            id: options.id.unwrap_or_else(uuidv7),
            created_at: now_iso(),
        };
        let session = Session::new(InMemorySessionStorage::with_options(
            Some(entries),
            Some(metadata.clone()),
        ));
        self.sessions.insert(metadata.id, session.clone());
        Ok(session)
    }
}

#[derive(Debug, Clone)]
pub struct JsonlSessionRepo {
    sessions_root: PathBuf,
}

impl JsonlSessionRepo {
    pub fn new(sessions_root: impl AsRef<Path>) -> Self {
        Self {
            sessions_root: sessions_root.as_ref().to_path_buf(),
        }
    }

    pub fn create(
        &self,
        cwd: impl Into<String>,
        id: Option<String>,
        parent_session_path: Option<String>,
    ) -> Result<Session, SessionError> {
        let cwd = cwd.into();
        let id = id.unwrap_or_else(uuidv7);
        let created_at = now_iso();
        let dir = self.session_dir(&cwd);
        fs::create_dir_all(&dir).map_err(storage_error)?;
        let file_path = dir.join(format!("{}_{}.jsonl", sanitize_timestamp(&created_at), id));
        let storage = JsonlSessionStorage::create(file_path, cwd, id, parent_session_path)?;
        Ok(Session::new(storage))
    }

    pub fn open(&self, metadata: &JsonlSessionMetadata) -> Result<Session, SessionError> {
        if !Path::new(&metadata.path).exists() {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {}", metadata.path),
            ));
        }
        Ok(Session::new(JsonlSessionStorage::open(&metadata.path)?))
    }

    pub fn list(&self, cwd: Option<&str>) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
        let dirs = if let Some(cwd) = cwd {
            vec![self.session_dir(cwd)]
        } else {
            self.session_dirs()?
        };
        let mut sessions = Vec::new();
        for dir in dirs {
            if !dir.exists() {
                continue;
            }
            for entry in fs::read_dir(&dir).map_err(storage_error)? {
                let entry = entry.map_err(storage_error)?;
                if entry.file_type().map_err(storage_error)?.is_dir() {
                    continue;
                }
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                    continue;
                }
                match load_jsonl_session_metadata(&path) {
                    Ok(metadata) => sessions.push(metadata),
                    Err(error) if error.code == SessionErrorCode::InvalidSession => {}
                    Err(error) => return Err(error),
                }
            }
        }
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    pub fn delete(&self, metadata: &JsonlSessionMetadata) -> Result<(), SessionError> {
        match fs::remove_file(&metadata.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(storage_error(error)),
        }
    }

    pub fn fork(
        &self,
        source_metadata: &JsonlSessionMetadata,
        options: JsonlSessionForkOptions,
    ) -> Result<Session, SessionError> {
        let source = self.open(source_metadata)?;
        let entries = entries_to_fork(&source.storage.lock(), &options.fork)?;
        let mut session = self.create(
            options.cwd,
            options.fork.id,
            Some(
                options
                    .parent_session_path
                    .unwrap_or_else(|| source_metadata.path.clone()),
            ),
        )?;
        for entry in entries {
            session.append_entry(entry)?;
        }
        Ok(session)
    }

    pub fn encoded_cwd(cwd: &str) -> String {
        format!(
            "--{}--",
            cwd.trim_start_matches(['/', '\\'])
                .replace(['/', '\\', ':'], "-")
        )
    }

    fn session_dir(&self, cwd: &str) -> PathBuf {
        self.sessions_root.join(Self::encoded_cwd(cwd))
    }

    fn session_dirs(&self) -> Result<Vec<PathBuf>, SessionError> {
        if !self.sessions_root.exists() {
            return Ok(Vec::new());
        }
        let mut dirs = Vec::new();
        for entry in fs::read_dir(&self.sessions_root).map_err(storage_error)? {
            let entry = entry.map_err(storage_error)?;
            if entry.file_type().map_err(storage_error)?.is_dir() {
                dirs.push(entry.path());
            }
        }
        Ok(dirs)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionForkOptions {
    pub entry_id: Option<String>,
    pub position: ForkPosition,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForkPosition {
    #[default]
    Before,
    At,
}

#[derive(Debug, Clone)]
pub struct JsonlSessionForkOptions {
    pub cwd: String,
    pub parent_session_path: Option<String>,
    pub fork: SessionForkOptions,
}

fn entries_to_fork(
    storage: &SessionStorageKind,
    options: &SessionForkOptions,
) -> Result<Vec<SessionTreeEntry>, SessionError> {
    let Some(entry_id) = &options.entry_id else {
        return Ok(storage.entries());
    };
    let target = storage.get_entry(entry_id).ok_or_else(|| {
        SessionError::new(
            SessionErrorCode::InvalidForkTarget,
            format!("Entry {entry_id} not found"),
        )
    })?;
    let effective_leaf_id = match options.position {
        ForkPosition::At => Some(target.id().to_owned()),
        ForkPosition::Before => match target {
            SessionTreeEntry::Message {
                message: SessionEntryMessage::Llm(Message::User(_)),
                ..
            } => target.parent_id().map(str::to_owned),
            _ => {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidForkTarget,
                    format!("Entry {entry_id} is not a user message"),
                ));
            }
        },
    };
    storage.path_to_root(effective_leaf_id.as_deref())
}

pub fn build_session_context(
    path_entries: &[SessionTreeEntry],
) -> Result<SessionContext, SessionError> {
    let mut thinking_level = "off".to_owned();
    let mut model = None;
    let mut compaction: Option<&SessionTreeEntry> = None;

    for entry in path_entries {
        match entry {
            SessionTreeEntry::ThinkingLevelChange {
                thinking_level: next,
                ..
            } => thinking_level = next.clone(),
            SessionTreeEntry::ModelChange {
                provider, model_id, ..
            } => {
                model = Some(SessionModelSelection {
                    provider: provider.clone(),
                    model_id: model_id.clone(),
                });
            }
            SessionTreeEntry::Message {
                message: SessionEntryMessage::Llm(Message::Assistant(message)),
                ..
            } => {
                model = Some(SessionModelSelection {
                    provider: message.provider.clone(),
                    model_id: message.model.clone(),
                });
            }
            entry @ SessionTreeEntry::Compaction { .. } => compaction = Some(entry),
            _ => {}
        }
    }

    let mut messages = Vec::new();
    if let Some(SessionTreeEntry::Compaction {
        id,
        summary,
        first_kept_entry_id,
        tokens_before,
        timestamp,
        ..
    }) = compaction
    {
        messages.push(SessionMessage::CompactionSummary {
            summary: summary.clone(),
            tokens_before: *tokens_before,
            timestamp: parse_timestamp_millis(timestamp),
        });
        let compaction_idx = path_entries
            .iter()
            .position(|entry| entry.id() == id)
            .ok_or_else(|| {
                SessionError::new(SessionErrorCode::InvalidSession, "Compaction entry missing")
            })?;
        let mut found_first_kept = false;
        for entry in &path_entries[..compaction_idx] {
            if entry.id() == first_kept_entry_id {
                found_first_kept = true;
            }
            if found_first_kept {
                append_context_message(&mut messages, entry);
            }
        }
        for entry in &path_entries[compaction_idx + 1..] {
            append_context_message(&mut messages, entry);
        }
    } else {
        for entry in path_entries {
            append_context_message(&mut messages, entry);
        }
    }

    Ok(SessionContext {
        messages,
        thinking_level,
        model,
    })
}

pub fn load_jsonl_session_metadata(
    file_path: impl AsRef<Path>,
) -> Result<JsonlSessionMetadata, SessionError> {
    let file_path = file_path.as_ref();
    let file = fs::File::open(file_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session file {} not found", file_path.display()),
            )
        } else {
            storage_error(error)
        }
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).map_err(storage_error)?;
        if bytes_read == 0 {
            return Err(invalid_session(file_path, "missing session header"));
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let header = parse_header_line(line.trim_end_matches(['\r', '\n']), file_path)?;
    Ok(JsonlSessionMetadata {
        id: header.id,
        created_at: header.timestamp,
        cwd: header.cwd,
        path: file_path.to_string_lossy().into_owned(),
        parent_session_path: header.parent_session,
    })
}

pub fn uuidv7() -> String {
    Uuid::now_v7().to_string()
}

pub fn user_message_text(text: impl Into<String>) -> Message {
    Message::User(ri_llm_provider::UserMessage::text(text.into()))
}

pub fn assistant_message_text(text: impl Into<String>) -> Message {
    Message::Assistant(ri_llm_provider::AssistantMessage {
        content: vec![ri_llm_provider::AssistantContent::Text(
            ri_llm_provider::TextContent::new(text),
        )],
        api: "test".to_owned(),
        provider: "test".to_owned(),
        model: "test-model".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: ri_llm_provider::Usage::zero(),
        stop_reason: ri_llm_provider::StopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    })
}

pub fn custom_text_content(text: impl Into<String>) -> CustomMessageContent {
    CustomMessageContent::Text(text.into())
}

pub fn bash_execution_to_text(message: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", message.command);
    if message.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str("```\n");
        text.push_str(&message.output);
        text.push_str("\n```");
    }
    if message.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(exit_code) = message.exit_code
        && exit_code != 0
    {
        text.push_str(&format!("\n\nCommand exited with code {exit_code}"));
    }
    if message.truncated
        && let Some(path) = &message.full_output_path
    {
        text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
    }
    text
}

fn append_context_message(messages: &mut Vec<SessionMessage>, entry: &SessionTreeEntry) {
    match entry {
        SessionTreeEntry::Message { message, .. } => match message {
            SessionEntryMessage::Llm(message) => messages.push(message.clone().into()),
            SessionEntryMessage::BashExecution(message) => messages.push(message.clone().into()),
        },
        SessionTreeEntry::CustomMessage {
            custom_type,
            content,
            display,
            details,
            timestamp,
            ..
        } => messages.push(SessionMessage::Custom {
            custom_type: custom_type.clone(),
            content: content.clone(),
            display: *display,
            details: details.clone(),
            timestamp: parse_timestamp_millis(timestamp),
        }),
        SessionTreeEntry::BranchSummary {
            summary,
            from_id,
            timestamp,
            ..
        } if !summary.is_empty() => messages.push(SessionMessage::BranchSummary {
            summary: summary.clone(),
            from_id: from_id.clone(),
            timestamp: parse_timestamp_millis(timestamp),
        }),
        _ => {}
    }
}

fn leaf_id_after_entry(entry: &SessionTreeEntry) -> Option<String> {
    match entry {
        SessionTreeEntry::Leaf { target_id, .. } => target_id.clone(),
        _ => Some(entry.id().to_owned()),
    }
}

fn update_label_cache(labels_by_id: &mut BTreeMap<String, String>, entry: &SessionTreeEntry) {
    let SessionTreeEntry::Label {
        target_id, label, ..
    } = entry
    else {
        return;
    };
    let trimmed = label
        .as_deref()
        .map(str::trim)
        .filter(|label| !label.is_empty());
    if let Some(label) = trimmed {
        labels_by_id.insert(target_id.clone(), label.to_owned());
    } else {
        labels_by_id.remove(target_id);
    }
}

fn build_labels_by_id(entries: &[SessionTreeEntry]) -> BTreeMap<String, String> {
    let mut labels = BTreeMap::new();
    for entry in entries {
        update_label_cache(&mut labels, entry);
    }
    labels
}

fn generate_entry_id(by_id: &BTreeMap<String, SessionTreeEntry>) -> String {
    for _ in 0..100 {
        let id = uuidv7().chars().take(8).collect::<String>();
        if !by_id.contains_key(&id) {
            return id;
        }
    }
    uuidv7()
}

fn validate_leaf(
    by_id: &BTreeMap<String, SessionTreeEntry>,
    leaf_id: Option<&str>,
) -> Result<(), SessionError> {
    if let Some(leaf_id) = leaf_id {
        if !by_id.contains_key(leaf_id) {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {leaf_id} not found"),
            ));
        }
    }
    Ok(())
}

fn validate_existing_leaf(
    by_id: &BTreeMap<String, SessionTreeEntry>,
    leaf_id: Option<&str>,
) -> Result<(), SessionError> {
    if let Some(leaf_id) = leaf_id {
        if !by_id.contains_key(leaf_id) {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {leaf_id} not found"),
            ));
        }
    }
    Ok(())
}

fn get_path_to_root(
    by_id: &BTreeMap<String, SessionTreeEntry>,
    leaf_id: Option<&str>,
) -> Result<Vec<SessionTreeEntry>, SessionError> {
    let Some(leaf_id) = leaf_id else {
        return Ok(Vec::new());
    };
    let mut path = Vec::new();
    let mut current = by_id.get(leaf_id).ok_or_else(|| {
        SessionError::new(
            SessionErrorCode::NotFound,
            format!("Entry {leaf_id} not found"),
        )
    })?;
    loop {
        path.push(current.clone());
        let Some(parent_id) = current.parent_id() else {
            break;
        };
        current = by_id.get(parent_id).ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {parent_id} not found"),
            )
        })?;
    }
    path.reverse();
    Ok(path)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionHeader {
    #[serde(rename = "type")]
    entry_type: String,
    version: u8,
    id: String,
    timestamp: String,
    cwd: String,
    #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none")]
    parent_session: Option<String>,
}

fn load_jsonl_storage(
    file_path: &Path,
) -> Result<(SessionHeader, Vec<SessionTreeEntry>, Option<String>), SessionError> {
    let content = fs::read_to_string(file_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session file {} not found", file_path.display()),
            )
        } else {
            storage_error(error)
        }
    })?;
    let lines: Vec<&str> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err(invalid_session(file_path, "missing session header"));
    }
    let header = parse_header_line(lines[0], file_path)?;
    let mut entries = Vec::new();
    let mut leaf_id = None;
    for (index, line) in lines.iter().enumerate().skip(1) {
        let entry = parse_entry_line(line, file_path, index + 1)?;
        leaf_id = leaf_id_after_entry(&entry);
        entries.push(entry);
    }
    Ok((header, entries, leaf_id))
}

fn parse_header_line(line: &str, file_path: &Path) -> Result<SessionHeader, SessionError> {
    let parsed: Value = serde_json::from_str(line).map_err(|error| {
        invalid_session(file_path, "first line is not a valid session header").with_cause(error)
    })?;
    if parsed.get("type").and_then(Value::as_str) != Some("session") {
        return Err(invalid_session(
            file_path,
            "first line is not a valid session header",
        ));
    }
    if parsed.get("version").and_then(Value::as_u64) != Some(3) {
        return Err(invalid_session(file_path, "unsupported session version"));
    }
    let header: SessionHeader = serde_json::from_value(parsed).map_err(|error| {
        invalid_session(file_path, "first line is not a valid session header").with_cause(error)
    })?;
    if header.id.is_empty() {
        return Err(invalid_session(file_path, "session header is missing id"));
    }
    if header.timestamp.is_empty() {
        return Err(invalid_session(
            file_path,
            "session header is missing timestamp",
        ));
    }
    if header.cwd.is_empty() {
        return Err(invalid_session(file_path, "session header is missing cwd"));
    }
    Ok(header)
}

fn parse_entry_line(
    line: &str,
    file_path: &Path,
    line_number: usize,
) -> Result<SessionTreeEntry, SessionError> {
    let parsed: Value = serde_json::from_str(line).map_err(|error| {
        invalid_entry(file_path, line_number, "is not valid JSON").with_cause(error)
    })?;
    let entry_type = parsed
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_entry(file_path, line_number, "is missing entry type"))?;
    if !parsed
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.is_empty())
    {
        return Err(invalid_entry(file_path, line_number, "is missing entry id"));
    }
    if !parsed
        .get("parentId")
        .is_some_and(|parent_id| parent_id.is_null() || parent_id.is_string())
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "has invalid parentId",
        ));
    }
    if !parsed
        .get("timestamp")
        .and_then(Value::as_str)
        .is_some_and(|timestamp| !timestamp.is_empty())
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "is missing timestamp",
        ));
    }
    if entry_type == "leaf"
        && !parsed
            .get("targetId")
            .is_some_and(|target_id| target_id.is_null() || target_id.is_string())
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "has invalid targetId",
        ));
    }
    let entry: SessionTreeEntry = serde_json::from_value(parsed).map_err(|error| {
        invalid_entry(file_path, line_number, "is not a valid session entry").with_cause(error)
    })?;
    Ok(entry)
}

fn invalid_session(file_path: &Path, message: &str) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidSession,
        format!(
            "Invalid JSONL session file {}: {message}",
            file_path.display()
        ),
    )
}

fn invalid_entry(file_path: &Path, line_number: usize, message: &str) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidEntry,
        format!(
            "Invalid JSONL session file {}: line {line_number} {message}",
            file_path.display()
        ),
    )
}

trait WithCause {
    fn with_cause(self, cause: impl Error) -> Self;
}

impl WithCause for SessionError {
    fn with_cause(mut self, cause: impl Error) -> Self {
        self.message = format!("{}: {cause}", self.message);
        self
    }
}

fn storage_error(error: impl Display) -> SessionError {
    SessionError::new(SessionErrorCode::Storage, error.to_string())
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn sanitize_timestamp(timestamp: &str) -> String {
    timestamp.replace([':', '.'], "-")
}

fn parse_timestamp_millis(timestamp: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|value| value.timestamp_millis())
        .unwrap_or_else(|_| now_millis())
}
