use async_trait::async_trait;
use ri_agent_core::*;
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

struct CalculateExecutor;

#[async_trait]
impl AgentToolExecutor for CalculateExecutor {
    async fn execute(&self, _tool_call_id: &str, params: Value) -> Result<AgentToolResult, String> {
        let expression = params
            .get("expression")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(AgentToolResult::text(format!("calculated: {expression}")))
    }
}

fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(&user.content),
            _ => None,
        })
        .flat_map(|content| match content {
            UserContentValue::Plain(text) => vec![text.clone()],
            UserContentValue::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect(),
        })
        .collect()
}

fn session_user_texts(session: &Session) -> Vec<String> {
    session
        .entries()
        .into_iter()
        .filter_map(|entry| match entry {
            SessionTreeEntry::Message { message, .. } => Some(message),
            _ => None,
        })
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content),
            _ => None,
        })
        .flat_map(|content| match content {
            UserContentValue::Plain(text) => vec![text],
            UserContentValue::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|block| match block {
                    UserContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect(),
        })
        .collect()
}

fn assistant_text(message: &AssistantMessage) -> Option<&str> {
    match message.content.first()? {
        AssistantContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    }
}

fn test_env() -> LocalExecutionEnv {
    LocalExecutionEnv::new("/tmp")
}

fn temp_harness_session_path(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-harness-{label}-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp harness dir");
    dir.join("session.jsonl")
}

fn calculate_tool() -> AgentTool {
    AgentTool {
        definition: Tool {
            name: "calculate".to_owned(),
            description: "Evaluate an expression".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string" }
                },
                "required": ["expression"]
            }),
        },
        label: "Calculate".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(CalculateExecutor),
    }
}

fn clock_tool() -> AgentTool {
    AgentTool {
        definition: Tool {
            name: "clock".to_owned(),
            description: "Read the clock".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        label: "Clock".to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor: Arc::new(CalculateExecutor),
    }
}

#[test]
fn agent_harness_constructs_and_exposes_queue_modes() {
    let model = Model::faux("harness-api", "harness-provider", "harness-model");
    let session = Session::new(InMemorySessionStorage::new());
    let mut options = AgentHarnessOptions::new(test_env(), session.clone(), model.clone());
    options.thinking_level = ThinkingLevel::High;
    options.system_prompt = "You are helpful.".to_owned();
    options.steering_mode = QueueMode::All;
    options.follow_up_mode = QueueMode::All;

    let harness = AgentHarness::new(options);

    assert_eq!(harness.get_model(), model);
    assert_eq!(harness.get_thinking_level(), ThinkingLevel::High);
    assert_eq!(harness.get_steering_mode(), QueueMode::All);
    assert_eq!(harness.get_follow_up_mode(), QueueMode::All);
    harness.set_steering_mode(QueueMode::OneAtATime);
    harness.set_follow_up_mode(QueueMode::OneAtATime);
    assert_eq!(harness.get_steering_mode(), QueueMode::OneAtATime);
    assert_eq!(harness.get_follow_up_mode(), QueueMode::OneAtATime);
    assert_eq!(harness.session().metadata_id(), session.metadata_id());
}

#[test]
fn agent_harness_resources_getters_clone_and_emit_update_events() {
    let model = Model::faux("resources-api", "resources-provider", "resources-model");
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        model,
    ));
    let updates = Arc::new(Mutex::new(Vec::new()));
    let updates_ref = updates.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::ResourcesUpdate(update) = event {
            updates_ref.lock().expect("mutex").push((
                update
                    .resources
                    .skills
                    .first()
                    .and_then(|skill| skill.source.clone()),
                update
                    .previous_resources
                    .skills
                    .first()
                    .and_then(|skill| skill.source.clone()),
            ));
        }
    });
    let resources = AgentHarnessResources {
        skills: vec![Skill {
            name: "inspect".to_owned(),
            description: "Inspect things".to_owned(),
            content: "Use inspection tools.".to_owned(),
            file_path: "/skills/inspect/SKILL.md".to_owned(),
            source: Some("project".to_owned()),
            disable_model_invocation: false,
        }],
        prompt_templates: vec![PromptTemplate {
            name: "review".to_owned(),
            description: "Review".to_owned(),
            content: "Review $1".to_owned(),
            source: Some("user".to_owned()),
        }],
    };

    harness.set_resources(resources.clone());
    harness.set_resources(resources.clone());
    let resolved = harness.get_resources();

    assert_eq!(resolved, resources);
    assert_eq!(
        *updates.lock().expect("mutex"),
        vec![
            (Some("project".to_owned()), None),
            (Some("project".to_owned()), Some("project".to_owned()))
        ]
    );
    assert_eq!(
        resolved
            .prompt_templates
            .first()
            .and_then(|template| template.source.as_deref()),
        Some("user")
    );
}

#[test]
fn agent_harness_appends_custom_messages_labels_and_session_name() {
    let model = Model::faux("session-api", "session-provider", "session-model");
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(test_env(), session.clone(), model));

    harness
        .append_message(Message::User(UserMessage::text("checkpoint text")))
        .expect("append message");
    let user_id = session
        .entries()
        .first()
        .expect("user entry")
        .id()
        .to_owned();
    harness
        .append_custom_entry("custom", Some(json!({ "kind": "metadata" })))
        .expect("append custom entry");
    harness
        .append_custom_message(
            "listener",
            CustomMessageContent::Text("custom write".to_owned()),
            true,
            Some(json!({ "ok": true })),
        )
        .expect("append custom message");
    harness
        .append_label(user_id.clone(), Some("checkpoint".to_owned()))
        .expect("append label");
    harness
        .append_session_name(" named session ")
        .expect("append session name");
    let missing_label = harness
        .append_label("missing", Some("checkpoint".to_owned()))
        .expect_err("missing label target");

    assert_eq!(missing_label.code, AgentHarnessErrorCode::Session);
    assert_eq!(session.label(&user_id).as_deref(), Some("checkpoint"));
    assert_eq!(session.session_name().as_deref(), Some("named session"));
    let entries = session.entries();
    assert!(matches!(
        entries.as_slice(),
        [
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            },
            SessionTreeEntry::Custom {
                custom_type: generic_custom_type,
                data: Some(data),
                ..
            },
            SessionTreeEntry::CustomMessage {
                custom_type,
                content: CustomMessageContent::Text(text),
                display: true,
                details: Some(details),
                ..
            },
            SessionTreeEntry::Label { label: Some(label), .. },
            SessionTreeEntry::SessionInfo { name: Some(name), .. },
        ] if generic_custom_type == "custom"
            && data == &json!({ "kind": "metadata" })
            && custom_type == "listener"
            && text == "custom write"
            && details == &json!({ "ok": true })
            && label == "checkpoint"
            && name == "named session"
    ));
}

#[tokio::test]
async fn agent_harness_runs_skills_and_prompt_templates_from_resources() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let requests = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let first_request = requests.clone();
    let second_request = requests.clone();
    registration.set_responses(vec![
        faux_response_factory(move |context, _, _, _| {
            first_request
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages));
            faux_assistant_message("skill", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            second_request
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages));
            faux_assistant_message("template", Default::default())
        }),
    ]);
    let skill = Skill {
        name: "inspect".to_owned(),
        description: "Inspect things".to_owned(),
        content: "Use inspection tools.".to_owned(),
        file_path: "/skills/inspect/SKILL.md".to_owned(),
        source: None,
        disable_model_invocation: false,
    };
    let template = PromptTemplate {
        name: "review".to_owned(),
        description: "Review".to_owned(),
        content: "Review $1 with $2".to_owned(),
        source: None,
    };
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.resources = AgentHarnessResources {
        skills: vec![skill.clone()],
        prompt_templates: vec![template.clone()],
    };
    let harness = AgentHarness::new(options);

    let missing_skill = harness
        .skill("missing", None)
        .await
        .expect_err("missing skill");
    assert_eq!(missing_skill.code, AgentHarnessErrorCode::InvalidArgument);
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    let missing_template = harness
        .prompt_from_template("missing", &[])
        .await
        .expect_err("missing template");
    assert_eq!(
        missing_template.code,
        AgentHarnessErrorCode::InvalidArgument
    );
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);

    harness
        .skill("inspect", Some("Check errors."))
        .await
        .expect("skill");
    let template_args = vec!["src/main.rs".to_owned(), "carefully".to_owned()];
    harness
        .prompt_from_template("review", &template_args)
        .await
        .expect("template");

    let skill_prompt = format_skill_invocation(&skill, Some("Check errors."));
    let template_prompt = format_prompt_template_invocation(&template, &template_args);
    assert_eq!(
        *requests.lock().expect("mutex"),
        vec![
            vec![skill_prompt.clone()],
            vec![skill_prompt, template_prompt]
        ]
    );
    registration.unregister();
}

#[test]
fn agent_harness_model_and_thinking_setters_emit_selection_events() {
    let initial_model = Model::faux("select-api", "select-provider", "first");
    let next_model = Model::faux("select-api", "select-provider", "second");
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        initial_model.clone(),
    ));
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::ModelSelect(event) => events_ref.lock().expect("mutex").push(format!(
            "model:{}->{}",
            event.previous_model.id, event.model.id
        )),
        AgentHarnessEvent::ThinkingLevelSelect(event) => {
            events_ref.lock().expect("mutex").push(format!(
                "thinking:{:?}->{:?}",
                event.previous_level, event.level
            ));
        }
        _ => {}
    });

    harness.set_model(next_model.clone()).expect("set model");
    harness
        .set_thinking_level(ThinkingLevel::High)
        .expect("set thinking level");

    assert_eq!(harness.get_model(), next_model);
    assert_eq!(harness.get_thinking_level(), ThinkingLevel::High);
    let context = session.build_context().expect("context");
    assert_eq!(context.thinking_level, "high");
    assert_eq!(
        context.model,
        Some(SessionModelSelection {
            provider: "select-provider".to_owned(),
            model_id: "second".to_owned(),
        })
    );
    assert_eq!(
        *events.lock().expect("mutex"),
        vec![
            "model:first->second".to_owned(),
            "thinking:Off->High".to_owned()
        ]
    );
}

#[tokio::test]
async fn agent_harness_queues_model_and_thinking_session_writes_during_turn() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));

    let running = harness.clone();
    let prompt_task = tokio::spawn(async move { running.prompt("hello").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness
        .set_model(Model::faux(
            "queued-select-api",
            "queued-select-provider",
            "queued-model",
        ))
        .expect("set model");
    harness
        .set_thinking_level(ThinkingLevel::Low)
        .expect("set thinking");
    prompt_task.await.expect("prompt task").expect("prompt");

    let context = session.build_context().expect("context");
    assert_eq!(context.thinking_level, "low");
    assert_eq!(
        context.model,
        Some(SessionModelSelection {
            provider: "queued-select-provider".to_owned(),
            model_id: "queued-model".to_owned(),
        })
    );
    let entries = session.entries();
    assert!(matches!(
        entries.as_slice(),
        [
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            },
            SessionTreeEntry::Message {
                message: Message::Assistant(_),
                ..
            },
            SessionTreeEntry::ModelChange { .. },
            SessionTreeEntry::ThinkingLevelChange { .. }
        ]
    ));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_orders_pending_append_message_after_agent_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let wrote_listener_message = Arc::new(Mutex::new(false));
    let save_points = Arc::new(Mutex::new(Vec::new()));
    let listener_harness = harness.clone();
    let wrote_listener_message_ref = wrote_listener_message.clone();
    let save_points_ref = save_points.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::Agent(AgentEvent::MessageEnd {
            message: AgentMessage::Assistant(_),
        }) = event
        {
            let mut wrote = wrote_listener_message_ref.lock().expect("mutex");
            if !*wrote {
                *wrote = true;
                listener_harness
                    .append_message(Message::User(UserMessage::text("listener write")))
                    .expect("append message");
            }
        }
        if let AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } = event
        {
            save_points_ref
                .lock()
                .expect("mutex")
                .push(*had_pending_mutations);
        }
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        session_user_texts(&session),
        vec!["hello".to_owned(), "listener write".to_owned()]
    );
    assert_eq!(*save_points.lock().expect("mutex"), vec![true]);
    let entries = session.entries();
    assert!(matches!(
        entries.as_slice(),
        [
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            },
            SessionTreeEntry::Message {
                message: Message::Assistant(_),
                ..
            },
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            }
        ]
    ));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_stream_options_getter_setter_clone_and_flow_to_provider() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_options_ref = seen_options.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        seen_options_ref
            .lock()
            .expect("mutex")
            .push(options.clone());
        faux_assistant_message("ok", Default::default())
    })]);
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    ));
    let mut stream_options = SimpleStreamOptions::default();
    stream_options
        .stream
        .headers
        .insert("x-harness-test".to_owned(), "configured".to_owned());
    stream_options.stream.max_retries = Some(3);
    stream_options.stream.session_id = Some("session-123".to_owned());

    harness.set_stream_options(stream_options.clone());
    let mut snapshot = harness.get_stream_options();
    snapshot
        .stream
        .headers
        .insert("x-harness-test".to_owned(), "mutated snapshot".to_owned());

    assert_eq!(harness.get_stream_options(), stream_options);
    harness.prompt("hello").await.expect("prompt");

    let seen = seen_options.lock().expect("mutex").clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].stream.headers.get("x-harness-test"),
        Some(&"configured".to_owned())
    );
    assert_eq!(seen[0].stream.max_retries, Some(3));
    assert_eq!(seen[0].stream.session_id.as_deref(), Some("session-123"));
    assert!(seen[0].stream.abort_flag.is_some());
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_provider_request_hooks_merge_auth_and_patch_stream_options() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_options_ref = seen_options.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        seen_options_ref
            .lock()
            .expect("mutex")
            .push(options.clone());
        faux_assistant_message("ok", Default::default())
    })]);

    let session = Session::new(InMemorySessionStorage::with_options(
        None,
        Some(SessionMetadata {
            id: "session-1".to_owned(),
            created_at: "now".to_owned(),
        }),
    ));
    let mut options = AgentHarnessOptions::new(test_env(), session, registration.get_model());
    options.get_api_key_and_headers = Some(Arc::new(|_| {
        Ok(ProviderAuth {
            api_key: Some("secret".to_owned()),
            headers: BTreeMap::from([("x-auth".to_owned(), "auth".to_owned())]),
        })
    }));
    options.stream_options.stream.timeout_ms = Some(1000);
    options.stream_options.stream.max_retries = Some(2);
    options.stream_options.stream.max_retry_delay_ms = Some(3000);
    options.stream_options.stream.cache_retention = Some(CacheRetention::None);
    options
        .stream_options
        .stream
        .headers
        .insert("x-base".to_owned(), "base".to_owned());
    options
        .stream_options
        .stream
        .metadata
        .insert("base".to_owned(), json!(true));

    let harness = AgentHarness::new(options);
    harness.on_before_provider_request(|event| {
        assert_eq!(event.session_id, "session-1");
        assert_eq!(
            event.stream_options.stream.headers,
            BTreeMap::from([
                ("x-auth".to_owned(), "auth".to_owned()),
                ("x-base".to_owned(), "base".to_owned())
            ])
        );
        Ok(Some(BeforeProviderRequestResult {
            stream_options: Some(AgentHarnessStreamOptionsPatch {
                headers: Some(HeaderMapPatch::Merge(BTreeMap::from([(
                    "x-hook".to_owned(),
                    Some("hook".to_owned()),
                )]))),
                metadata: Some(MetadataMapPatch::Merge(BTreeMap::from([(
                    "hook".to_owned(),
                    Some(json!(true)),
                )]))),
                ..Default::default()
            }),
        }))
    });

    harness.prompt("hello").await.expect("prompt");

    let seen = seen_options.lock().expect("mutex").clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].stream.api_key.as_deref(), Some("secret"));
    assert_eq!(seen[0].stream.session_id.as_deref(), Some("session-1"));
    assert_eq!(seen[0].stream.timeout_ms, Some(1000));
    assert_eq!(seen[0].stream.max_retries, Some(2));
    assert_eq!(seen[0].stream.max_retry_delay_ms, Some(3000));
    assert_eq!(seen[0].stream.cache_retention, Some(CacheRetention::None));
    assert_eq!(
        seen[0].stream.headers,
        BTreeMap::from([
            ("x-auth".to_owned(), "auth".to_owned()),
            ("x-base".to_owned(), "base".to_owned()),
            ("x-hook".to_owned(), "hook".to_owned())
        ])
    );
    assert_eq!(seen[0].stream.metadata.get("base"), Some(&json!(true)));
    assert_eq!(seen[0].stream.metadata.get("hook"), Some(&json!(true)));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_provider_request_hooks_chain_and_delete_options() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_options_ref = seen_options.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        seen_options_ref
            .lock()
            .expect("mutex")
            .push(options.clone());
        faux_assistant_message("ok", Default::default())
    })]);

    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.stream_options.stream.timeout_ms = Some(1000);
    options.stream_options.stream.max_retries = Some(2);
    options
        .stream_options
        .stream
        .headers
        .extend(BTreeMap::from([
            ("keep".to_owned(), "base".to_owned()),
            ("remove".to_owned(), "base".to_owned()),
        ]));
    options
        .stream_options
        .stream
        .metadata
        .extend(Map::from_iter([
            ("keep".to_owned(), json!("base")),
            ("remove".to_owned(), json!("base")),
        ]));
    let harness = AgentHarness::new(options);

    harness.on_before_provider_request(|event| {
        assert_eq!(
            event.stream_options.stream.headers.get("remove"),
            Some(&"base".to_owned())
        );
        Ok(Some(BeforeProviderRequestResult {
            stream_options: Some(AgentHarnessStreamOptionsPatch {
                headers: Some(HeaderMapPatch::Merge(BTreeMap::from([
                    ("first".to_owned(), Some("1".to_owned())),
                    ("remove".to_owned(), None),
                ]))),
                metadata: Some(MetadataMapPatch::Merge(BTreeMap::from([
                    ("first".to_owned(), Some(json!(1))),
                    ("remove".to_owned(), None),
                ]))),
                ..Default::default()
            }),
        }))
    });
    harness.on_before_provider_request(|event| {
        assert_eq!(
            event.stream_options.stream.headers,
            BTreeMap::from([
                ("first".to_owned(), "1".to_owned()),
                ("keep".to_owned(), "base".to_owned())
            ])
        );
        assert_eq!(
            event.stream_options.stream.metadata.get("first"),
            Some(&json!(1))
        );
        Ok(Some(BeforeProviderRequestResult {
            stream_options: Some(AgentHarnessStreamOptionsPatch {
                timeout_ms: Some(OptionPatch::Clear),
                headers: Some(HeaderMapPatch::Merge(BTreeMap::from([(
                    "second".to_owned(),
                    Some("2".to_owned()),
                )]))),
                metadata: Some(MetadataMapPatch::Clear),
                ..Default::default()
            }),
        }))
    });

    harness.prompt("hello").await.expect("prompt");

    let seen = seen_options.lock().expect("mutex").clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].stream.timeout_ms, None);
    assert_eq!(seen[0].stream.max_retries, Some(2));
    assert_eq!(
        seen[0].stream.headers,
        BTreeMap::from([
            ("first".to_owned(), "1".to_owned()),
            ("keep".to_owned(), "base".to_owned()),
            ("second".to_owned(), "2".to_owned())
        ])
    );
    assert!(seen[0].stream.metadata.is_empty());
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_provider_payload_hooks_chain_payloads() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let seen_payloads = Arc::new(Mutex::new(Vec::<Value>::new()));
    let final_payload = Arc::new(Mutex::new(None::<Value>));
    let final_payload_ref = final_payload.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, model| {
        let payload = options
            .apply_payload_hooks(model, json!({ "steps": ["provider"] }))
            .expect("payload hooks");
        *final_payload_ref.lock().expect("mutex") = Some(payload);
        faux_assistant_message("ok", Default::default())
    })]);

    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    ));
    let seen_first = seen_payloads.clone();
    harness.on_before_provider_payload(move |event| {
        seen_first
            .lock()
            .expect("mutex")
            .push(event.payload.clone());
        Ok(Some(BeforeProviderPayloadResult {
            payload: json!({ "steps": ["provider", "first"] }),
        }))
    });
    let seen_second = seen_payloads.clone();
    harness.on_before_provider_payload(move |event| {
        seen_second
            .lock()
            .expect("mutex")
            .push(event.payload.clone());
        Ok(Some(BeforeProviderPayloadResult {
            payload: json!({ "steps": ["provider", "first", "second"] }),
        }))
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *seen_payloads.lock().expect("mutex"),
        vec![
            json!({ "steps": ["provider"] }),
            json!({ "steps": ["provider", "first"] })
        ]
    );
    assert_eq!(
        *final_payload.lock().expect("mutex"),
        Some(json!({ "steps": ["provider", "first", "second"] }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_removes_registered_listeners_and_hooks() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let removed = Arc::new(Mutex::new(Vec::<String>::new()));
    let active = Arc::new(Mutex::new(Vec::<String>::new()));
    let captured_payload = Arc::new(Mutex::new(None::<Value>));
    let captured_headers = Arc::new(Mutex::new(Vec::<BTreeMap<String, String>>::new()));

    let mut args = Map::new();
    args.insert("expression".to_owned(), Value::String("3 + 4".to_owned()));
    let payload_ref = captured_payload.clone();
    let headers_first = captured_headers.clone();
    let headers_second = captured_headers.clone();
    registration.set_responses(vec![
        faux_response_factory(move |_, options, _, model| {
            headers_first
                .lock()
                .expect("headers")
                .push(options.stream.headers.clone());
            let payload = options
                .apply_payload_hooks(model, json!({ "steps": ["provider"] }))
                .expect("payload hooks");
            *payload_ref.lock().expect("payload") = Some(payload);
            faux_assistant_message(
                faux_tool_call("calculate", args.clone(), Some("call-remove".to_owned())),
                FauxAssistantOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
        }),
        faux_response_factory(move |_, options, _, _| {
            headers_second
                .lock()
                .expect("headers")
                .push(options.stream.headers.clone());
            faux_assistant_message("done", Default::default())
        }),
    ]);

    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.tools = vec![calculate_tool()];
    let harness = AgentHarness::new(options);

    let removed_ref = removed.clone();
    let listener_id = harness.subscribe(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("listener".to_owned());
    });
    harness.unsubscribe(listener_id);
    let active_ref = active.clone();
    harness.subscribe(move |_| {
        active_ref
            .lock()
            .expect("active")
            .push("listener".to_owned());
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_before_agent_start(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("before".to_owned());
        Ok(None)
    });
    harness.remove_before_agent_start_hook(hook_id);
    let active_ref = active.clone();
    harness.on_before_agent_start(move |_| {
        active_ref.lock().expect("active").push("before".to_owned());
        Ok(None)
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_context(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("context".to_owned());
        Ok(None)
    });
    harness.remove_context_hook(hook_id);
    let active_ref = active.clone();
    harness.on_context(move |_| {
        active_ref
            .lock()
            .expect("active")
            .push("context".to_owned());
        Ok(None)
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_before_provider_request(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("request".to_owned());
        Ok(Some(BeforeProviderRequestResult {
            stream_options: Some(AgentHarnessStreamOptionsPatch {
                headers: Some(HeaderMapPatch::Merge(BTreeMap::from([(
                    "x-removed-request".to_owned(),
                    Some("removed".to_owned()),
                )]))),
                ..Default::default()
            }),
        }))
    });
    harness.remove_before_provider_request_hook(hook_id);
    let active_ref = active.clone();
    harness.on_before_provider_request(move |_| {
        active_ref
            .lock()
            .expect("active")
            .push("request".to_owned());
        Ok(Some(BeforeProviderRequestResult {
            stream_options: Some(AgentHarnessStreamOptionsPatch {
                headers: Some(HeaderMapPatch::Merge(BTreeMap::from([(
                    "x-active-request".to_owned(),
                    Some("active".to_owned()),
                )]))),
                ..Default::default()
            }),
        }))
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_before_provider_payload(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("payload".to_owned());
        Ok(Some(BeforeProviderPayloadResult {
            payload: json!({ "steps": ["removed"] }),
        }))
    });
    harness.remove_before_provider_payload_hook(hook_id);
    let active_ref = active.clone();
    harness.on_before_provider_payload(move |event| {
        active_ref
            .lock()
            .expect("active")
            .push("payload".to_owned());
        let mut payload = event.payload;
        payload["steps"]
            .as_array_mut()
            .expect("payload steps")
            .push(json!("active"));
        Ok(Some(BeforeProviderPayloadResult { payload }))
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_tool_call(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("tool_call".to_owned());
        Ok(None)
    });
    harness.remove_tool_call_hook(hook_id);
    let active_ref = active.clone();
    harness.on_tool_call(move |_| {
        active_ref
            .lock()
            .expect("active")
            .push("tool_call".to_owned());
        Ok(None)
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_tool_result(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("tool_result".to_owned());
        Ok(None)
    });
    harness.remove_tool_result_hook(hook_id);
    let active_ref = active.clone();
    harness.on_tool_result(move |_| {
        active_ref
            .lock()
            .expect("active")
            .push("tool_result".to_owned());
        Ok(None)
    });

    let removed_ref = removed.clone();
    let hook_id = harness.on_after_agent_finish(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("after".to_owned());
        Ok(None)
    });
    harness.remove_after_agent_finish_hook(hook_id);
    let active_ref = active.clone();
    harness.on_after_agent_finish(move |_| {
        active_ref.lock().expect("active").push("after".to_owned());
        Ok(None)
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *removed.lock().expect("removed"),
        Vec::<String>::new(),
        "removed listeners and hooks must not fire"
    );
    let active = active.lock().expect("active").clone();
    for expected in [
        "listener",
        "before",
        "context",
        "request",
        "payload",
        "tool_call",
        "tool_result",
        "after",
    ] {
        assert!(
            active.contains(&expected.to_owned()),
            "active hook/listener should have run: {expected}; active={active:?}"
        );
    }
    assert_eq!(
        *captured_payload.lock().expect("payload"),
        Some(json!({ "steps": ["provider", "active"] }))
    );
    let captured_headers = captured_headers.lock().expect("headers");
    assert!(captured_headers.iter().all(|headers| {
        headers.get("x-removed-request").is_none()
            && headers.get("x-active-request").map(String::as_str) == Some("active")
    }));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_removes_registered_summary_hooks() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("remove-summary-hooks-model")],
        ..Default::default()
    });

    let mut compaction_session = Session::new(InMemorySessionStorage::new());
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append old user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append old assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent user");

    let compact_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session.clone(),
        registration.get_model(),
    ));
    let removed = Arc::new(Mutex::new(Vec::<String>::new()));
    let active = Arc::new(Mutex::new(Vec::<String>::new()));

    let removed_ref = removed.clone();
    let hook_id = compact_harness.on_session_before_compact(move |event| {
        removed_ref
            .lock()
            .expect("removed")
            .push("compact".to_owned());
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "Removed compact summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                },
            }),
        }))
    });
    compact_harness.remove_session_before_compact_hook(hook_id);
    let active_ref = active.clone();
    compact_harness.on_session_before_compact(move |event| {
        active_ref
            .lock()
            .expect("active")
            .push("compact".to_owned());
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "Active compact summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                },
            }),
        }))
    });

    let compact_result = compact_harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact")
        .expect("compact result");
    assert_eq!(compact_result.summary, "Active compact summary");
    assert!(compaction_session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Compaction {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Active compact summary"
    )));
    assert!(compaction_session.entries().iter().all(|entry| !matches!(
        entry,
        SessionTreeEntry::Compaction { summary, .. } if summary == "Removed compact summary"
    )));

    let mut branch_session = Session::new(InMemorySessionStorage::new());
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");
    let branch_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session.clone(),
        registration.get_model(),
    ));

    let removed_ref = removed.clone();
    let hook_id = branch_harness.on_session_before_branch_summary(move |_| {
        removed_ref
            .lock()
            .expect("removed")
            .push("branch".to_owned());
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "Removed branch summary".to_owned(),
                read_files: Vec::new(),
                modified_files: Vec::new(),
            }),
        }))
    });
    branch_harness.remove_session_before_branch_summary_hook(hook_id);
    let active_ref = active.clone();
    branch_harness.on_session_before_branch_summary(move |_| {
        active_ref.lock().expect("active").push("branch".to_owned());
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "Active branch summary".to_owned(),
                read_files: Vec::new(),
                modified_files: Vec::new(),
            }),
        }))
    });

    let branch_result = branch_harness
        .move_session_to(
            Some(anchor),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("move")
        .expect("branch summary");
    assert_eq!(branch_result.summary, "Active branch summary");
    assert!(branch_session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::BranchSummary {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Active branch summary"
    )));
    assert!(branch_session.entries().iter().all(|entry| !matches!(
        entry,
        SessionTreeEntry::BranchSummary { summary, .. } if summary == "Removed branch summary"
    )));
    assert_eq!(*removed.lock().expect("removed"), Vec::<String>::new());
    assert_eq!(
        *active.lock().expect("active"),
        vec!["compact".to_owned(), "branch".to_owned()]
    );

    registration.unregister();
}

#[tokio::test]
async fn agent_harness_compacts_session_and_persists_summary() {
    let mut definition = FauxModelDefinition::new("summary-model");
    definition.max_tokens = 8_192;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![definition],
        ..Default::default()
    });
    let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
    let options_seen = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let prompts_ref = prompts.clone();
    let options_ref = options_seen.clone();
    registration.set_responses(vec![faux_response_factory(
        move |context, options, _, _| {
            prompts_ref
                .lock()
                .expect("prompts")
                .extend(user_texts(&context.messages));
            options_ref.lock().expect("options").push(options.clone());
            faux_assistant_message("## Goal\nHarness summary", Default::default())
        },
    )]);

    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let mut options =
        AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    options.get_api_key_and_headers = Some(Arc::new(|_| {
        Ok(ProviderAuth {
            api_key: Some("summary-key".to_owned()),
            headers: BTreeMap::from([("x-summary".to_owned(), "yes".to_owned())]),
        })
    }));
    let harness = AgentHarness::new(options);
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::Compaction(result) => events_ref
            .lock()
            .expect("events")
            .push(format!("compaction:{}", result.first_kept_entry_id)),
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        _ => {}
    });

    let result = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: Some("Preserve decisions.".to_owned()),
        })
        .await
        .expect("compact")
        .expect("compaction result");

    assert!(result.summary.contains("Harness summary"));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    let entries = session.entries();
    let compaction = entries
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::Compaction {
                summary,
                first_kept_entry_id,
                tokens_before,
                ..
            } => Some((summary, first_kept_entry_id, tokens_before)),
            _ => None,
        })
        .expect("compaction entry");
    assert!(compaction.0.contains("Harness summary"));
    assert_eq!(compaction.1, &result.first_kept_entry_id);
    assert_eq!(*compaction.2, result.tokens_before);

    let context = session.build_context().expect("context");
    assert!(matches!(
        context.messages.first(),
        Some(SessionMessage::CompactionSummary { summary, .. })
            if summary.contains("Harness summary")
    ));
    let options_seen = options_seen.lock().expect("options");
    assert_eq!(
        options_seen[0].stream.api_key.as_deref(),
        Some("summary-key")
    );
    assert_eq!(
        options_seen[0]
            .stream
            .headers
            .get("x-summary")
            .map(String::as_str),
        Some("yes")
    );
    let prompts = prompts.lock().expect("prompts");
    assert!(prompts[0].contains("Preserve decisions."));
    let events = events.lock().expect("events");
    assert!(events.iter().any(|event| event.starts_with("compaction:")));
    assert!(events.iter().any(|event| event == "save:false"));

    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_compact_hook_can_supply_summary() {
    let mut definition = FauxModelDefinition::new("hook-summary-model");
    definition.max_tokens = 8_192;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![definition],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let options = AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    let harness = AgentHarness::new(options);
    let hook_seen = Arc::new(Mutex::new(Vec::<(usize, Option<String>)>::new()));
    let hook_seen_ref = hook_seen.clone();
    harness.on_session_before_compact(move |event| {
        hook_seen_ref.lock().expect("hook seen").push((
            event.branch_entries.len(),
            event.custom_instructions.clone(),
        ));
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "Hook supplied summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: vec!["src/lib.rs".to_owned()],
                    modified_files: vec!["src/main.rs".to_owned()],
                },
            }),
        }))
    });
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SessionCompact(event) => {
            let from_hook = match &event.compaction_entry {
                SessionTreeEntry::Compaction { from_hook, .. } => *from_hook,
                _ => None,
            };
            events_ref
                .lock()
                .expect("events")
                .push(format!("session_compact:{}:{from_hook:?}", event.from_hook));
        }
        AgentHarnessEvent::Compaction(result) => events_ref
            .lock()
            .expect("events")
            .push(format!("compaction:{}", result.summary)),
        _ => {}
    });

    let result = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: Some("hook instructions".to_owned()),
        })
        .await
        .expect("compact")
        .expect("hook compaction result");

    assert_eq!(result.summary, "Hook supplied summary");
    assert_eq!(
        *hook_seen.lock().expect("hook seen"),
        vec![(3, Some("hook instructions".to_owned()))]
    );
    let entries = session.entries();
    let compaction = entries
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::Compaction {
                summary,
                details,
                from_hook,
                ..
            } => Some((summary, details, from_hook)),
            _ => None,
        })
        .expect("compaction entry");
    assert_eq!(compaction.0, "Hook supplied summary");
    assert_eq!(*compaction.2, Some(true));
    assert_eq!(
        compaction
            .1
            .as_ref()
            .and_then(|details| details.get("readFiles")),
        Some(&json!(["src/lib.rs"]))
    );
    assert_eq!(
        *events.lock().expect("events"),
        vec![
            "session_compact:true:Some(true)".to_owned(),
            "compaction:Hook supplied summary".to_owned()
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_summary_hooks_run_all_and_persist_last_result() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("multi-hook-summary-model")],
        ..Default::default()
    });

    let mut compaction_session = Session::new(InMemorySessionStorage::new());
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let compact_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session.clone(),
        registration.get_model(),
    ));
    let compact_seen = Arc::new(Mutex::new(Vec::<String>::new()));
    for (label, summary) in [
        ("compact-first", "First compact summary"),
        ("compact-second", "Second compact summary"),
    ] {
        let seen = compact_seen.clone();
        compact_harness.on_session_before_compact(move |event| {
            seen.lock().expect("compact seen").push(label.to_owned());
            Ok(Some(SessionBeforeCompactResult {
                cancel: false,
                compaction: Some(CompactionResult {
                    summary: summary.to_owned(),
                    first_kept_entry_id: event.preparation.first_kept_entry_id,
                    tokens_before: event.preparation.tokens_before,
                    details: CompactionDetails {
                        read_files: Vec::new(),
                        modified_files: Vec::new(),
                    },
                }),
            }))
        });
    }

    let compact_result = compact_harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact")
        .expect("compact result");
    assert_eq!(compact_result.summary, "Second compact summary");
    assert_eq!(
        *compact_seen.lock().expect("compact seen"),
        vec!["compact-first".to_owned(), "compact-second".to_owned()]
    );
    assert!(compaction_session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Compaction {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Second compact summary"
    )));

    let mut branch_session = Session::new(InMemorySessionStorage::new());
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let branch_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session.clone(),
        registration.get_model(),
    ));
    let branch_seen = Arc::new(Mutex::new(Vec::<String>::new()));
    for (label, summary) in [
        ("branch-first", "First branch summary"),
        ("branch-second", "Second branch summary"),
    ] {
        let seen = branch_seen.clone();
        branch_harness.on_session_before_branch_summary(move |_| {
            seen.lock().expect("branch seen").push(label.to_owned());
            Ok(Some(SessionBeforeBranchSummaryResult {
                skip_summary: false,
                summary: Some(BranchSummaryResult {
                    summary: summary.to_owned(),
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                }),
            }))
        });
    }

    let branch_result = branch_harness
        .move_session_to(
            Some(anchor),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("move with branch summary")
        .expect("branch result");
    assert_eq!(branch_result.summary, "Second branch summary");
    assert_eq!(
        *branch_seen.lock().expect("branch seen"),
        vec!["branch-first".to_owned(), "branch-second".to_owned()]
    );
    assert!(branch_session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::BranchSummary {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Second branch summary"
    )));

    registration.unregister();
}

#[tokio::test]
async fn agent_harness_summary_hooks_persist_to_jsonl_storage() {
    let compaction_path = temp_harness_session_path("compact-jsonl");
    let compaction_storage = JsonlSessionStorage::create(
        &compaction_path,
        "/tmp/compact-jsonl",
        "compact-jsonl-session",
        None,
    )
    .expect("create compact jsonl storage");
    let mut compaction_session = Session::new(compaction_storage);
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append old user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append old assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent user");
    let compaction_metadata = compaction_session
        .jsonl_metadata()
        .expect("compact metadata");
    let compact_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    ));
    compact_harness.on_session_before_compact(|event| {
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "JSONL hook compaction summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: vec!["src/read.rs".to_owned()],
                    modified_files: vec!["src/write.rs".to_owned()],
                },
            }),
        }))
    });

    let compact_result = compact_harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: Some("persist compaction".to_owned()),
        })
        .await
        .expect("compact jsonl")
        .expect("compaction result");
    assert_eq!(compact_result.summary, "JSONL hook compaction summary");

    let reloaded_compaction =
        Session::new(JsonlSessionStorage::open(&compaction_metadata.path).expect("reload compact"));
    let compaction_entry = reloaded_compaction
        .entries()
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::Compaction {
                summary,
                details,
                from_hook,
                ..
            } => Some((summary.clone(), details.clone(), *from_hook)),
            _ => None,
        })
        .expect("persisted compaction entry");
    assert_eq!(compaction_entry.0, "JSONL hook compaction summary");
    assert_eq!(compaction_entry.2, Some(true));
    assert_eq!(
        compaction_entry
            .1
            .as_ref()
            .and_then(|details| details.get("readFiles")),
        Some(&json!(["src/read.rs"]))
    );
    assert_eq!(
        compaction_entry
            .1
            .as_ref()
            .and_then(|details| details.get("modifiedFiles")),
        Some(&json!(["src/write.rs"]))
    );
    assert!(matches!(
        reloaded_compaction
            .build_context()
            .expect("compact context")
            .messages
            .first(),
        Some(SessionMessage::CompactionSummary { summary, .. })
            if summary.contains("JSONL hook compaction summary")
    ));

    let branch_path = temp_harness_session_path("branch-jsonl");
    let branch_storage = JsonlSessionStorage::create(
        &branch_path,
        "/tmp/branch-jsonl",
        "branch-jsonl-session",
        None,
    )
    .expect("create branch jsonl storage");
    let mut branch_session = Session::new(branch_storage);
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch work");
    let branch_metadata = branch_session.jsonl_metadata().expect("branch metadata");
    let branch_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    ));
    branch_harness.on_session_before_branch_summary(|_| {
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "JSONL hook branch summary".to_owned(),
                read_files: vec!["src/branch-read.rs".to_owned()],
                modified_files: vec!["src/branch-write.rs".to_owned()],
            }),
        }))
    });

    let branch_result = branch_harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("move jsonl")
        .expect("branch summary result");
    assert_eq!(branch_result.summary, "JSONL hook branch summary");

    let reloaded_branch =
        Session::new(JsonlSessionStorage::open(&branch_metadata.path).expect("reload branch"));
    let branch_summary_entry = reloaded_branch
        .entries()
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::BranchSummary {
                summary,
                from_id,
                details,
                from_hook,
                ..
            } => Some((
                summary.clone(),
                from_id.clone(),
                details.clone(),
                *from_hook,
            )),
            _ => None,
        })
        .expect("persisted branch summary entry");
    assert_eq!(branch_summary_entry.0, "JSONL hook branch summary");
    assert_eq!(branch_summary_entry.1, anchor);
    assert_eq!(branch_summary_entry.3, Some(true));
    assert_eq!(
        branch_summary_entry
            .2
            .as_ref()
            .and_then(|details| details.get("readFiles")),
        Some(&json!(["src/branch-read.rs"]))
    );
    assert_eq!(
        branch_summary_entry
            .2
            .as_ref()
            .and_then(|details| details.get("modifiedFiles")),
        Some(&json!(["src/branch-write.rs"]))
    );
    assert_eq!(
        reloaded_branch
            .build_context()
            .expect("branch context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "branchSummary"]
    );
}

#[tokio::test]
async fn agent_harness_summary_hook_errors_flush_jsonl_pending_writes_without_summary() {
    let compaction_path = temp_harness_session_path("compact-error-jsonl");
    let compaction_storage = JsonlSessionStorage::create(
        &compaction_path,
        "/tmp/compact-error-jsonl",
        "compact-error-jsonl-session",
        None,
    )
    .expect("create compact error jsonl storage");
    let mut compaction_session = Session::new(compaction_storage);
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append old user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append old assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent user");
    let compaction_metadata = compaction_session
        .jsonl_metadata()
        .expect("compact error metadata");
    let compact_harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    )));
    let compact_hook_harness = compact_harness.clone();
    compact_harness.on_session_before_compact(move |_| {
        compact_hook_harness
            .append_session_name("compact jsonl hook failure flush")
            .expect("queue compact session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "compact jsonl hook exploded",
        ))
    });

    let compact_error = compact_harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect_err("compact hook error");

    assert_eq!(compact_error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(compact_error.message, "compact jsonl hook exploded");
    let reloaded_compaction = Session::new(
        JsonlSessionStorage::open(&compaction_metadata.path).expect("reload compact error"),
    );
    assert!(reloaded_compaction.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "compact jsonl hook failure flush"
    )));
    assert!(
        reloaded_compaction
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::Compaction { .. }))
    );

    let branch_path = temp_harness_session_path("branch-error-jsonl");
    let branch_storage = JsonlSessionStorage::create(
        &branch_path,
        "/tmp/branch-error-jsonl",
        "branch-error-jsonl-session",
        None,
    )
    .expect("create branch error jsonl storage");
    let mut branch_session = Session::new(branch_storage);
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch work");
    let branch_leaf = branch_session
        .leaf_id()
        .expect("current branch leaf")
        .expect("branch leaf");
    let branch_metadata = branch_session
        .jsonl_metadata()
        .expect("branch error metadata");
    let branch_harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    )));
    let branch_hook_harness = branch_harness.clone();
    branch_harness.on_session_before_branch_summary(move |_| {
        branch_hook_harness
            .append_session_name("branch jsonl hook failure flush")
            .expect("queue branch session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "branch jsonl hook exploded",
        ))
    });

    let branch_error = branch_harness
        .move_session_to(
            Some(anchor),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect_err("branch hook error");

    assert_eq!(branch_error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(branch_error.message, "branch jsonl hook exploded");
    let reloaded_branch = Session::new(
        JsonlSessionStorage::open(&branch_metadata.path).expect("reload branch error"),
    );
    let session_info_parent = reloaded_branch
        .entries()
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::SessionInfo {
                name: Some(name),
                parent_id,
                ..
            } if name == "branch jsonl hook failure flush" => Some(parent_id.clone()),
            _ => None,
        })
        .expect("persisted branch session name");
    assert_eq!(session_info_parent, Some(branch_leaf));
    assert!(
        reloaded_branch
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::BranchSummary { .. }))
    );
    assert_eq!(
        reloaded_branch
            .build_context()
            .expect("branch context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "user"]
    );
}

#[tokio::test]
async fn agent_harness_summary_hook_cancel_and_skip_flush_jsonl_pending_writes_without_summary() {
    let compaction_path = temp_harness_session_path("compact-cancel-jsonl");
    let compaction_storage = JsonlSessionStorage::create(
        &compaction_path,
        "/tmp/compact-cancel-jsonl",
        "compact-cancel-jsonl-session",
        None,
    )
    .expect("create compact cancel jsonl storage");
    let mut compaction_session = Session::new(compaction_storage);
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append old user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append old assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent user");
    let compaction_metadata = compaction_session
        .jsonl_metadata()
        .expect("compact cancel metadata");
    let compact_harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    )));
    let compact_events = Arc::new(Mutex::new(Vec::<String>::new()));
    let compact_events_ref = compact_events.clone();
    compact_harness.subscribe(move |event| {
        if let AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } = event
        {
            compact_events_ref
                .lock()
                .expect("compact events")
                .push(format!("save:{had_pending_mutations}"));
        }
    });
    let compact_hook_harness = compact_harness.clone();
    compact_harness.on_session_before_compact(move |_| {
        compact_hook_harness
            .append_session_name("compact jsonl hook cancel flush")
            .expect("queue compact cancel session name");
        Ok(Some(SessionBeforeCompactResult {
            cancel: true,
            compaction: None,
        }))
    });

    let compact_result = compact_harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact cancel");

    assert!(compact_result.is_none());
    assert_eq!(
        *compact_events.lock().expect("compact events"),
        vec!["save:true"]
    );
    let reloaded_compaction = Session::new(
        JsonlSessionStorage::open(&compaction_metadata.path).expect("reload compact cancel"),
    );
    assert!(reloaded_compaction.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "compact jsonl hook cancel flush"
    )));
    assert!(
        reloaded_compaction
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::Compaction { .. }))
    );

    let branch_path = temp_harness_session_path("branch-skip-jsonl");
    let branch_storage = JsonlSessionStorage::create(
        &branch_path,
        "/tmp/branch-skip-jsonl",
        "branch-skip-jsonl-session",
        None,
    )
    .expect("create branch skip jsonl storage");
    let mut branch_session = Session::new(branch_storage);
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch work");
    let branch_metadata = branch_session
        .jsonl_metadata()
        .expect("branch skip metadata");
    let branch_harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session,
        Model::faux("jsonl-api", "jsonl-provider", "jsonl-model"),
    )));
    let branch_events = Arc::new(Mutex::new(Vec::<String>::new()));
    let branch_events_ref = branch_events.clone();
    branch_harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => branch_events_ref
            .lock()
            .expect("branch events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::SessionBranchSummary(_) => branch_events_ref
            .lock()
            .expect("branch events")
            .push("session_branch".to_owned()),
        AgentHarnessEvent::BranchSummary(_) => branch_events_ref
            .lock()
            .expect("branch events")
            .push("branch".to_owned()),
        _ => {}
    });
    let branch_hook_harness = branch_harness.clone();
    branch_harness.on_session_before_branch_summary(move |_| {
        branch_hook_harness
            .append_session_name("branch jsonl hook skip flush")
            .expect("queue branch skip session name");
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: true,
            summary: None,
        }))
    });

    let branch_result = branch_harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("branch skip");

    assert!(branch_result.is_none());
    assert_eq!(
        *branch_events.lock().expect("branch events"),
        vec!["save:true"]
    );
    let reloaded_branch =
        Session::new(JsonlSessionStorage::open(&branch_metadata.path).expect("reload branch skip"));
    let session_info_parent = reloaded_branch
        .entries()
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::SessionInfo {
                name: Some(name),
                parent_id,
                ..
            } if name == "branch jsonl hook skip flush" => Some(parent_id.clone()),
            _ => None,
        })
        .expect("persisted branch skip session name");
    assert_eq!(session_info_parent, Some(anchor));
    assert!(
        reloaded_branch
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::BranchSummary { .. }))
    );
}

#[tokio::test]
async fn agent_harness_session_before_compact_hook_flushes_pending_writes_on_success() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("compact-hook-flush-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } = event
        {
            events_ref
                .lock()
                .expect("events")
                .push(format!("save:{had_pending_mutations}"));
        }
    });
    let hook_harness = harness.clone();
    harness.on_session_before_compact(move |event| {
        hook_harness
            .append_session_name("compact hook success flush")
            .expect("queue session name");
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "Hook summary with queued write".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                },
            }),
        }))
    });

    let result = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact")
        .expect("compaction result");

    assert_eq!(result.summary, "Hook summary with queued write");
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "compact hook success flush"
    )));
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Compaction {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Hook summary with queued write"
    )));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_compact_hook_can_cancel() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("cancel-summary-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let options = AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    let harness = AgentHarness::new(options);
    harness.on_session_before_compact(|_| {
        Ok(Some(SessionBeforeCompactResult {
            cancel: true,
            compaction: None,
        }))
    });

    let result = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact");

    assert!(result.is_none());
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::Compaction { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_compact_hook_errors_flush_pending_writes_without_summary() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("compact-error-summary-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::SessionCompact(_) => events_ref
            .lock()
            .expect("events")
            .push("session_compact".to_owned()),
        AgentHarnessEvent::Compaction(_) => events_ref
            .lock()
            .expect("events")
            .push("compaction".to_owned()),
        _ => {}
    });
    let hook_harness = harness.clone();
    harness.on_session_before_compact(move |_| {
        hook_harness
            .append_session_name("compact hook failure flush")
            .expect("queue session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "compact hook exploded",
        ))
    });

    let error = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect_err("compact hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "compact hook exploded");
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "compact hook failure flush"
    )));
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::Compaction { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_compaction_generation_errors_flush_pending_writes_without_summary() {
    let mut definition = FauxModelDefinition::new("compact-generation-error-model");
    definition.max_tokens = 8_192;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![definition],
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            "not a summary",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("summary provider failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append user");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append recent");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::SessionCompact(_) => events_ref
            .lock()
            .expect("events")
            .push("session_compact".to_owned()),
        AgentHarnessEvent::Compaction(_) => events_ref
            .lock()
            .expect("events")
            .push("compaction".to_owned()),
        _ => {}
    });
    let hook_harness = harness.clone();
    harness.on_session_before_compact(move |_| {
        hook_harness
            .append_session_name("compact generation failure flush")
            .expect("queue session name");
        Ok(None)
    });

    let error = harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect_err("compaction generation error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert!(error.message.contains("summary provider failed"));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "compact generation failure flush"
    )));
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::Compaction { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_move_session_to_generates_and_persists_branch_summary() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("branch-summary-model")],
        ..Default::default()
    });
    let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
    let options_seen = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let prompts_ref = prompts.clone();
    let options_ref = options_seen.clone();
    registration.set_responses(vec![faux_response_factory(
        move |context, options, _, _| {
            prompts_ref
                .lock()
                .expect("prompts")
                .extend(user_texts(&context.messages));
            options_ref.lock().expect("options").push(options.clone());
            faux_assistant_message("## Goal\nGenerated branch summary", Default::default())
        },
    )]);

    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::Assistant(faux_assistant_message(
            faux_tool_call(
                "read",
                json!({ "path": "src/branch.rs" })
                    .as_object()
                    .cloned()
                    .unwrap_or_else(Map::new),
                Some("call-read".to_owned()),
            ),
            Default::default(),
        )))
        .expect("append assistant");
    session
        .append_message(Message::User(UserMessage::text("abandoned branch request")))
        .expect("append branch user");

    let mut options =
        AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    options.get_api_key_and_headers = Some(Arc::new(|_| {
        Ok(ProviderAuth {
            api_key: Some("branch-key".to_owned()),
            headers: BTreeMap::from([("x-branch".to_owned(), "yes".to_owned())]),
        })
    }));
    let harness = AgentHarness::new(options);
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SessionBranchSummary(event) => {
            let from_hook = match &event.branch_summary_entry {
                SessionTreeEntry::BranchSummary { from_hook, .. } => *from_hook,
                _ => None,
            };
            events_ref
                .lock()
                .expect("events")
                .push(format!("session_branch:{}:{from_hook:?}", event.from_hook));
        }
        AgentHarnessEvent::BranchSummary(result) => events_ref
            .lock()
            .expect("events")
            .push(format!("branch:{}", result.summary.contains("Generated"))),
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        _ => {}
    });

    let result = harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions {
                    custom_instructions: Some("Focus on branch decisions.".to_owned()),
                    replace_instructions: false,
                    reserve_tokens: Some(128),
                }),
            },
        )
        .await
        .expect("move with summary")
        .expect("summary result");

    assert!(result.summary.contains("Generated branch summary"));
    assert_eq!(result.read_files, vec!["src/branch.rs"]);
    assert!(result.modified_files.is_empty());
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);

    let entries = session.entries();
    let branch_summary = entries
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::BranchSummary {
                summary,
                from_id,
                details,
                from_hook,
                ..
            } => Some((summary, from_id, details, from_hook)),
            _ => None,
        })
        .expect("branch summary entry");
    assert!(branch_summary.0.contains("Generated branch summary"));
    assert_eq!(branch_summary.1, &anchor);
    assert_eq!(*branch_summary.3, None);
    assert_eq!(
        branch_summary
            .2
            .as_ref()
            .and_then(|details| details.get("readFiles")),
        Some(&json!(["src/branch.rs"]))
    );
    assert_eq!(
        session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "branchSummary"]
    );

    let options_seen = options_seen.lock().expect("options");
    assert_eq!(
        options_seen[0].stream.api_key.as_deref(),
        Some("branch-key")
    );
    assert_eq!(
        options_seen[0]
            .stream
            .headers
            .get("x-branch")
            .map(String::as_str),
        Some("yes")
    );
    let prompts = prompts.lock().expect("prompts");
    assert!(prompts[0].contains("Additional focus: Focus on branch decisions."));
    assert!(prompts[0].contains("[Assistant tool calls]: read(path=\"src/branch.rs\")"));
    assert_eq!(
        *events.lock().expect("events"),
        vec![
            "session_branch:false:None".to_owned(),
            "branch:true".to_owned(),
            "save:false".to_owned()
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_branch_summary_hook_can_supply_summary() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("hook-branch-summary-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    ));
    let hook_seen = Arc::new(Mutex::new(
        Vec::<(usize, Option<String>, Option<String>)>::new(),
    ));
    let hook_seen_ref = hook_seen.clone();
    harness.on_session_before_branch_summary(move |event| {
        hook_seen_ref.lock().expect("hook seen").push((
            event.entries.len(),
            event.common_ancestor_id.clone(),
            event.custom_instructions.clone(),
        ));
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "Hook branch summary".to_owned(),
                read_files: vec!["src/read.rs".to_owned()],
                modified_files: vec!["src/write.rs".to_owned()],
            }),
        }))
    });

    let result = harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions {
                    custom_instructions: Some("hook branch instructions".to_owned()),
                    replace_instructions: false,
                    reserve_tokens: None,
                }),
            },
        )
        .await
        .expect("move with hook summary")
        .expect("hook summary result");

    assert_eq!(result.summary, "Hook branch summary");
    assert_eq!(
        *hook_seen.lock().expect("hook seen"),
        vec![(
            1,
            Some(anchor.clone()),
            Some("hook branch instructions".to_owned())
        )]
    );
    let branch_summary = session
        .entries()
        .iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::BranchSummary {
                summary,
                details,
                from_hook,
                ..
            } => Some((summary.clone(), details.clone(), *from_hook)),
            _ => None,
        })
        .expect("branch summary entry");
    assert_eq!(branch_summary.0, "Hook branch summary");
    assert_eq!(branch_summary.2, Some(true));
    assert_eq!(
        branch_summary
            .1
            .as_ref()
            .and_then(|details| details.get("readFiles")),
        Some(&json!(["src/read.rs"]))
    );
    assert_eq!(
        branch_summary
            .1
            .as_ref()
            .and_then(|details| details.get("modifiedFiles")),
        Some(&json!(["src/write.rs"]))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_branch_summary_hook_flushes_pending_writes_on_success() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("branch-hook-flush-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } = event
        {
            events_ref
                .lock()
                .expect("events")
                .push(format!("save:{had_pending_mutations}"));
        }
    });
    let hook_harness = harness.clone();
    harness.on_session_before_branch_summary(move |_| {
        hook_harness
            .append_session_name("branch hook success flush")
            .expect("queue session name");
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "Hook branch summary with queued write".to_owned(),
                read_files: Vec::new(),
                modified_files: Vec::new(),
            }),
        }))
    });

    let result = harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("move with hook summary")
        .expect("summary result");

    assert_eq!(result.summary, "Hook branch summary with queued write");
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert_eq!(
        session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "branchSummary"]
    );
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "branch hook success flush"
    )));
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::BranchSummary {
            summary,
            from_hook: Some(true),
            ..
        } if summary == "Hook branch summary with queued write"
    )));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_branch_summary_hook_can_skip_summary() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("skip-branch-summary-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    ));
    harness.on_session_before_branch_summary(|_| {
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: true,
            summary: None,
        }))
    });

    let result = harness
        .move_session_to(
            Some(anchor.clone()),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect("move and skip summary");

    assert!(result.is_none());
    assert_eq!(session.leaf_id().expect("leaf"), Some(anchor));
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::BranchSummary { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_branch_summary_generation_errors_flush_pending_writes_without_move() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("branch-generation-error-model")],
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            "not a branch summary",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("branch provider failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::SessionBranchSummary(_) => events_ref
            .lock()
            .expect("events")
            .push("session_branch".to_owned()),
        AgentHarnessEvent::BranchSummary(_) => {
            events_ref.lock().expect("events").push("branch".to_owned())
        }
        _ => {}
    });
    let hook_harness = harness.clone();
    harness.on_session_before_branch_summary(move |_| {
        hook_harness
            .append_session_name("branch generation failure flush")
            .expect("queue session name");
        Ok(None)
    });

    let error = harness
        .move_session_to(
            Some(anchor),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect_err("branch generation error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert!(error.message.contains("branch provider failed"));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert_eq!(
        session_user_texts(&session),
        vec!["main path".to_owned(), "branch work".to_owned()]
    );
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "branch generation failure flush"
    )));
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::BranchSummary { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_session_before_branch_summary_hook_errors_flush_pending_writes_without_move()
{
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("branch-error-summary-model")],
        ..Default::default()
    });
    let mut session = Session::new(InMemorySessionStorage::new());
    let anchor = session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append anchor");
    session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch");

    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::SessionBranchSummary(_) => events_ref
            .lock()
            .expect("events")
            .push("session_branch".to_owned()),
        AgentHarnessEvent::BranchSummary(_) => {
            events_ref.lock().expect("events").push("branch".to_owned())
        }
        _ => {}
    });
    let hook_harness = harness.clone();
    harness.on_session_before_branch_summary(move |_| {
        hook_harness
            .append_session_name("branch hook failure flush")
            .expect("queue session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "branch hook exploded",
        ))
    });

    let error = harness
        .move_session_to(
            Some(anchor),
            AgentHarnessMoveSessionOptions {
                branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
            },
        )
        .await
        .expect_err("branch hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "branch hook exploded");
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert_eq!(
        session_user_texts(&session),
        vec!["main path".to_owned(), "branch work".to_owned()]
    );
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "branch hook failure flush"
    )));
    assert!(
        session
            .entries()
            .iter()
            .all(|entry| !matches!(entry, SessionTreeEntry::BranchSummary { .. }))
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_refreshes_stream_options_between_tool_turns() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let first_options = seen_options.clone();
    let second_options = seen_options.clone();
    registration.set_responses(vec![
        faux_response_factory(move |_, options, _, _| {
            first_options.lock().expect("mutex").push(options.clone());
            faux_assistant_message(
                faux_tool_call(
                    "calculate",
                    json!({ "expression": "1 + 1" })
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new),
                    Some("call-1".to_owned()),
                ),
                FauxAssistantOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
        }),
        faux_response_factory(move |_, options, _, _| {
            second_options.lock().expect("mutex").push(options.clone());
            faux_assistant_message("done", Default::default())
        }),
    ]);

    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.tools = vec![calculate_tool()];
    options.stream_options.stream.timeout_ms = Some(1000);
    options
        .stream_options
        .stream
        .headers
        .insert("turn".to_owned(), "first".to_owned());
    let harness = Arc::new(AgentHarness::new(options));
    let harness_for_listener = harness.clone();
    harness.subscribe(move |event| {
        if matches!(
            event,
            AgentHarnessEvent::Agent(AgentEvent::ToolExecutionStart { .. })
        ) {
            let mut options = SimpleStreamOptions::default();
            options.stream.timeout_ms = Some(2000);
            options
                .stream
                .headers
                .insert("turn".to_owned(), "second".to_owned());
            harness_for_listener.set_stream_options(options);
        }
    });

    harness.prompt("hello").await.expect("prompt");

    let seen = seen_options.lock().expect("mutex").clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].stream.timeout_ms, Some(1000));
    assert_eq!(
        seen[0].stream.headers.get("turn"),
        Some(&"first".to_owned())
    );
    assert_eq!(seen[1].stream.timeout_ms, Some(2000));
    assert_eq!(
        seen[1].stream.headers.get("turn"),
        Some(&"second".to_owned())
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_tracks_tools_and_uses_only_active_tools_in_requests() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_tools = Arc::new(Mutex::new(Vec::<String>::new()));
    let request_tools_ref = request_tools.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_tools_ref
            .lock()
            .expect("mutex")
            .extend(context.tools.iter().map(|tool| tool.name.clone()));
        faux_assistant_message("ok", Default::default())
    })]);
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.tools = vec![calculate_tool(), clock_tool()];
    options.active_tool_names = Some(vec!["calculate".to_owned()]);
    let harness = AgentHarness::new(options);

    assert_eq!(
        harness
            .get_tools()
            .iter()
            .map(|tool| tool.definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["calculate", "clock"]
    );
    assert_eq!(
        harness.get_active_tool_names(),
        vec!["calculate".to_owned()]
    );
    harness
        .set_active_tools(vec!["clock".to_owned()])
        .expect("set active");
    assert_eq!(harness.get_active_tool_names(), vec!["clock".to_owned()]);
    let error = harness
        .set_tools(vec![calculate_tool()], Some(vec!["missing".to_owned()]))
        .expect_err("missing active tool");
    assert_eq!(error.code, AgentHarnessErrorCode::InvalidArgument);
    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *request_tools.lock().expect("mutex"),
        vec!["clock".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_orders_pending_async_append_message_after_agent_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let wrote_listener_message = Arc::new(Mutex::new(false));
    let save_points = Arc::new(Mutex::new(Vec::new()));
    let listener_harness = harness.clone();
    let wrote_listener_message_ref = wrote_listener_message.clone();
    let save_points_ref = save_points.clone();
    harness.subscribe_async(move |event| {
        let listener_harness = listener_harness.clone();
        let wrote_listener_message_ref = wrote_listener_message_ref.clone();
        let save_points_ref = save_points_ref.clone();
        async move {
            if matches!(
                &event,
                AgentHarnessEvent::Agent(AgentEvent::MessageEnd {
                    message: AgentMessage::Assistant(_),
                })
            ) {
                let should_write = {
                    let mut wrote = wrote_listener_message_ref.lock().expect("mutex");
                    if *wrote {
                        false
                    } else {
                        *wrote = true;
                        true
                    }
                };
                if should_write {
                    tokio::task::yield_now().await;
                    listener_harness
                        .append_message(Message::User(UserMessage::text("async listener write")))
                        .expect("append message");
                }
            }
            if let AgentHarnessEvent::SavePoint {
                had_pending_mutations,
            } = &event
            {
                save_points_ref
                    .lock()
                    .expect("mutex")
                    .push(*had_pending_mutations);
            }
        }
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        session_user_texts(&session),
        vec!["hello".to_owned(), "async listener write".to_owned()]
    );
    assert_eq!(*save_points.lock().expect("mutex"), vec![true]);
    let entries = session.entries();
    assert!(matches!(
        entries.as_slice(),
        [
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            },
            SessionTreeEntry::Message {
                message: Message::Assistant(_),
                ..
            },
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            }
        ]
    ));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_injects_next_turn_messages_into_next_prompt() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let requests = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let requests_first = requests.clone();
    let requests_second = requests.clone();
    registration.set_responses(vec![
        faux_response_factory(move |context, _, _, _| {
            requests_first
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages));
            faux_assistant_message("first", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            requests_second
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages));
            faux_assistant_message("second", Default::default())
        }),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    ));
    let queue_updates = Arc::new(Mutex::new(Vec::new()));
    let queue_updates_ref = queue_updates.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::QueueUpdate(update) = event {
            queue_updates_ref
                .lock()
                .expect("mutex")
                .push(update.next_turn.len());
        }
    });

    harness.prompt("first").await.expect("first prompt");
    harness.next_turn("next");
    harness.prompt("second").await.expect("second prompt");

    assert_eq!(
        *requests.lock().expect("mutex"),
        vec![
            vec!["first".to_owned()],
            vec!["first".to_owned(), "next".to_owned(), "second".to_owned()],
        ]
    );
    assert_eq!(
        session_user_texts(&session),
        vec!["first".to_owned(), "next".to_owned(), "second".to_owned()]
    );
    assert_eq!(*queue_updates.lock().expect("mutex"), vec![1, 0]);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_before_agent_start_appends_messages_and_persists_them() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let request_text = Arc::new(Mutex::new(Vec::<String>::new()));
    let request_system_prompt = Arc::new(Mutex::new(String::new()));
    let request_text_ref = request_text.clone();
    let request_system_prompt_ref = request_system_prompt.clone();
    registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        request_text_ref
            .lock()
            .expect("mutex")
            .extend(user_texts(&context.messages));
        *request_system_prompt_ref.lock().expect("mutex") =
            context.system_prompt.clone().unwrap_or_default();
        faux_assistant_message("ok", Default::default())
    })]);
    let session = Session::new(InMemorySessionStorage::new());
    let mut options =
        AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    options.system_prompt = "base prompt".to_owned();
    let harness = AgentHarness::new(options);
    harness.on_before_agent_start(|event| {
        assert_eq!(event.prompt, "hello");
        assert_eq!(event.system_prompt, "base prompt");
        Ok(Some(BeforeAgentStartResult {
            messages: Some(vec![Message::User(UserMessage::text("hook")).into()]),
            system_prompt: Some("hook prompt".to_owned()),
        }))
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *request_text.lock().expect("mutex"),
        vec!["hello".to_owned(), "hook".to_owned()]
    );
    assert_eq!(
        request_system_prompt.lock().expect("mutex").as_str(),
        "hook prompt"
    );
    assert_eq!(
        session_user_texts(&session),
        vec!["hello".to_owned(), "hook".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_before_agent_start_errors_flush_pending_writes_without_settled() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("should not run", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::Settled { .. } => events_ref
            .lock()
            .expect("events")
            .push("settled".to_owned()),
        _ => {}
    });
    let hook_harness = harness.clone();
    harness.on_before_agent_start(move |event| {
        assert_eq!(event.prompt, "hello");
        hook_harness
            .append_session_name("before failure flush")
            .expect("queue session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "before start exploded",
        ))
    });

    let error = harness
        .prompt("hello")
        .await
        .expect_err("before hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "before start exploded");
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(*events.lock().expect("events"), vec!["save:true"]);
    assert!(session_user_texts(&session).is_empty());
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "before failure flush"
    )));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_before_agent_start_error_waits_for_async_savepoint_listener() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("should not run", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let listener_started = Arc::new(AtomicBool::new(false));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let settled_seen = Arc::new(AtomicBool::new(false));
    let listener_started_notify = Arc::new(tokio::sync::Notify::new());
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let listener_started_ref = listener_started.clone();
    let listener_finished_ref = listener_finished.clone();
    let settled_seen_ref = settled_seen.clone();
    let listener_started_notify_ref = listener_started_notify.clone();
    let release_listener_ref = release_listener.clone();
    harness.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let listener_finished = listener_finished_ref.clone();
        let settled_seen = settled_seen_ref.clone();
        let listener_started_notify = listener_started_notify_ref.clone();
        let release_listener = release_listener_ref.clone();
        async move {
            match event {
                AgentHarnessEvent::SavePoint {
                    had_pending_mutations,
                } => {
                    assert!(had_pending_mutations);
                    listener_started.store(true, Ordering::SeqCst);
                    listener_started_notify.notify_waiters();
                    release_listener.notified().await;
                    listener_finished.store(true, Ordering::SeqCst);
                }
                AgentHarnessEvent::Settled { .. } => {
                    settled_seen.store(true, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    });
    let hook_harness = harness.clone();
    harness.on_before_agent_start(move |_| {
        hook_harness
            .append_session_name("async before failure flush")
            .expect("queue session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "async before start exploded",
        ))
    });

    let prompt_future = harness.prompt("hello");
    tokio::pin!(prompt_future);
    tokio::select! {
        result = &mut prompt_future => panic!("prompt resolved before async savepoint listener blocked: {result:?}"),
        _ = listener_started_notify.notified() => {}
    }
    assert!(listener_started.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Turn);
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "async before failure flush"
    )));

    release_listener.notify_one();
    let error = prompt_future.await.expect_err("before hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "async before start exploded");
    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(!settled_seen.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert!(session_user_texts(&session).is_empty());
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_after_agent_finish_observes_result_and_persists_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let model = registration.get_model();
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        model.clone(),
    ));
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_ref = seen.clone();
    harness.on_after_agent_finish(move |event| {
        let assistant_text = assistant_text(&event.assistant)
            .unwrap_or_default()
            .to_owned();
        let roles = event
            .messages
            .iter()
            .filter_map(AgentMessage::role)
            .collect::<Vec<_>>()
            .join(",");
        let session_roles = event
            .session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>()
            .join(",");
        seen_ref.lock().expect("seen").push(format!(
            "{}:{}:{}:{}",
            event.model.id, assistant_text, roles, session_roles
        ));
        Ok(Some(AfterAgentFinishResult {
            messages: Some(vec![
                Message::User(UserMessage::text("after hook note")).into(),
            ]),
        }))
    });

    let assistant = harness.prompt("hello").await.expect("prompt");

    assert_eq!(assistant_text(&assistant), Some("done"));
    assert_eq!(
        *seen.lock().expect("seen"),
        vec![format!("{}:done:user,assistant:user,assistant", model.id)]
    );
    assert_eq!(
        session_user_texts(&session),
        vec!["hello".to_owned(), "after hook note".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_after_agent_finish_runs_for_provider_start_failure_and_persists_messages() {
    let model = Model::faux(
        "missing-harness-test-api",
        "missing-provider",
        "missing-model",
    );
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        model.clone(),
    ));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. }) => events_ref
            .lock()
            .expect("events")
            .push("agent_end".to_owned()),
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::Settled { next_turn_count } => events_ref
            .lock()
            .expect("events")
            .push(format!("settled:{next_turn_count}")),
        _ => {}
    });
    let seen_after = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_after_ref = seen_after.clone();
    harness.on_after_agent_finish(move |event| {
        let roles = event
            .messages
            .iter()
            .filter_map(AgentMessage::role)
            .collect::<Vec<_>>()
            .join(",");
        let session_roles = event
            .session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>()
            .join(",");
        seen_after_ref.lock().expect("seen").push(format!(
            "{:?}:{}:{roles}:{session_roles}",
            event.assistant.stop_reason,
            event.assistant.error_message.as_deref().unwrap_or_default()
        ));
        Ok(Some(AfterAgentFinishResult {
            messages: Some(vec![
                Message::User(UserMessage::text("after provider failure hook note")).into(),
            ]),
        }))
    });

    let assistant = harness.prompt("hello").await.expect("prompt resolves");

    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("No API provider registered for api: missing-harness-test-api")
    );
    assert_eq!(
        *seen_after.lock().expect("seen"),
        vec![
            "Error:No API provider registered for api: missing-harness-test-api:user,assistant:user,assistant".to_owned()
        ]
    );
    assert_eq!(
        session_user_texts(&session),
        vec![
            "hello".to_owned(),
            "after provider failure hook note".to_owned()
        ]
    );
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Message {
            message: Message::Assistant(assistant),
            ..
        } if assistant.stop_reason == StopReason::Error
            && assistant.error_message.as_deref()
                == Some("No API provider registered for api: missing-harness-test-api")
    )));
    assert_eq!(
        *events.lock().expect("events"),
        vec![
            "agent_end".to_owned(),
            "save:false".to_owned(),
            "settled:0".to_owned()
        ]
    );
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
}

#[tokio::test]
async fn agent_harness_after_agent_finish_runs_for_aborted_turn_and_settles() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    let listener_harness = harness.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. }) => {
            listener_harness
                .append_session_name("aborted turn flushed")
                .expect("queue session name");
            events_ref
                .lock()
                .expect("events")
                .push("agent_end".to_owned());
        }
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::Settled { next_turn_count } => events_ref
            .lock()
            .expect("events")
            .push(format!("settled:{next_turn_count}")),
        _ => {}
    });
    let seen_after = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_after_ref = seen_after.clone();
    harness.on_after_agent_finish(move |event| {
        seen_after_ref.lock().expect("seen").push(format!(
            "{:?}:{}",
            event.assistant.stop_reason,
            event.assistant.error_message.as_deref().unwrap_or_default()
        ));
        Ok(Some(AfterAgentFinishResult {
            messages: Some(vec![
                Message::User(UserMessage::text("after abort hook note")).into(),
            ]),
        }))
    });

    let running = harness.clone();
    let prompt = tokio::spawn(async move { running.prompt("abort me").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness.abort();
    let assistant = prompt
        .await
        .expect("prompt task")
        .expect("aborted prompt resolves");

    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("Request was aborted")
    );
    assert_eq!(
        *seen_after.lock().expect("seen"),
        vec!["Aborted:Request was aborted".to_owned()]
    );
    assert_eq!(
        *events.lock().expect("events"),
        vec![
            "agent_end".to_owned(),
            "save:true".to_owned(),
            "settled:0".to_owned()
        ]
    );
    assert_eq!(
        session_user_texts(&session),
        vec!["abort me".to_owned(), "after abort hook note".to_owned()]
    );
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "aborted turn flushed"
    )));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_after_agent_finish_errors_flush_pending_writes_without_settled() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_ref = events.clone();
    let listener_harness = harness.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. }) => {
            listener_harness
                .append_session_name("after failure flush")
                .expect("queue session name");
            events_ref
                .lock()
                .expect("events")
                .push("agent_end".to_owned());
        }
        AgentHarnessEvent::SavePoint {
            had_pending_mutations,
        } => events_ref
            .lock()
            .expect("events")
            .push(format!("save:{had_pending_mutations}")),
        AgentHarnessEvent::Settled { .. } => events_ref
            .lock()
            .expect("events")
            .push("settled".to_owned()),
        _ => {}
    });
    harness.on_after_agent_finish(|_| {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "after finish exploded",
        ))
    });

    let error = harness.prompt("hello").await.expect_err("after hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "after finish exploded");
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(
        *events.lock().expect("events"),
        vec!["agent_end".to_owned(), "save:true".to_owned()]
    );
    let entries = session.entries();
    assert_eq!(session_user_texts(&session), vec!["hello".to_owned()]);
    assert!(entries.iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Message {
            message: Message::Assistant(assistant),
            ..
        } if assistant_text(assistant) == Some("done")
    )));
    assert!(entries.iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "after failure flush"
    )));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_after_agent_finish_error_waits_for_async_savepoint_listener() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    )));
    let listener_started = Arc::new(AtomicBool::new(false));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let settled_seen = Arc::new(AtomicBool::new(false));
    let listener_started_notify = Arc::new(tokio::sync::Notify::new());
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let listener_started_ref = listener_started.clone();
    let listener_finished_ref = listener_finished.clone();
    let settled_seen_ref = settled_seen.clone();
    let listener_started_notify_ref = listener_started_notify.clone();
    let release_listener_ref = release_listener.clone();
    harness.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let listener_finished = listener_finished_ref.clone();
        let settled_seen = settled_seen_ref.clone();
        let listener_started_notify = listener_started_notify_ref.clone();
        let release_listener = release_listener_ref.clone();
        async move {
            match event {
                AgentHarnessEvent::SavePoint {
                    had_pending_mutations,
                } => {
                    assert!(had_pending_mutations);
                    listener_started.store(true, Ordering::SeqCst);
                    listener_started_notify.notify_waiters();
                    release_listener.notified().await;
                    listener_finished.store(true, Ordering::SeqCst);
                }
                AgentHarnessEvent::Settled { .. } => {
                    settled_seen.store(true, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    });
    let hook_harness = harness.clone();
    harness.on_after_agent_finish(move |_| {
        hook_harness
            .append_session_name("async after failure flush")
            .expect("queue session name");
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "async after finish exploded",
        ))
    });

    let prompt_future = harness.prompt("hello");
    tokio::pin!(prompt_future);
    tokio::select! {
        result = &mut prompt_future => panic!("prompt resolved before async savepoint listener blocked: {result:?}"),
        _ = listener_started_notify.notified() => {}
    }
    assert!(listener_started.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Turn);
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::Message {
            message: Message::Assistant(assistant),
            ..
        } if assistant_text(assistant) == Some("done")
    )));
    assert!(session.entries().iter().any(|entry| matches!(
        entry,
        SessionTreeEntry::SessionInfo {
            name: Some(name),
            ..
        } if name == "async after failure flush"
    )));

    release_listener.notify_one();
    let error = prompt_future.await.expect_err("after hook error");

    assert_eq!(error.code, AgentHarnessErrorCode::Unknown);
    assert_eq!(error.message, "async after finish exploded");
    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(!settled_seen.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    assert_eq!(session_user_texts(&session), vec!["hello".to_owned()]);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_context_hook_failures_persist_assistant_error_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("should not run", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    ));
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_ref = events.clone();
    harness.subscribe(move |event| match event {
        AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. }) => {
            events_ref.lock().expect("mutex").push("agent_end");
        }
        AgentHarnessEvent::Settled { .. } => {
            events_ref.lock().expect("mutex").push("settled");
        }
        _ => {}
    });
    harness.on_context(|_| {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "context exploded",
        ))
    });

    let response = harness.prompt("hello").await.expect("prompt resolves");

    assert_eq!(response.stop_reason, StopReason::Error);
    assert_eq!(response.error_message.as_deref(), Some("context exploded"));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    let messages: Vec<Message> = session
        .entries()
        .into_iter()
        .filter_map(|entry| match entry {
            SessionTreeEntry::Message { message, .. } => Some(message),
            _ => None,
        })
        .collect();
    assert!(matches!(messages.first(), Some(Message::User(_))));
    let Some(Message::Assistant(assistant)) = messages.get(1) else {
        panic!("assistant error message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(assistant.error_message.as_deref(), Some("context exploded"));
    assert_eq!(*events.lock().expect("mutex"), vec!["agent_end", "settled"]);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_after_agent_finish_runs_for_context_hook_failure() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("should not run", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session.clone(),
        registration.get_model(),
    ));
    harness.on_context(|_| {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::Unknown,
            "context transform failed",
        ))
    });
    let seen_after = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_after_ref = seen_after.clone();
    harness.on_after_agent_finish(move |event| {
        seen_after_ref.lock().expect("seen").push(format!(
            "{:?}:{}:{}",
            event.assistant.stop_reason,
            event.assistant.error_message.as_deref().unwrap_or_default(),
            event
                .messages
                .iter()
                .filter_map(AgentMessage::role)
                .collect::<Vec<_>>()
                .join(",")
        ));
        Ok(Some(AfterAgentFinishResult {
            messages: Some(vec![
                Message::User(UserMessage::text("after context failure note")).into(),
            ]),
        }))
    });

    let response = harness.prompt("hello").await.expect("prompt resolves");

    assert_eq!(response.stop_reason, StopReason::Error);
    assert_eq!(
        response.error_message.as_deref(),
        Some("context transform failed")
    );
    assert_eq!(
        *seen_after.lock().expect("seen"),
        vec!["Error:context transform failed:user,assistant".to_owned()]
    );
    assert_eq!(
        session_user_texts(&session),
        vec!["hello".to_owned(), "after context failure note".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_wait_for_idle_waits_for_async_subscribers() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("ok", Default::default()).into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        session,
        registration.get_model(),
    ));
    let listener_started = Arc::new(AtomicBool::new(false));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let listener_started_notify = Arc::new(tokio::sync::Notify::new());
    let release_listener = Arc::new(tokio::sync::Notify::new());
    let listener_started_ref = listener_started.clone();
    let listener_finished_ref = listener_finished.clone();
    let listener_started_notify_ref = listener_started_notify.clone();
    let release_listener_ref = release_listener.clone();
    harness.subscribe_async(move |event| {
        let listener_started = listener_started_ref.clone();
        let listener_finished = listener_finished_ref.clone();
        let listener_started_notify = listener_started_notify_ref.clone();
        let release_listener = release_listener_ref.clone();
        async move {
            if matches!(event, AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. })) {
                listener_started.store(true, Ordering::SeqCst);
                listener_started_notify.notify_waiters();
                release_listener.notified().await;
                listener_finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let prompt_future = harness.prompt("hello");
    tokio::pin!(prompt_future);
    tokio::select! {
        result = &mut prompt_future => panic!("prompt resolved before async listener blocked: {result:?}"),
        _ = listener_started_notify.notified() => {}
    }
    assert!(listener_started.load(Ordering::SeqCst));
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Turn);

    let idle_future = harness.wait_for_idle();
    tokio::pin!(idle_future);
    tokio::select! {
        _ = &mut idle_future => panic!("wait_for_idle resolved before async listener finished"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
    }
    assert_eq!(harness.phase(), AgentHarnessPhase::Turn);

    release_listener.notify_one();
    let (assistant, ()) = tokio::join!(prompt_future, idle_future);
    let assistant = assistant.expect("prompt");
    assert_eq!(assistant.stop_reason, StopReason::Stop);
    assert!(listener_finished.load(Ordering::SeqCst));
    assert_eq!(harness.phase(), AgentHarnessPhase::Idle);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_summary_events_wait_for_async_subscribers_before_idle() {
    let mut compaction_session = Session::new(InMemorySessionStorage::new());
    compaction_session
        .append_message(Message::User(UserMessage::text("old user context")))
        .expect("append compact user");
    compaction_session
        .append_message(Message::Assistant(faux_assistant_message(
            "old assistant context",
            Default::default(),
        )))
        .expect("append compact assistant");
    compaction_session
        .append_message(Message::User(UserMessage::text("recent request")))
        .expect("append compact recent");
    let compact_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        compaction_session,
        Model::faux("async-api", "async-provider", "async-model"),
    ));
    compact_harness.on_session_before_compact(|event| {
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "Async listener compaction summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                },
            }),
        }))
    });
    let compact_started = Arc::new(AtomicBool::new(false));
    let compact_finished = Arc::new(AtomicBool::new(false));
    let compact_started_notify = Arc::new(tokio::sync::Notify::new());
    let compact_release = Arc::new(tokio::sync::Notify::new());
    let compact_started_ref = compact_started.clone();
    let compact_finished_ref = compact_finished.clone();
    let compact_started_notify_ref = compact_started_notify.clone();
    let compact_release_ref = compact_release.clone();
    compact_harness.subscribe_async(move |event| {
        let compact_started = compact_started_ref.clone();
        let compact_finished = compact_finished_ref.clone();
        let compact_started_notify = compact_started_notify_ref.clone();
        let compact_release = compact_release_ref.clone();
        async move {
            if matches!(event, AgentHarnessEvent::Compaction(_)) {
                compact_started.store(true, Ordering::SeqCst);
                compact_started_notify.notify_waiters();
                compact_release.notified().await;
                compact_finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let compact_future = compact_harness.compact_session(AgentHarnessCompactionOptions {
        settings: CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 128,
            keep_recent_tokens: 1,
        },
        custom_instructions: None,
    });
    tokio::pin!(compact_future);
    tokio::select! {
        result = &mut compact_future => panic!("compact resolved before async compaction listener blocked: {result:?}"),
        _ = compact_started_notify.notified() => {}
    }
    assert!(compact_started.load(Ordering::SeqCst));
    assert!(!compact_finished.load(Ordering::SeqCst));
    assert_eq!(compact_harness.phase(), AgentHarnessPhase::Compaction);

    let compact_idle = compact_harness.wait_for_idle();
    tokio::pin!(compact_idle);
    tokio::select! {
        _ = &mut compact_idle => panic!("compaction wait_for_idle resolved before async listener finished"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
    }
    compact_release.notify_one();
    let (compact_result, ()) = tokio::join!(compact_future, compact_idle);
    let compact_result = compact_result
        .expect("compact result")
        .expect("compaction summary");
    assert_eq!(compact_result.summary, "Async listener compaction summary");
    assert!(compact_finished.load(Ordering::SeqCst));
    assert_eq!(compact_harness.phase(), AgentHarnessPhase::Idle);

    let mut branch_session = Session::new(InMemorySessionStorage::new());
    let anchor = branch_session
        .append_message(Message::User(UserMessage::text("main path")))
        .expect("append branch anchor");
    branch_session
        .append_message(Message::User(UserMessage::text("branch work")))
        .expect("append branch work");
    let branch_harness = AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        branch_session,
        Model::faux("async-api", "async-provider", "async-model"),
    ));
    branch_harness.on_session_before_branch_summary(|_| {
        Ok(Some(SessionBeforeBranchSummaryResult {
            skip_summary: false,
            summary: Some(BranchSummaryResult {
                summary: "Async listener branch summary".to_owned(),
                read_files: Vec::new(),
                modified_files: Vec::new(),
            }),
        }))
    });
    let branch_started = Arc::new(AtomicBool::new(false));
    let branch_finished = Arc::new(AtomicBool::new(false));
    let branch_started_notify = Arc::new(tokio::sync::Notify::new());
    let branch_release = Arc::new(tokio::sync::Notify::new());
    let branch_started_ref = branch_started.clone();
    let branch_finished_ref = branch_finished.clone();
    let branch_started_notify_ref = branch_started_notify.clone();
    let branch_release_ref = branch_release.clone();
    branch_harness.subscribe_async(move |event| {
        let branch_started = branch_started_ref.clone();
        let branch_finished = branch_finished_ref.clone();
        let branch_started_notify = branch_started_notify_ref.clone();
        let branch_release = branch_release_ref.clone();
        async move {
            if matches!(event, AgentHarnessEvent::BranchSummary(_)) {
                branch_started.store(true, Ordering::SeqCst);
                branch_started_notify.notify_waiters();
                branch_release.notified().await;
                branch_finished.store(true, Ordering::SeqCst);
            }
        }
    });

    let branch_future = branch_harness.move_session_to(
        Some(anchor),
        AgentHarnessMoveSessionOptions {
            branch_summary: Some(AgentHarnessBranchSummaryOptions::default()),
        },
    );
    tokio::pin!(branch_future);
    tokio::select! {
        result = &mut branch_future => panic!("move_session_to resolved before async branch summary listener blocked: {result:?}"),
        _ = branch_started_notify.notified() => {}
    }
    assert!(branch_started.load(Ordering::SeqCst));
    assert!(!branch_finished.load(Ordering::SeqCst));
    assert_eq!(branch_harness.phase(), AgentHarnessPhase::BranchSummary);

    let branch_idle = branch_harness.wait_for_idle();
    tokio::pin!(branch_idle);
    tokio::select! {
        _ = &mut branch_idle => panic!("branch wait_for_idle resolved before async listener finished"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
    }
    branch_release.notify_one();
    let (branch_result, ()) = tokio::join!(branch_future, branch_idle);
    let branch_result = branch_result
        .expect("branch result")
        .expect("branch summary");
    assert_eq!(branch_result.summary, "Async listener branch summary");
    assert!(branch_finished.load(Ordering::SeqCst));
    assert_eq!(branch_harness.phase(), AgentHarnessPhase::Idle);
}

#[tokio::test]
async fn agent_harness_runs_tool_call_and_tool_result_hooks_through_direct_loop() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let mut args = Map::new();
    args.insert("expression".to_owned(), Value::String("2 + 2".to_owned()));
    registration.set_responses(vec![
        faux_assistant_message(
            faux_tool_call("calculate", args, Some("call-1".to_owned())),
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let session = Session::new(InMemorySessionStorage::new());
    let mut options =
        AgentHarnessOptions::new(test_env(), session.clone(), registration.get_model());
    options.tools = vec![calculate_tool()];
    let harness = AgentHarness::new(options);
    let seen_tool_calls = Arc::new(Mutex::new(Vec::new()));
    let seen_tool_calls_ref = seen_tool_calls.clone();
    harness.on_tool_call(move |event| {
        seen_tool_calls_ref.lock().expect("mutex").push((
            event.tool_call_id,
            event.tool_name,
            event.input["expression"].clone(),
        ));
        Ok(None)
    });
    harness.on_tool_result(|event| {
        assert_eq!(event.tool_call_id, "call-1");
        assert_eq!(event.tool_name, "calculate");
        assert_eq!(event.input["expression"], Value::String("2 + 2".to_owned()));
        assert!(!event.is_error);
        Ok(Some(ToolResultPatch {
            content: Some(vec![AgentToolResultContent::Text(TextContent::new(
                "patched result",
            ))]),
            details: Some(json!({ "patched": true })),
            terminate: Some(true),
        }))
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *seen_tool_calls.lock().expect("mutex"),
        vec![(
            "call-1".to_owned(),
            "calculate".to_owned(),
            Value::String("2 + 2".to_owned())
        )]
    );
    let tool_result = session
        .entries()
        .into_iter()
        .find_map(|entry| match entry {
            SessionTreeEntry::Message {
                message: Message::ToolResult(result),
                ..
            } => Some(result),
            _ => None,
        })
        .expect("tool result");
    assert!(matches!(
        tool_result.content.first(),
        Some(ToolResultContent::Text(text)) if text.text == "patched result"
    ));
    assert_eq!(tool_result.details, Some(json!({ "patched": true })));
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_refreshes_model_thinking_and_active_tools_at_prepare_next_turn() {
    let mut first_model = FauxModelDefinition::new("first");
    first_model.reasoning = true;
    let mut second_model = FauxModelDefinition::new("second");
    second_model.reasoning = true;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![first_model, second_model],
        ..Default::default()
    });
    let second_model = registration
        .get_model_by_id("second")
        .expect("second model");
    let captured = Arc::new(Mutex::new(Vec::<(
        String,
        Option<ThinkingLevel>,
        String,
        Vec<String>,
    )>::new()));
    let first_capture = captured.clone();
    let second_capture = captured.clone();
    let mut args = Map::new();
    args.insert("expression".to_owned(), Value::String("1 + 1".to_owned()));
    registration.set_responses(vec![
        faux_response_factory(move |context, options, _, model| {
            first_capture.lock().expect("mutex").push((
                model.id.clone(),
                options.reasoning,
                context.system_prompt.clone().unwrap_or_default(),
                context.tools.iter().map(|tool| tool.name.clone()).collect(),
            ));
            faux_assistant_message(
                faux_tool_call("calculate", args.clone(), Some("call-1".to_owned())),
                FauxAssistantOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
        }),
        faux_response_factory(move |context, options, _, model| {
            second_capture.lock().expect("mutex").push((
                model.id.clone(),
                options.reasoning,
                context.system_prompt.clone().unwrap_or_default(),
                context.tools.iter().map(|tool| tool.name.clone()).collect(),
            ));
            faux_assistant_message("done", Default::default())
        }),
    ]);
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.tools = vec![calculate_tool()];
    options.resources = AgentHarnessResources {
        skills: vec![Skill {
            name: "prompt".to_owned(),
            description: "prompt".to_owned(),
            content: "first prompt".to_owned(),
            file_path: "/skills/prompt/SKILL.md".to_owned(),
            source: None,
            disable_model_invocation: false,
        }],
        prompt_templates: Vec::new(),
    };
    options.system_prompt_provider = Some(Arc::new(|context: SystemPromptContext| {
        Ok(context
            .resources
            .skills
            .first()
            .map(|skill| skill.content.clone())
            .unwrap_or_else(|| "missing prompt".to_owned()))
    }));
    let harness = Arc::new(AgentHarness::new(options));
    let hook_harness = harness.clone();
    harness.on_tool_call(move |_| {
        hook_harness
            .set_model(second_model.clone())
            .expect("set model");
        hook_harness
            .set_thinking_level(ThinkingLevel::High)
            .expect("set thinking");
        hook_harness
            .set_tools(
                vec![calculate_tool(), clock_tool()],
                Some(vec!["clock".to_owned()]),
            )
            .expect("set tools");
        Ok(None)
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *captured.lock().expect("mutex"),
        vec![
            (
                "first".to_owned(),
                None,
                "first prompt".to_owned(),
                vec!["calculate".to_owned()]
            ),
            (
                "second".to_owned(),
                Some(ThinkingLevel::High),
                "first prompt".to_owned(),
                vec!["clock".to_owned()]
            ),
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_listener_updates_are_live_before_prepare_next_turn() {
    let mut first_model = FauxModelDefinition::new("first-live");
    first_model.reasoning = true;
    let mut second_model = FauxModelDefinition::new("second-live");
    second_model.reasoning = true;
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![first_model, second_model],
        ..Default::default()
    });
    let second_model = registration
        .get_model_by_id("second-live")
        .expect("second model");
    let captured = Arc::new(Mutex::new(Vec::<(
        String,
        Option<ThinkingLevel>,
        String,
        Vec<String>,
    )>::new()));
    let first_capture = captured.clone();
    let second_capture = captured.clone();
    let mut args = Map::new();
    args.insert("expression".to_owned(), Value::String("1 + 1".to_owned()));
    registration.set_responses(vec![
        faux_response_factory(move |context, options, _, model| {
            first_capture.lock().expect("mutex").push((
                model.id.clone(),
                options.reasoning,
                context.system_prompt.clone().unwrap_or_default(),
                context.tools.iter().map(|tool| tool.name.clone()).collect(),
            ));
            faux_assistant_message(
                faux_tool_call("calculate", args.clone(), Some("call-live".to_owned())),
                FauxAssistantOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
        }),
        faux_response_factory(move |context, options, _, model| {
            second_capture.lock().expect("mutex").push((
                model.id.clone(),
                options.reasoning,
                context.system_prompt.clone().unwrap_or_default(),
                context.tools.iter().map(|tool| tool.name.clone()).collect(),
            ));
            faux_assistant_message("done", Default::default())
        }),
    ]);
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.tools = vec![calculate_tool()];
    options.resources = AgentHarnessResources {
        skills: vec![Skill {
            name: "prompt".to_owned(),
            description: "prompt".to_owned(),
            content: "first prompt".to_owned(),
            file_path: "/skills/prompt/SKILL.md".to_owned(),
            source: None,
            disable_model_invocation: false,
        }],
        prompt_templates: Vec::new(),
    };
    options.system_prompt_provider = Some(Arc::new(|context: SystemPromptContext| {
        Ok(context
            .resources
            .skills
            .first()
            .map(|skill| skill.content.clone())
            .unwrap_or_else(|| "missing prompt".to_owned()))
    }));
    let harness = Arc::new(AgentHarness::new(options));
    let listener_harness = harness.clone();
    harness.subscribe(move |event| {
        if matches!(
            event,
            AgentHarnessEvent::Agent(AgentEvent::ToolExecutionStart { .. })
        ) {
            listener_harness
                .set_model(second_model.clone())
                .expect("set model");
            listener_harness
                .set_thinking_level(ThinkingLevel::High)
                .expect("set thinking");
            listener_harness.set_resources(AgentHarnessResources {
                skills: vec![Skill {
                    name: "prompt".to_owned(),
                    description: "prompt".to_owned(),
                    content: "second prompt".to_owned(),
                    file_path: "/skills/prompt/SKILL.md".to_owned(),
                    source: None,
                    disable_model_invocation: false,
                }],
                prompt_templates: Vec::new(),
            });
            listener_harness
                .set_tools(
                    vec![calculate_tool(), clock_tool()],
                    Some(vec!["clock".to_owned()]),
                )
                .expect("set tools");
        }
    });

    harness.prompt("hello").await.expect("prompt");

    assert_eq!(
        *captured.lock().expect("mutex"),
        vec![
            (
                "first-live".to_owned(),
                None,
                "first prompt".to_owned(),
                vec!["calculate".to_owned()]
            ),
            (
                "second-live".to_owned(),
                Some(ThinkingLevel::High),
                "second prompt".to_owned(),
                vec!["clock".to_owned()]
            ),
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_drains_steering_one_at_a_time_and_emits_queue_updates() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    let user_counts = Arc::new(Mutex::new(Vec::new()));
    let first_counts = user_counts.clone();
    let second_counts = user_counts.clone();
    let third_counts = user_counts.clone();
    registration.set_responses(vec![
        faux_response_factory(move |context, _, _, _| {
            first_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("abcdefghij", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            second_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("second", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            third_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("third", Default::default())
        }),
    ]);
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.steering_mode = QueueMode::OneAtATime;
    let harness = Arc::new(AgentHarness::new(options));
    let steer_lengths = Arc::new(Mutex::new(Vec::new()));
    let steer_lengths_ref = steer_lengths.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::QueueUpdate(update) = event {
            steer_lengths_ref
                .lock()
                .expect("mutex")
                .push(update.steer.len());
        }
    });

    let running = harness.clone();
    let prompt_task = tokio::spawn(async move { running.prompt("hello").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness.steer("one").expect("steer one");
    harness.steer("two").expect("steer two");
    prompt_task.await.expect("prompt task").expect("prompt");

    assert_eq!(*user_counts.lock().expect("mutex"), vec![1, 2, 3]);
    assert_eq!(*steer_lengths.lock().expect("mutex"), vec![1, 2, 1, 0]);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_drains_follow_up_one_at_a_time_after_agent_would_stop() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    let user_counts = Arc::new(Mutex::new(Vec::new()));
    let first_counts = user_counts.clone();
    let second_counts = user_counts.clone();
    let third_counts = user_counts.clone();
    registration.set_responses(vec![
        faux_response_factory(move |context, _, _, _| {
            first_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("abcdefghij", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            second_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("second", Default::default())
        }),
        faux_response_factory(move |context, _, _, _| {
            third_counts
                .lock()
                .expect("mutex")
                .push(user_texts(&context.messages).len());
            faux_assistant_message("third", Default::default())
        }),
    ]);
    let mut options = AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    );
    options.follow_up_mode = QueueMode::OneAtATime;
    let harness = Arc::new(AgentHarness::new(options));
    let follow_up_lengths = Arc::new(Mutex::new(Vec::new()));
    let follow_up_lengths_ref = follow_up_lengths.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::QueueUpdate(update) = event {
            follow_up_lengths_ref
                .lock()
                .expect("mutex")
                .push(update.follow_up.len());
        }
    });

    let running = harness.clone();
    let prompt_task = tokio::spawn(async move { running.prompt("hello").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness.follow_up("one").expect("follow up one");
    harness.follow_up("two").expect("follow up two");
    prompt_task.await.expect("prompt task").expect("prompt");

    assert_eq!(*user_counts.lock().expect("mutex"), vec![1, 2, 3]);
    assert_eq!(*follow_up_lengths.lock().expect("mutex"), vec![1, 2, 1, 0]);
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_abort_clears_steer_and_follow_up_but_preserves_next_turn() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    let second_request_text = Arc::new(Mutex::new(Vec::new()));
    let second_request_text_ref = second_request_text.clone();
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
        faux_response_factory(move |context, _, _, _| {
            second_request_text_ref
                .lock()
                .expect("mutex")
                .extend(user_texts(&context.messages));
            faux_assistant_message("second", Default::default())
        }),
    ]);
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    )));
    let queue_updates = Arc::new(Mutex::new(Vec::new()));
    let queue_updates_ref = queue_updates.clone();
    harness.subscribe(move |event| {
        if let AgentHarnessEvent::QueueUpdate(update) = event {
            queue_updates_ref.lock().expect("mutex").push((
                update.steer.len(),
                update.follow_up.len(),
                update.next_turn.len(),
            ));
        }
    });

    let running = harness.clone();
    let first_prompt = tokio::spawn(async move { running.prompt("first").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness.steer("steer").expect("steer");
    harness.follow_up("follow").expect("follow up");
    harness.next_turn("next");
    let abort_result = harness.abort();
    let first = first_prompt
        .await
        .expect("first prompt")
        .expect("first prompt");
    harness.prompt("second").await.expect("second prompt");

    assert_eq!(first.stop_reason, StopReason::Aborted);
    assert_eq!(abort_result.cleared_steer.len(), 1);
    assert_eq!(abort_result.cleared_follow_up.len(), 1);
    assert!(queue_updates.lock().expect("mutex").contains(&(0, 0, 1)));
    assert_eq!(
        *second_request_text.lock().expect("mutex"),
        vec!["first".to_owned(), "next".to_owned(), "second".to_owned()]
    );
    registration.unregister();
}

#[tokio::test]
async fn agent_harness_abort_and_wait_emits_abort_after_idle_and_waits_for_listener() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(30.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
    ]);
    let harness = Arc::new(AgentHarness::new(AgentHarnessOptions::new(
        test_env(),
        Session::new(InMemorySessionStorage::new()),
        registration.get_model(),
    )));
    let order = Arc::new(Mutex::new(Vec::new()));
    let abort_started = Arc::new(tokio::sync::Notify::new());
    let abort_release = Arc::new(tokio::sync::Notify::new());
    let order_ref = order.clone();
    let abort_started_ref = abort_started.clone();
    let abort_release_ref = abort_release.clone();
    harness.subscribe_async(move |event| {
        let order = order_ref.clone();
        let abort_started = abort_started_ref.clone();
        let abort_release = abort_release_ref.clone();
        async move {
            match event {
                AgentHarnessEvent::Agent(AgentEvent::AgentEnd { .. }) => {
                    order.lock().expect("mutex").push("agent_end");
                }
                AgentHarnessEvent::Abort(_) => {
                    order.lock().expect("mutex").push("abort");
                    abort_started.notify_waiters();
                    abort_release.notified().await;
                    order.lock().expect("mutex").push("abort_listener_done");
                }
                _ => {}
            }
        }
    });

    let running = harness.clone();
    let first_prompt = tokio::spawn(async move { running.prompt("first").await });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    harness.steer("steer").expect("steer");
    harness.follow_up("follow").expect("follow up");

    let aborting = harness.clone();
    let mut abort_task = tokio::spawn(async move { aborting.abort_and_wait().await });
    abort_started.notified().await;
    assert_eq!(*order.lock().expect("mutex"), vec!["agent_end", "abort"]);
    tokio::select! {
        result = &mut abort_task => panic!("abort_and_wait resolved before async abort listener finished: {result:?}"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(30)) => {}
    }

    abort_release.notify_waiters();
    let abort_result = abort_task.await.expect("abort task");
    let first = first_prompt
        .await
        .expect("first prompt")
        .expect("first prompt");

    assert_eq!(first.stop_reason, StopReason::Aborted);
    assert_eq!(abort_result.cleared_steer.len(), 1);
    assert_eq!(abort_result.cleared_follow_up.len(), 1);
    assert_eq!(
        *order.lock().expect("mutex"),
        vec!["agent_end", "abort", "abort_listener_done"]
    );
    registration.unregister();
}
