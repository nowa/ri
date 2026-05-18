use async_trait::async_trait;
use ri_agent_core::*;
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
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

fn test_env() -> LocalExecutionEnv {
    LocalExecutionEnv::new("/tmp")
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
                    .map(|skill| skill.name.clone()),
                update
                    .previous_resources
                    .skills
                    .first()
                    .map(|skill| skill.name.clone()),
            ));
        }
    });
    let resources = AgentHarnessResources {
        skills: vec![Skill {
            name: "inspect".to_owned(),
            description: "Inspect things".to_owned(),
            content: "Use inspection tools.".to_owned(),
            file_path: "/skills/inspect/SKILL.md".to_owned(),
            disable_model_invocation: false,
        }],
        prompt_templates: vec![PromptTemplate {
            name: "review".to_owned(),
            description: "Review".to_owned(),
            content: "Review $1".to_owned(),
        }],
    };

    harness.set_resources(resources.clone());
    harness.set_resources(resources.clone());
    let resolved = harness.get_resources();

    assert_eq!(resolved, resources);
    assert_eq!(
        *updates.lock().expect("mutex"),
        vec![
            (Some("inspect".to_owned()), None),
            (Some("inspect".to_owned()), Some("inspect".to_owned()))
        ]
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
        disable_model_invocation: false,
    };
    let template = PromptTemplate {
        name: "review".to_owned(),
        description: "Review".to_owned(),
        content: "Review $1 with $2".to_owned(),
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
