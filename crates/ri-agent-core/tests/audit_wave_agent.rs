//! Test-parity audit wave: pi `agent.test.ts` "should ignore a settled
//! parallel tool update while another tool is still running".

use async_trait::async_trait;
use ri_agent_core::*;
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::sync::{Arc, Mutex};

fn agent_context() -> AgentContext {
    AgentContext {
        system_prompt: "You are helpful.".to_owned(),
        messages: Vec::new(),
        tools: Vec::new(),
    }
}

fn tool(name: &str, label: &str, executor: Arc<dyn AgentToolExecutor>) -> AgentTool {
    AgentTool {
        definition: Tool {
            name: name.to_owned(),
            description: format!("The {name} tool"),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        label: label.to_owned(),
        execution_mode: None,
        argument_preparer: None,
        executor,
    }
}

/// Settles immediately but leaks its update callback for a late call.
struct SettledToolExecutor {
    slot: Arc<Mutex<Option<AgentToolUpdateCallback>>>,
}

#[async_trait]
impl AgentToolExecutor for SettledToolExecutor {
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
        _params: Value,
        on_update: AgentToolUpdateCallback,
    ) -> Result<AgentToolResult, String> {
        *self.slot.lock().expect("slot") = Some(on_update);
        Ok(AgentToolResult::text("done"))
    }
}

/// Keeps the agent run active until the test releases it.
struct SlowToolExecutor {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl AgentToolExecutor for SlowToolExecutor {
    async fn execute(
        &self,
        _tool_call_id: &str,
        _params: Value,
    ) -> Result<AgentToolResult, String> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(AgentToolResult::text("done"))
    }
}

/// Records every event the loop emits, in real time.
struct RecordingSink {
    events: Arc<Mutex<Vec<AgentEvent>>>,
}

#[async_trait]
impl AgentEventSink for RecordingSink {
    async fn on_event(&self, event: &AgentEvent) {
        self.events.lock().expect("events").push(event.clone());
    }
}

/// Tool-result hooks run after the per-invocation update guard flips, so this
/// signals deterministically that the settled tool no longer accepts updates.
struct SettledNotifyHook {
    tool_call_id: &'static str,
    settled: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl AgentToolResultHook for SettledNotifyHook {
    async fn on_tool_result(
        &self,
        context: AgentToolResultHookContext,
    ) -> Result<Option<AgentToolResultHookResult>, String> {
        if context.tool_call_id == self.tool_call_id {
            self.settled.notify_one();
        }
        Ok(None)
    }
}

#[tokio::test]
async fn agent_loop_ignores_settled_parallel_tool_update_while_another_tool_runs() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call("settled_tool", Map::new(), Some("call-1".to_owned())),
                faux_tool_call("slow_tool", Map::new(), Some("call-2".to_owned())),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message("done", Default::default()).into(),
    ]);

    let slot = Arc::new(Mutex::new(None::<AgentToolUpdateCallback>));
    let slow_started = Arc::new(tokio::sync::Notify::new());
    let release_slow = Arc::new(tokio::sync::Notify::new());
    let settled_tool_settled = Arc::new(tokio::sync::Notify::new());
    let sink_events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));

    let mut context = agent_context();
    context.tools.push(tool(
        "settled_tool",
        "Settled Tool",
        Arc::new(SettledToolExecutor { slot: slot.clone() }),
    ));
    context.tools.push(tool(
        "slow_tool",
        "Slow Tool",
        Arc::new(SlowToolExecutor {
            started: slow_started.clone(),
            release: release_slow.clone(),
        }),
    ));

    let mut config = AgentLoopConfig::new(registration.get_model());
    config.event_sink = Some(Arc::new(RecordingSink {
        events: sink_events.clone(),
    }));
    config.tool_result_hooks.push(Arc::new(SettledNotifyHook {
        tool_call_id: "call-1",
        settled: settled_tool_settled.clone(),
    }));

    let loop_task = tokio::spawn(agent_loop_prompt(context, "run tools", config));

    // The settled tool has resolved (its update guard flipped) while the slow
    // tool still keeps the run active.
    slow_started.notified().await;
    settled_tool_settled.notified().await;
    let events_before_late_update = sink_events.lock().expect("events").len();

    // A late update from the settled tool must be ignored: no event reaches
    // the sink while the run is still in flight.
    let late_callback = slot
        .lock()
        .expect("slot")
        .clone()
        .expect("captured update callback");
    late_callback(AgentToolResult::text("late")).await;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(
        sink_events.lock().expect("events").len(),
        events_before_late_update
    );

    release_slow.notify_one();
    let (messages, events) = loop_task.await.expect("join").expect("loop");

    // Neither the live sink nor the collected event log saw a tool update.
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
            .count(),
        0
    );
    assert!(
        sink_events
            .lock()
            .expect("events")
            .iter()
            .all(|event| !matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
    );

    // The run itself completed normally with both tool results.
    let tool_result_ids = messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::ToolResult(result) => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_result_ids, vec!["call-1", "call-2"]);
    registration.unregister();
}
