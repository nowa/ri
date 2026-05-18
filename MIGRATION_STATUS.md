# Migration Status

Objective: build Rust crates in this `ri` directory that correspond to
`pi-agent-core` and required dependencies such as `pi-ai`, named with the `ri-*`
pattern, and finish only when all `pi-agent-core` / `pi-ai` tests have Rust
counterparts that pass.

## Source Scope

- Source agent package: `/home/nowa/Projects/src/pi/packages/agent`
- Source AI package: `/home/nowa/Projects/src/pi/packages/ai`
- Source agent tests counted: 16 `*.test.ts` files, about 150 `it/test` cases.
- Source AI tests counted: 68 `*.test.ts` files, about 721 `it/test` cases.

## Rust Artifacts Created

- `ri-llm-provider`
  - Core message/content/model/event types.
  - API provider registry.
  - `stream`, `complete`, `stream_simple`, `complete_simple`.
  - Built-in model registry seed and thinking-level helpers.
  - Faux provider with queued responses, multi-model registrations,
    model-aware response factories, event deltas, terminal error/abort events,
    abort flags before and during paced streams, usage estimates, session cache
    simulation, and unregister behavior. Faux response factories can inspect
    `SimpleStreamOptions` for helper tests.
  - JSON repair/hash helpers, including malformed control-character repair and
    unpaired UTF-16 surrogate escape replacement.
  - JSON-schema subset validation and coercion.
  - Context-overflow detection with provider-specific error-shape corpus and
    silent/length-stop overflow signals.
  - Environment API key lookup.
  - HTTP proxy URL resolution with `NO_PROXY` handling and unsupported protocol errors.
  - Bedrock endpoint/region config helpers plus Converse payload helpers for
    message conversion, Claude thinking fields, GovCloud display omission, and
    application-inference-profile cache points, including image tool-result
    content, streamed Converse block aggregation, usage mapping, stop reasons,
    and SDK exception error events.
  - Azure OpenAI base URL/config normalization, default API version,
    deployment-name map resolution, and Responses payload construction with
    deployment overrides, tools, session cache keys, and reasoning options.
  - Google Vertex API-key/ADC/custom-base-URL client config helpers.
  - Google/Gemini shared tool conversion helpers with OpenAPI schema sanitization,
    message conversion, multimodal function response routing, thought-signature
    handling, `streamSimple` thinking payload construction/disable rules, and
    streamed chunk aggregation for response IDs, text/thinking signatures,
    function calls, usage, and safety/error finish reasons.
  - Fireworks and Together model metadata/base-URL/provider compatibility overrides.
  - OpenCode Zen/Go model metadata/base-URL/provider compatibility overrides.
  - Mistral chat payload helpers for tool schema serialization, image
    tool-result content, cross-provider tool-call ID normalization, missing
    tool-result synthesis, request header/session-affinity handling, stream
    chunk aggregation, response IDs, usage mapping, and reasoning mode selection.
  - Cross-provider message transform helpers: image downgrade, thinking cleanup,
    tool-call ID normalization, orphaned tool-result synthesis.
  - Anthropic raw SSE parsing helpers with malformed JSON repair, streamed
    tool-argument parsing, response IDs, usage mapping, partial usage
    preservation, and unknown event filtering.
  - Anthropic Messages payload helpers for eager tool input streaming compatibility,
    fine-grained streaming beta headers, thinking disable/adaptive payloads, and
    cache-control markers/TTL for system prompts, final user turns, and tools,
    assistant/tool-result replay with image tool-result content, including
    Fireworks tool-field compatibility omissions.
  - Anthropic client/header config helpers for GitHub Copilot Claude Bearer auth,
    Copilot static/dynamic headers, vision request detection, Fireworks
    session-affinity headers, and adaptive-model interleaved-thinking beta
    omission.
  - Anthropic Claude Code tool-name casing helpers.
  - Anthropic OAuth helpers for authorization URL construction, token/refresh
    JSON requests, localhost callback preservation, and token response parsing.
  - OpenAI Codex OAuth helpers for authorization URL construction,
    form-encoded token/refresh requests, and refresh failure message formatting.
  - OpenAI Codex Responses helpers for ChatGPT JWT account-id extraction,
    SSE/WebSocket headers, request-body construction, URL resolution, reasoning
    effort mapping, cached WebSocket input-delta continuation, SSE frame parsing
    and retry-delay calculation, WebSocket debug-stat accounting, and
    service-tier usage cost resolution.
  - GitHub Copilot OAuth helpers for device-flow request construction,
    slow-down-aware polling intervals, enterprise domain normalization,
    Copilot token refresh headers, and base-URL derivation.
  - OpenAI Responses stream and message conversion helpers for function-call
    partial JSON cleanup, foreign tool-call ID normalization, tool-result
    images, prompt-cache fields, session-affinity headers, default reasoning
    payload rules, function tools with strict defaults, text deltas/signature
    replay, incomplete terminal events, aborted reasoning history pruning,
    same-provider model handoff item-id omission, empty assistant-turn pruning,
    and service-tier usage cost multipliers.
  - OpenAI Completions payload helpers for empty-tools omission, tool choice,
    strict-mode compatibility, provider reasoning fields, z.ai tool streaming,
    Anthropic-style cache-control markers, tool-result image replay, and
    thinking-as-text replay, prompt-cache fields, session-affinity headers,
    empty user-block pruning, stream usage parsing, streamed text/thinking/tool
    delta aggregation, finish-reason mapping, and routed response model metadata.
  - Images API provider registry and `generate_images` dispatch, plus
    OpenRouter image model registry and image payload/response helpers for
    chat-completions image generation, inline data URLs, headers, usage, and
    aborted/error result mapping.
  - Usage helpers for enforcing `total_tokens == input + output + cache_read +
    cache_write` across provider usage parsers.

- `ri-agent-core`
  - Agent message/tool/state/event types.
  - Low-level prompt and continue agent loop with sequential tool execution,
    tool-result continuation turns, tool argument preparation, tool-call hooks
    with argument replacement, tool-result replacement hooks, hook-driven
    tool-result termination, tool-result message lifecycle events, and
    transform/convert hooks for LLM request context. Prepare-next-turn hooks can
    replace the next-turn context/model/thinking state after `turn_end`, and
    should-stop-after-turn hooks can stop before the next provider request.
    Queued messages can be injected before the initial provider request or
    after all tool-result messages are written and before the next LLM request,
    and follow-up messages can restart the loop after the agent would otherwise
    stop. Tool batches terminate only when every finalized result terminates.
    Parallel tool execution is the default, emits completion events in
    completion order while persisting tool-result messages in source order, and
    per-tool sequential mode forces sequential execution even under parallel
    loop config.
  - Stateful `Agent` wrapper with default/custom state initialization,
    thinking-level and session-id forwarding, subscriptions, text/image
    prompt normalization, prompt-message runs, source-compatible busy/continue
    validation errors, and steering/follow-up queues with one-at-a-time/all
    drain modes plus queued state introspection, queue clearing/reset helpers,
    abort handles that cancel active provider streams, and provider start
    failures persisted as assistant error messages with lifecycle events.
  - Harness basics: system prompt helper, system skill formatting, UTF-8/line
    truncation, token/usage estimate, compaction predicates, cut-point
    selection, compaction preparation, file-operation metadata extraction, and
    summarization conversation serialization/generation, compact execution, and
    branch-summary collection/preparation/generation, UUID v7 session id.
  - Minimal high-level `AgentHarness` foundation with env/session/model access,
    thinking-level and queue-mode state, subscriptions, queue update events,
    `before_agent_start` message/system-prompt hooks, `context` hooks with
    assistant-error persistence on hook failure, `tool_call`/`tool_result`
    hooks through the direct loop, fixed skill/prompt-template resources with
    `resources_update` events, direct skill and prompt-template invocation,
    stream-options accessors, model/thinking selection events with session
    persistence, listener-safe pending `append_message`, custom-entry,
    custom-message, label, and session-name writes, dynamic system-prompt
    providers over resources, live listener events and
    prepare-next-turn refresh for model/thinking/system prompt/resources/tools,
    `save_point` events for flushed pending writes, `next_turn`
    injection/persistence, tool/active-tool state management with request-time
    active-tool filtering, running-turn steering/follow-up queues, abort queue
    clearing that preserves next-turn messages, and idle waiting.
  - In-memory and JSONL session storage/repositories with branching, labels, metadata, and context building.
  - Skill and prompt-template invocation formatting plus skill/prompt-template
    loading, argument substitution, and skill ignore-file handling.
  - Local execution environment foundation for filesystem operations, shell
    execution, stdout/stderr callbacks, callback error propagation, command
    timeout/abort handling, pre-aborted file-operation cancellation, symlink
    directory entries, text-line read limits, shell spawn error mapping,
    best-effort cleanup, and shell-output capture/truncation with full-output
    log files.

## Rust Test Coverage Now

Current Rust tests: 276 passing.

- `crates/ri-llm-provider/tests/provider_core.rs`
  - `supports_xhigh_model_metadata_port`
  - `fireworks_and_together_model_metadata_match_provider_catalog`
  - `fireworks_and_together_env_keys_resolve_from_provider_specific_variables`
  - `cloudflare_model_metadata_and_base_url_resolution_match_provider_catalog`
  - `opencode_model_metadata_and_env_key_match_provider_catalog`
  - `openrouter_image_model_registry_matches_generated_catalog`
  - `openrouter_images_payload_uses_chat_completions_image_modalities`
  - `openrouter_images_payload_formats_image_input_and_image_only_output`
  - `openrouter_images_response_returns_text_images_response_id_and_usage`
  - `openrouter_images_usage_and_error_mapping_match_provider`
  - `images_api_registry_dispatches_generate_images_and_reports_missing_provider`
  - `mistral_payload_serializes_tool_schema_as_plain_json`
  - `mistral_simple_payload_selects_prompt_or_effort_reasoning_controls`
  - `mistral_payload_preserves_image_tool_results_for_vision_models`
  - `mistral_payload_synthesizes_missing_tool_results_and_normalizes_ids`
  - `mistral_request_headers_apply_session_affinity_without_overriding_callers`
  - `mistral_stream_chunks_preserve_response_id_usage_and_tool_calls`
  - `bedrock_model_registry_exposes_available_models`
  - `bedrock_endpoint_resolution_matches_region_and_profile_rules`
  - `bedrock_raw_message_conversion_skips_unknown_user_content_blocks`
  - `bedrock_raw_message_conversion_skips_unknown_assistant_content_blocks`
  - `bedrock_raw_message_conversion_skips_user_messages_with_only_unknown_blocks`
  - `bedrock_raw_message_conversion_skips_assistant_messages_with_only_unknown_blocks`
  - `bedrock_payload_uses_adaptive_thinking_for_claude_opus_47`
  - `bedrock_payload_maps_xhigh_to_native_opus_47_effort`
  - `bedrock_payload_omits_display_for_govcloud_nonadaptive_thinking`
  - `bedrock_payload_omits_display_for_govcloud_adaptive_region`
  - `bedrock_payload_uses_model_name_for_application_profile_adaptive_thinking`
  - `bedrock_payload_injects_cache_points_when_application_profile_name_supports_claude_cache`
  - `bedrock_payload_uses_model_name_for_application_profile_fixed_budget_thinking`
  - `bedrock_payload_preserves_image_tool_results_in_converse_messages`
  - `bedrock_stream_events_preserve_blocks_usage_and_stop_reason`
  - `bedrock_stream_events_format_exception_as_error_event`
  - `azure_openai_base_url_normalization_matches_provider_rules`
  - `azure_openai_config_builds_default_resource_url_from_env`
  - `azure_openai_deployment_name_prefers_option_env_map_then_model_id`
  - `azure_openai_responses_payload_uses_deployment_tools_session_and_reasoning`
  - `google_vertex_client_config_resolves_api_keys_adc_and_custom_base_urls`
  - `google_vertex_client_config_forwards_custom_base_url_to_api_key_client`
  - `google_convert_tools_strips_schema_meta_keys_for_parameters`
  - `google_convert_tools_recursively_strips_nested_schema_meta_keys`
  - `google_convert_tools_preserves_ref_when_stripping_meta_keys`
  - `google_convert_tools_does_not_mutate_original_parameters`
  - `google_convert_tools_preserves_schema_for_parameters_json_schema`
  - `google_convert_tools_handles_tools_without_schema_meta`
  - `google_convert_tools_returns_none_for_empty_tools`
  - `google_thinking_detection_uses_explicit_thought_marker_only`
  - `google_simple_payload_disables_thinking_for_gemini_reasoning_models`
  - `google_simple_payload_maps_reasoning_to_budget_or_level`
  - `google_retain_thought_signature_preserves_and_updates_non_empty_values`
  - `google_stream_chunks_preserve_response_id_signatures_usage_and_tool_calls`
  - `google_stream_chunks_map_safety_finish_to_error_event`
  - `google_convert_messages_keeps_separate_image_turn_for_gemini_2`
  - `google_convert_messages_nests_image_tool_results_for_gemini_3`
  - `google_convert_messages_omits_validator_marker_for_unsigned_gemini_3_tool_calls`
  - `google_convert_messages_omits_validator_marker_for_unsigned_vertex_tool_calls`
  - `google_convert_messages_preserves_valid_same_model_thought_signature`
  - `google_convert_messages_does_not_add_thought_signature_for_non_gemini_3_models`
  - `message_transform_normalizes_cross_provider_tool_call_ids`
  - `message_transform_copilot_openai_to_anthropic_downgrades_thinking_and_signatures`
  - `message_transform_synthesizes_only_missing_trailing_tool_results_after_normalization`
  - `message_transform_downgrades_images_thinking_and_orphaned_tool_calls`
  - `anthropic_sse_parser_repairs_malformed_event_and_streamed_tool_json`
  - `anthropic_sse_parser_ignores_unknown_events_after_message_stop`
  - `anthropic_sse_parser_preserves_response_id_and_initial_input_usage`
  - `anthropic_sse_parser_preserves_start_usage_when_delta_omits_fields`
  - `anthropic_payload_sends_per_tool_eager_input_streaming_by_default`
  - `anthropic_payload_uses_legacy_fine_grained_tool_streaming_beta_when_eager_disabled`
  - `anthropic_payload_omits_fine_grained_beta_when_no_tools`
  - `anthropic_payload_adds_short_cache_control_to_system_last_user_and_last_tool`
  - `anthropic_payload_sets_one_hour_cache_ttl_for_long_retention`
  - `anthropic_payload_omits_cache_control_for_none_and_ttl_when_unsupported`
  - `anthropic_payload_preserves_assistant_tool_use_and_image_tool_results`
  - `anthropic_simple_payload_disables_budget_reasoning_when_thinking_is_off`
  - `anthropic_simple_payload_disables_adaptive_reasoning_when_thinking_is_off`
  - `anthropic_simple_payload_uses_adaptive_thinking_for_opus_47`
  - `anthropic_simple_payload_maps_xhigh_to_opus_47_effort`
  - `anthropic_claude_code_tool_name_normalization_round_trips_known_tools`
  - `anthropic_oauth_authorize_url_uses_localhost_callback`
  - `anthropic_oauth_authorization_code_request_keeps_localhost_redirect_uri`
  - `anthropic_oauth_refresh_request_omits_scope`
  - `anthropic_oauth_token_response_maps_credentials_and_expiry`
  - `openai_codex_oauth_authorize_url_matches_cli_flow_parameters`
  - `openai_codex_oauth_refresh_request_uses_form_encoded_body`
  - `openai_codex_oauth_refresh_failure_message_includes_status_and_body`
  - `openai_codex_responses_extracts_account_id_and_builds_transport_headers`
  - `openai_codex_responses_omits_session_affinity_without_session_id`
  - `openai_codex_responses_resolves_urls`
  - `openai_codex_responses_payload_matches_request_body_defaults_and_session`
  - `openai_codex_responses_payload_maps_minimal_reasoning_to_low`
  - `openai_codex_responses_sse_parser_maps_text_and_terminal_statuses`
  - `openai_codex_responses_cached_websocket_request_sends_only_input_delta`
  - `openai_codex_responses_websocket_debug_stats_match_cached_request_accounting`
  - `openai_codex_responses_retry_delay_respects_headers_and_backoff`
  - `openai_codex_responses_usage_uses_client_tier_when_response_echoes_default`
  - `github_copilot_oauth_device_flow_requests_match_provider`
  - `github_copilot_oauth_poll_waits_before_first_poll_and_slows_down`
  - `github_copilot_oauth_poll_uses_remaining_lifetime_before_slow_down_timeout`
  - `github_copilot_oauth_refresh_and_base_url_helpers_match_provider`
  - `github_copilot_anthropic_client_config_matches_provider_headers`
  - `fireworks_anthropic_client_config_applies_session_affinity_rules`
  - `fireworks_anthropic_payload_applies_tool_compat_rules`
  - `openai_responses_stream_cleans_partial_json_from_tool_calls`
  - `openai_responses_message_conversion_hashes_foreign_tool_item_ids`
  - `openai_responses_message_conversion_keeps_tool_result_images_in_function_output`
  - `openai_responses_stream_maps_text_deltas_and_replays_text_signature`
  - `openai_responses_payload_sets_prompt_cache_fields_for_long_retention`
  - `openai_responses_payload_includes_function_tools_with_default_strict_false`
  - `openai_responses_payload_sets_long_retention_for_proxy_when_supported`
  - `openai_responses_payload_omits_cache_fields_when_retention_is_none`
  - `openai_responses_payload_omits_long_retention_when_compat_disables_it`
  - `openai_responses_default_headers_apply_session_affinity_and_overrides`
  - `openai_responses_payload_sends_default_none_reasoning_for_supported_openai_models`
  - `openai_responses_payload_omits_default_reasoning_when_off_is_unsupported`
  - `openai_responses_payload_omits_default_reasoning_for_github_copilot`
  - `openai_responses_payload_maps_explicit_reasoning_and_includes_encrypted_content`
  - `openai_responses_payload_skips_aborted_reasoning_only_history`
  - `openai_responses_payload_omits_function_call_item_id_for_same_provider_model_handoff`
  - `openai_responses_usage_applies_service_tier_cost_multiplier`
  - `openai_responses_stream_maps_response_incomplete_event_to_length`
  - `openai_completions_payload_omits_empty_tools_unless_tool_history_exists`
  - `openai_completions_payload_forwards_tool_choice_and_strict_compat`
  - `openai_completions_payload_maps_reasoning_and_zai_tool_stream_compat`
  - `openai_completions_payload_keeps_normal_groq_reasoning_effort_without_mapping`
  - `openai_completions_zai_tool_stream_metadata_override_and_no_tools_match_provider`
  - `openai_completions_cloudflare_gateway_compat_uses_conservative_payload_and_headers`
  - `openai_completions_payload_applies_anthropic_cache_control_format`
  - `openai_completions_messages_batch_tool_result_images_after_tool_results`
  - `openai_completions_messages_replay_thinking_as_text_parts`
  - `openai_completions_messages_replay_reasoning_signature_and_details`
  - `openai_completions_messages_add_empty_reasoning_content_when_required`
  - `openai_completions_payload_sets_prompt_cache_fields_for_direct_openai`
  - `openai_completions_payload_sets_long_retention_for_proxy_when_supported`
  - `openai_completions_payload_omits_proxy_cache_fields_and_uses_env_retention`
  - `openai_completions_default_headers_apply_session_affinity_and_overrides`
  - `openai_completions_chunk_usage_preserves_cache_write_tokens_and_totals`
  - `openai_completions_chunk_usage_does_not_double_count_reasoning_tokens`
  - `openai_completions_chunk_metadata_preserves_choice_usage_cache_write_tokens`
  - `openai_completions_stream_coalesces_tool_call_deltas_by_stable_index`
  - `openai_completions_stream_accumulates_mixed_deltas_independently`
  - `openai_completions_stream_normalizes_reasoning_field_by_provider`
  - `openai_completions_stream_ignores_null_chunks_and_finishes`
  - `openai_completions_stream_maps_finish_reason_errors_and_requires_terminal_reason`
  - `openai_completions_stream_attaches_reasoning_details_to_tool_calls`
  - `usage_total_tokens_match_components_for_provider_parsers`
  - `openai_completions_chunk_metadata_sets_response_model_without_changing_requested_model`
  - `faux_provider_registers_and_estimates_usage`
  - `faux_provider_supports_helper_blocks_and_stream_order`
  - `faux_provider_supports_multiple_models_factories_and_message_rewrites`
  - `faux_provider_replaces_appends_exhausts_and_unregisters`
  - `faux_provider_streams_multiple_tool_call_deltas`
  - `faux_provider_streams_terminal_error_and_aborted_messages`
  - `faux_provider_respects_abort_flag_before_and_during_streaming`
  - `faux_provider_consumes_responses_and_caches_per_session`
  - `validation_coerces_json_schema_primitives`
  - `validation_matches_ajv_style_plain_schema_coercions`
  - `env_api_keys_ignore_generic_github_tokens_for_copilot`
  - `empty_message_conversion_skips_empty_user_blocks_and_empty_assistant_turns`
  - `node_http_proxy_respects_no_proxy_and_rejects_unsupported_protocols`
  - `overflow_detects_error_and_silent_overflow_modes`
  - `overflow_matches_provider_error_shapes_without_rate_limit_false_positives`
  - `overflow_matches_context_overflow_provider_error_corpus`
  - `json_repair_and_hash_match_core_semantics`
  - `unicode_surrogate_repair_preserves_pairs_and_replaces_unpaired_escapes`

- `crates/ri-agent-core/tests/agent_core.rs`
  - `agent_loop_emits_lifecycle_events_and_messages`
  - `agent_loop_executes_tool_calls_and_appends_results`
  - `agent_loop_injects_queued_messages_before_initial_provider_request`
  - `agent_loop_injects_queued_messages_after_all_tool_results`
  - `agent_loop_uses_prepare_next_turn_snapshot_before_continuing`
  - `agent_loop_should_stop_after_current_turn_when_hook_returns_true`
  - `agent_loop_processes_follow_up_messages_after_agent_would_stop`
  - `agent_loop_stops_after_tool_batch_when_all_results_terminate`
  - `agent_loop_continues_after_parallel_tool_batch_when_not_all_results_terminate`
  - `agent_loop_prepares_tool_arguments_before_execution`
  - `agent_loop_tool_call_hook_can_replace_arguments_before_execution`
  - `agent_loop_parallel_tool_execution_emits_completion_order_and_persists_source_order`
  - `agent_loop_forces_sequential_execution_when_tool_requires_it`
  - `agent_loop_forces_sequential_execution_when_any_tool_requires_it`
  - `agent_loop_runs_parallel_tools_in_parallel_by_default`
  - `agent_loop_transforms_context_before_converting_to_llm`
  - `agent_loop_runs_tool_call_and_tool_result_hooks`
  - `agent_loop_tool_result_hook_can_terminate_tool_batch`
  - `agent_loop_continue_validates_context_tail`
  - `agent_loop_continue_from_existing_context_omits_existing_user_events`
  - `agent_stateful_wrapper_initializes_state_and_forwards_thinking_level`
  - `agent_stateful_wrapper_updates_state_and_notifies_subscribers`
  - `agent_stateful_wrapper_rejects_prompt_and_continue_while_streaming`
  - `agent_stateful_wrapper_validates_continue_tail_before_loop`
  - `agent_stateful_wrapper_forwards_session_id_to_provider_options`
  - `agent_prompt_with_images_builds_multimodal_user_message`
  - `agent_stateful_wrapper_persists_provider_start_failures_as_error_messages`
  - `agent_abort_handle_cancels_active_provider_stream`
  - `agent_queues_steering_and_follow_up_without_mutating_state`
  - `agent_has_queued_messages_tracks_steering_follow_up_and_clears`
  - `agent_prompt_injects_queued_steering_before_first_provider_request`
  - `agent_clear_queues_prevents_queued_messages_from_running`
  - `agent_reset_clears_state_and_queued_messages`
  - `agent_continue_from_assistant_tail_processes_queued_follow_up`
  - `agent_continue_from_assistant_tail_drains_steering_one_at_a_time`
  - `harness_utilities_cover_utf8_truncation_compaction_and_uuid`

- `crates/ri-agent-core/tests/agent_harness.rs`
  - `agent_harness_constructs_and_exposes_queue_modes`
  - `agent_harness_resources_getters_clone_and_emit_update_events`
  - `agent_harness_model_and_thinking_setters_emit_selection_events`
  - `agent_harness_tracks_tools_and_uses_only_active_tools_in_requests`
  - `agent_harness_injects_next_turn_messages_into_next_prompt`
  - `agent_harness_before_agent_start_appends_messages_and_persists_them`
  - `agent_harness_context_hook_failures_persist_assistant_error_messages`
  - `agent_harness_runs_tool_call_and_tool_result_hooks_through_direct_loop`
  - `agent_harness_drains_steering_one_at_a_time_and_emits_queue_updates`
  - `agent_harness_abort_clears_steer_and_follow_up_but_preserves_next_turn`

- `crates/ri-agent-core/tests/execution_env.rs`
  - `local_execution_env_reads_writes_lists_and_removes_files`
  - `local_execution_env_reports_symlinks_without_following_them`
  - `local_execution_env_lists_symlinks_as_symlinks`
  - `local_execution_env_read_text_lines_stops_at_requested_limit`
  - `local_execution_env_returns_file_errors_for_missing_and_wrong_kinds`
  - `local_execution_env_appends_creates_temps_and_removes_recursively`
  - `local_execution_env_executes_shell_commands_in_cwd_with_env`
  - `local_execution_env_returns_spawn_error_for_non_executable_shell`
  - `local_execution_env_cleanup_is_best_effort`
  - `local_execution_env_streams_stdout_and_stderr_callbacks`
  - `local_execution_env_returns_callback_errors_from_exec_handlers`
  - `local_execution_env_times_out_long_running_commands`
  - `local_execution_env_returns_aborted_for_aborted_commands`
  - `local_execution_env_returns_aborted_for_pre_aborted_file_operations`
  - `shell_capture_sanitizes_and_writes_large_output_file`

- `crates/ri-agent-core/tests/harness_compaction.rs`
  - `compaction_calculates_context_tokens_from_usage_and_thresholds`
  - `compaction_estimates_tokens_and_uses_latest_valid_assistant_usage`
  - `compaction_finds_cut_points_and_turn_start_edges`
  - `compaction_prepares_entries_previous_summary_split_turn_and_file_ops`
  - `compaction_prepares_custom_branch_messages_and_serializes_tool_results`
  - `compaction_generate_summary_builds_prompt_and_passes_reasoning_options`
  - `compaction_generate_summary_maps_error_and_aborted_results`
  - `compaction_compact_returns_summary_details_and_clamps_max_tokens`
  - `compaction_compact_summarizes_split_turn_and_maps_prefix_errors`
  - `branch_summary_collects_abandoned_branch_entries`
  - `branch_summary_prepares_messages_budget_and_file_ops`
  - `branch_summary_generate_builds_prompt_options_and_file_details`
  - `branch_summary_generate_replaces_prompt_and_maps_errors`

- `crates/ri-agent-core/tests/session_storage.rs`
  - `in_memory_storage_matches_core_storage_behaviour`
  - `in_memory_storage_walks_paths_to_root`
  - `in_memory_storage_finds_entries_by_type`
  - `jsonl_storage_writes_loads_metadata_entries_leaf_and_labels`
  - `jsonl_storage_rejects_missing_files_and_finds_entries_by_type`
  - `jsonl_storage_label_lookup_can_be_cleared_and_reloaded`
  - `jsonl_storage_rejects_malformed_headers_and_entries`
  - `session_supports_branching_for_memory_and_jsonl_storage`
  - `session_builds_context_tracks_model_thinking_and_moves_to_root_for_storage_kinds`
  - `session_move_with_branch_summary_appears_in_context_for_storage_kinds`
  - `session_builds_model_thinking_compaction_custom_and_branch_summary_context`
  - `session_labels_and_session_info_do_not_affect_context`
  - `jsonl_session_persists_leaf_entries_and_wire_entry_types`
  - `in_memory_repo_opens_deletes_and_forks_by_metadata`
  - `jsonl_repo_stores_lists_opens_deletes_and_forks_by_metadata`

- `crates/ri-agent-core/tests/resources.rs`
  - `formats_skill_and_prompt_template_invocations`
  - `formats_visible_skills_for_system_prompt`
  - `loads_skills_from_skill_files_and_root_markdown`
  - `loads_skills_through_symlinked_directories`
  - `load_skills_honors_ignore_files`
  - `sourced_skills_preserve_source_and_attach_diagnostics`
  - `loads_prompt_templates_non_recursively_from_dirs_and_files`
  - `loads_prompt_templates_from_symlinked_markdown_files`
  - `sourced_prompt_templates_preserve_source_and_attach_diagnostics`
  - `prompt_template_argument_substitution_matches_pi_placeholders`

- `crates/ri-agent-core/tests/harness_truncate.rs`
  - `truncate_counts_utf8_bytes_without_buffer_dependencies`
  - `truncate_head_uses_complete_utf8_lines_and_reports_first_line_overflow`
  - `truncate_tail_keeps_utf8_suffix_and_marks_partial_last_line`
  - `truncate_tail_matches_buffer_semantics_for_multibyte_edges`
  - `truncate_tail_matches_buffer_semantics_across_deterministic_fuzz_cases`
  - `truncate_line_and_format_size_match_harness_helpers`

Verified with:

```sh
cargo test --workspace
```

## Known Missing Work

This migration is not complete.

- Most `pi-ai` real provider implementations are not migrated yet:
  OpenAI Responses, OpenAI Completions, OpenAI Codex, Anthropic, Bedrock,
  Google/Gemini, Google Vertex, Mistral, Azure OpenAI Responses, remaining
  image API networking behavior, remaining OAuth providers and live OAuth flows,
  HTTP proxy agent construction, and provider-specific payload transforms.
- Most provider E2E behavior tests are not ported.
- `pi-agent-core` advanced loop behavior is incomplete:
  remaining high-level AgentHarness hooks/events beyond queue update basics,
  remaining before/after lifecycle hooks,
  remaining termination edge cases across abort/error paths, and async listener
  settlement.
- Harness/session implementation is only a foundation:
  higher-level harness integration for summary persistence/hooks remains to be ported.
- Test parity is far from complete: 276 Rust tests currently cover only the first
  slice of roughly 871 source test cases.
