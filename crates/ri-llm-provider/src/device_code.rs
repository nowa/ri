//! Shared OAuth 2.0 Device Authorization Grant polling (RFC 8628).
//!
//! Mirrors pi's `pollOAuthDeviceCodeFlow`: the poll cadence is a pure state
//! machine (deadline, current interval, slow_down bookkeeping) so drivers can
//! run it against real timers while tests drive it with virtual time.

use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub const DEVICE_FLOW_CANCEL_MESSAGE: &str = "Login cancelled";
pub const DEVICE_FLOW_TIMEOUT_MESSAGE: &str = "Device flow timed out";
pub const DEVICE_FLOW_SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";

const MINIMUM_INTERVAL_MS: u64 = 1000;
/// RFC 8628 section 3.2: if the authorization server omits `interval`, the
/// client must use 5 seconds.
const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 5;
/// RFC 8628 section 3.5: `slow_down` means the polling interval must increase
/// by 5 seconds.
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

/// One token-endpoint poll result, as classified by the provider layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCodePollResponse<T> {
    Pending,
    SlowDown { interval_seconds: Option<u64> },
    Complete(T),
    Failed { message: String },
}

/// What the driver should do after applying a poll response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCodePollProgress<T> {
    Continue,
    Complete(T),
    Failed { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCodePollConfig {
    pub interval_seconds: Option<u64>,
    pub expires_in_seconds: Option<u64>,
    /// GitHub answers a poll that lands inside the throttle window with
    /// `slow_down`, so callers like Copilot wait one interval before the
    /// first poll instead of polling immediately.
    pub wait_before_first_poll: bool,
}

impl Default for DeviceCodePollConfig {
    fn default() -> Self {
        Self {
            interval_seconds: None,
            expires_in_seconds: None,
            wait_before_first_poll: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCodePollState {
    deadline_ms: Option<i64>,
    interval_ms: u64,
    slow_down_responses: u32,
    wait_before_first_poll: bool,
}

impl DeviceCodePollState {
    pub fn new(now_ms: i64, config: &DeviceCodePollConfig) -> Self {
        Self {
            deadline_ms: config
                .expires_in_seconds
                .map(|seconds| now_ms + seconds as i64 * 1000),
            interval_ms: config
                .interval_seconds
                .unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS)
                .saturating_mul(1000)
                .max(MINIMUM_INTERVAL_MS),
            slow_down_responses: 0,
            wait_before_first_poll: config.wait_before_first_poll,
        }
    }

    /// Delay to apply before the first poll, if the flow opted into waiting.
    pub fn initial_wait_ms(&self, now_ms: i64) -> Option<u64> {
        if !self.wait_before_first_poll {
            return None;
        }
        match self.remaining_ms(now_ms) {
            Some(remaining) if remaining <= 0 => None,
            Some(remaining) => Some(self.interval_ms.min(remaining as u64)),
            None => Some(self.interval_ms),
        }
    }

    /// Whether the device code is still within its lifetime.
    pub fn should_poll(&self, now_ms: i64) -> bool {
        self.remaining_ms(now_ms)
            .is_none_or(|remaining| remaining > 0)
    }

    pub fn apply_response<T>(
        &mut self,
        response: DeviceCodePollResponse<T>,
    ) -> DeviceCodePollProgress<T> {
        match response {
            DeviceCodePollResponse::Pending => DeviceCodePollProgress::Continue,
            DeviceCodePollResponse::SlowDown { interval_seconds } => {
                self.slow_down_responses += 1;
                // Use the server-provided interval when given (GitHub reports
                // the new required minimum in `interval`); trusting only a
                // client-tracked value risks polling early forever under
                // WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5:
                // increase by 5 seconds.
                self.interval_ms = interval_seconds
                    .filter(|interval| *interval > 0)
                    .map(|interval| interval.saturating_mul(1000).max(MINIMUM_INTERVAL_MS))
                    .unwrap_or_else(|| {
                        (self.interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS)
                            .max(MINIMUM_INTERVAL_MS)
                    });
                DeviceCodePollProgress::Continue
            }
            DeviceCodePollResponse::Complete(value) => DeviceCodePollProgress::Complete(value),
            DeviceCodePollResponse::Failed { message } => {
                DeviceCodePollProgress::Failed { message }
            }
        }
    }

    /// Delay before the next poll, or `None` once the lifetime is spent.
    pub fn wait_after_poll_ms(&self, now_ms: i64) -> Option<u64> {
        match self.remaining_ms(now_ms) {
            Some(remaining) if remaining <= 0 => None,
            Some(remaining) => Some(self.interval_ms.min(remaining as u64)),
            None => Some(self.interval_ms),
        }
    }

    pub fn slow_down_seen(&self) -> bool {
        self.slow_down_responses > 0
    }

    pub fn timeout_message(&self) -> String {
        device_flow_timeout_message(self.slow_down_seen())
    }

    fn remaining_ms(&self, now_ms: i64) -> Option<i64> {
        self.deadline_ms.map(|deadline| deadline - now_ms)
    }
}

pub fn device_flow_timeout_message(slow_down_seen: bool) -> String {
    if slow_down_seen {
        DEVICE_FLOW_SLOW_DOWN_TIMEOUT_MESSAGE.to_owned()
    } else {
        DEVICE_FLOW_TIMEOUT_MESSAGE.to_owned()
    }
}

/// Drive a device-code flow to completion with an injected sleeper.
///
/// Time is virtual: the driver advances `now_ms` by exactly the delays it
/// hands to `sleep`, so deterministic tests can pass a recording sleeper.
pub async fn poll_device_code_flow_with_sleeper<T, P, PFut, S, SFut>(
    config: &DeviceCodePollConfig,
    start_ms: i64,
    poll: P,
    sleep: S,
) -> Result<T, String>
where
    P: FnMut() -> PFut,
    PFut: Future<Output = Result<DeviceCodePollResponse<T>, String>>,
    S: FnMut(u64) -> SFut,
    SFut: Future<Output = ()>,
{
    poll_device_code_flow_with_sleeper_and_abort(config, start_ms, poll, sleep, None).await
}

/// [`poll_device_code_flow_with_sleeper`] with a cooperative abort flag.
///
/// Mirrors pi's `pollOAuthDeviceCodeFlow` abort signal: the flag is checked
/// before the initial wait, before every poll, and before every inter-poll
/// wait, failing with [`DEVICE_FLOW_CANCEL_MESSAGE`] once set.
pub async fn poll_device_code_flow_with_sleeper_and_abort<T, P, PFut, S, SFut>(
    config: &DeviceCodePollConfig,
    start_ms: i64,
    mut poll: P,
    mut sleep: S,
    abort_flag: Option<Arc<AtomicBool>>,
) -> Result<T, String>
where
    P: FnMut() -> PFut,
    PFut: Future<Output = Result<DeviceCodePollResponse<T>, String>>,
    S: FnMut(u64) -> SFut,
    SFut: Future<Output = ()>,
{
    let aborted = || -> bool {
        abort_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    };
    let mut state = DeviceCodePollState::new(start_ms, config);
    let mut now_ms = start_ms;
    if let Some(delay_ms) = state.initial_wait_ms(now_ms) {
        if aborted() {
            return Err(DEVICE_FLOW_CANCEL_MESSAGE.to_owned());
        }
        sleep(delay_ms).await;
        now_ms += delay_ms as i64;
    }
    loop {
        if !state.should_poll(now_ms) {
            return Err(state.timeout_message());
        }
        if aborted() {
            return Err(DEVICE_FLOW_CANCEL_MESSAGE.to_owned());
        }
        match state.apply_response(poll().await?) {
            DeviceCodePollProgress::Continue => {}
            DeviceCodePollProgress::Complete(value) => return Ok(value),
            DeviceCodePollProgress::Failed { message } => return Err(message),
        }
        match state.wait_after_poll_ms(now_ms) {
            Some(delay_ms) => {
                if aborted() {
                    return Err(DEVICE_FLOW_CANCEL_MESSAGE.to_owned());
                }
                sleep(delay_ms).await;
                now_ms += delay_ms as i64;
            }
            None => return Err(state.timeout_message()),
        }
    }
}
