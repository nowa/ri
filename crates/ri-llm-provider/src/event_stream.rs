use crate::types::{AssistantMessage, AssistantMessageEvent};
use futures::Stream;
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Clone)]
pub struct AssistantMessageEventSender {
    state: Arc<Mutex<AssistantMessageEventSenderState>>,
}

struct AssistantMessageEventSenderState {
    sender: Option<mpsc::UnboundedSender<AssistantMessageEvent>>,
    result_sender: Option<oneshot::Sender<AssistantMessage>>,
    done: bool,
}

impl AssistantMessageEventSender {
    pub fn push(&self, event: AssistantMessageEvent) {
        let final_message = event.final_message();
        let mut state = self
            .state
            .lock()
            .expect("event stream sender mutex poisoned");
        if state.done {
            return;
        }
        if let Some(message) = final_message.clone() {
            state.done = true;
            if let Some(sender) = state.result_sender.take() {
                let _ = sender.send(message);
            }
        }
        if let Some(sender) = state.sender.as_ref() {
            let _ = sender.send(event);
        }
        if final_message.is_some() {
            state.sender.take();
        }
    }

    pub fn end(&self, result: AssistantMessage) {
        let mut state = self
            .state
            .lock()
            .expect("event stream sender mutex poisoned");
        if state.done {
            return;
        }
        state.done = true;
        if let Some(sender) = state.result_sender.take() {
            let _ = sender.send(result);
        }
        state.sender.take();
    }
}

pub struct AssistantMessageEventStream {
    receiver: UnboundedReceiverStream<AssistantMessageEvent>,
    result_receiver: oneshot::Receiver<AssistantMessage>,
}

impl AssistantMessageEventStream {
    pub async fn try_result(self) -> Result<AssistantMessage, String> {
        self.result_receiver
            .await
            .map_err(|_| "assistant message stream closed without a final result".to_owned())
    }

    pub async fn result(self) -> AssistantMessage {
        self.try_result()
            .await
            .expect("assistant message stream closed without a final result")
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

pub fn assistant_message_event_stream() -> (AssistantMessageEventSender, AssistantMessageEventStream)
{
    let (sender, receiver) = mpsc::unbounded_channel();
    let (result_sender, result_receiver) = oneshot::channel();
    (
        AssistantMessageEventSender {
            state: Arc::new(Mutex::new(AssistantMessageEventSenderState {
                sender: Some(sender),
                result_sender: Some(result_sender),
                done: false,
            })),
        },
        AssistantMessageEventStream {
            receiver: UnboundedReceiverStream::new(receiver),
            result_receiver,
        },
    )
}
