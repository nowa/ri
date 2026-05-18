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
    sender: mpsc::UnboundedSender<AssistantMessageEvent>,
    result_sender: Arc<Mutex<Option<oneshot::Sender<AssistantMessage>>>>,
}

impl AssistantMessageEventSender {
    pub fn push(&self, event: AssistantMessageEvent) {
        if let Some(message) = event.final_message() {
            if let Some(sender) = self
                .result_sender
                .lock()
                .expect("result mutex poisoned")
                .take()
            {
                let _ = sender.send(message);
            }
        }
        let _ = self.sender.send(event);
    }

    pub fn end(&self, result: AssistantMessage) {
        if let Some(sender) = self
            .result_sender
            .lock()
            .expect("result mutex poisoned")
            .take()
        {
            let _ = sender.send(result);
        }
    }
}

pub struct AssistantMessageEventStream {
    receiver: UnboundedReceiverStream<AssistantMessageEvent>,
    result_receiver: oneshot::Receiver<AssistantMessage>,
}

impl AssistantMessageEventStream {
    pub async fn result(self) -> AssistantMessage {
        self.result_receiver
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
            sender,
            result_sender: Arc::new(Mutex::new(Some(result_sender))),
        },
        AssistantMessageEventStream {
            receiver: UnboundedReceiverStream::new(receiver),
            result_receiver,
        },
    )
}
