use crate::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: SessionId,
    pub message_count: usize,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(id: SessionId) -> Self {
        Self {
            id,
            messages: Vec::new(),
        }
    }

    pub fn is_over_budget(&self, limit: usize) -> bool {
        self.messages.iter().map(|m| m.token_count).sum::<usize>() > limit
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}
