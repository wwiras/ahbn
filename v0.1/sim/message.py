
from dataclasses import dataclass
import uuid

@dataclass
class Message:
    message_id: str
    origin_id: str
    sender_id: str
    created_at: float
    hop_count: int
    payload: str

    @staticmethod
    def new(origin_id: str, payload: str):
        return Message(
            message_id=str(uuid.uuid4()),
            origin_id=origin_id,
            sender_id=origin_id,
            created_at=0.0,
            hop_count=0,
            payload=payload
        )

    def copy_for_forward(self, new_sender_id: str):
        return Message(
            message_id=self.message_id,
            origin_id=self.origin_id,
            sender_id=new_sender_id,
            created_at=self.created_at,
            hop_count=self.hop_count + 1,
            payload=self.payload
        )
