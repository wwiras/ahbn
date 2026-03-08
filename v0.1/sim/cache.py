
class MessageCache:
    def __init__(self):
        self.seen=set()
    def has_seen(self,message_id):
        return message_id in self.seen
    def mark_seen(self,message_id):
        self.seen.add(message_id)
