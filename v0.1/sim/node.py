
from cache import MessageCache
from ewma import EWMA
from controller import AdaptiveController

class Node:
    def __init__(self,node_id,cluster_id,role,neighbors,cluster_head,gateway_peers,mode_setting):
        self.node_id=node_id
        self.cluster_id=cluster_id
        self.role=role
        self.neighbors=neighbors
        self.cluster_head=cluster_head
        self.gateway_peers=gateway_peers
        self.mode_setting=mode_setting

        self.cache=MessageCache()
        self.duplicate_ewma=EWMA()
        self.controller=AdaptiveController()

        self.duplicate_count=0
        self.first_receive_time=None

    def select_mode(self,sender):
        if self.mode_setting=="gossip":
            return 0,"gossip"
        if self.mode_setting=="cluster":
            return 1,"cluster"
        return self.controller.choose_mode(self,sender)

    def gossip_targets(self,sender):
        return [n for n in self.neighbors if n!=sender]

    def cluster_targets(self,sender,topology):
        if self.role=="CH":
            t=[]
            for m in topology["clusters"][self.cluster_id]:
                if m!=self.node_id and m!=sender:
                    t.append(m)
            for g in self.gateway_peers:
                if g!=sender and g not in t:
                    t.append(g)
            return t
        if self.cluster_head and self.cluster_head!=sender:
            return [self.cluster_head]
        return []

    def receive(self,time,msg,sender,topology):
        seen=self.cache.has_seen(msg.message_id)

        if seen:
            self.duplicate_count+=1
            return []

        self.cache.mark_seen(msg.message_id)

        if self.first_receive_time is None:
            self.first_receive_time=time

        _,mode=self.select_mode(sender)

        if mode=="gossip":
            targets=self.gossip_targets(sender)
        else:
            targets=self.cluster_targets(sender,topology)

        events=[]
        proc=topology["processing_delay"].get(self.node_id,0)

        for t in targets:
            # delay=topology["link_delays"][self.node_id][t]+proc
            base_delay = topology["link_delays"].get(self.node_id, {}).get(t, 1.0)
            delay = base_delay + proc
            fwd=msg.copy_for_forward(self.node_id)
            events.append((time+delay,t,fwd,self.node_id))

        return events
