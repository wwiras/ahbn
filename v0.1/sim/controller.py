
class AdaptiveController:
    def __init__(self,threshold=0.5):
        self.threshold=threshold
    def choose_mode(self,node,sender):
        if node.role=="CH":
            return 0.8,"cluster"
        if node.cluster_head and sender==node.cluster_head:
            return 0.8,"cluster"
        if node.duplicate_ewma.value>=self.threshold:
            return node.duplicate_ewma.value,"cluster"
        return node.duplicate_ewma.value,"gossip"
