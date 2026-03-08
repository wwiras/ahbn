
class EWMA:
    def __init__(self,alpha=0.5,initial=0.0):
        self.alpha=alpha
        self.value=initial
    def update(self,x):
        self.value=self.alpha*x+(1-self.alpha)*self.value
        return self.value
