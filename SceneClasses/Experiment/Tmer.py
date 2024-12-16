import time,numpy as np

class evnt():
    def __init__(self,on=None):
        self.start = time.time() if on else None #self.name = name
        self.on = bool(on)
        self.record = []
        self.last = 0
        
    def __call__(self,op=None):
        if op is None:
            op = not self.on
        if bool(op) != self.on:
            self.on = bool(op)
            if self.on:
                self.start = time.time()
            else:
                end = time.time()
                self.record.append((self.start, end))
                r = (end - self.start)
                self.last += r
                self.start = None
                return r
            
    def clear(self):
        self.record = []
    
class tmer():
    def __init__(self):
        self.evnts = {}
    
    def __getitem__(self,name):
        return self.evnts[name]

    def __call__(self,name,on=None):
        if name in self.evnts:
            self.evnts[name](on)
        else:
            self.evnts[name] = evnt(on)

    def sum(self):
        return np.array([self.evnts[e].last for e in self.evnts]).sum()
        
    def clear(self):
        [self.evnts[e].clear() for e in self.evnts if e != "accum"]
    
    def dct(self):
        return {e: self.evnts[e].last for e in self.evnts}
                
class tme(): #a fake class for timer, doing nothing
    def __init__(self):
        return
    
    def __getitem__(self,name):
        return

    def __call__(self,name,on=None):
        return
    
    def sum(self):
        return 0
        
    def clear(self):
        return
    
    def dct(self):
        return {}