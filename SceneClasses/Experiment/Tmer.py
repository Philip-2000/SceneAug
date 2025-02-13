import time,numpy as np
#regulate our usage

class evnt():
    def __init__(self,on):
        self.base = time.time()
        self.start = self.base if on else None #self.name = name
        self.on = bool(on)
        self.record = [(0,0)] if not on else []
        
    def __call__(self,op): #to operate on the timer (open or close)
        #if op is None: op = not self.on
        assert bool(op) == (not self.on)
        self.on = bool(op)#print(self.on)
        if self.on: self.start = time.time()
        else: self.record.append((self.start-self.base, time.time()-self.base)) #self.start = None

    def __len__(self):
        return len(self.record)
            
    def __getitem__(self,idx): #to get a record from the timer
        return sum([r[1]-r[0] for r in self.record[:idx]])
    
    @property
    def last(self):
        return sum([r[1]-r[0] for r in self.record])

    def load(self, dct):
        self.record = dct["record"]

    def save(self):
        return {"record":[(round(r[0],6),round(r[1],6)) for r in self.record]}

    def clear(self):
        self.record,self.on = [],False
    
class tmer():
    def __init__(self):
        self.evnts = {} #"" preserved for the whole process
    
    def __getitem__(self,name):
        return self.evnts[name]

    def __call__(self,name,on): #print(name)
        if name in self.evnts: self.evnts[name](on)
        else: self.evnts[name] = evnt(on)
        
    def __len__(self):
        return len(self.evnts[""])

    def sum(self):
        return np.array([self.evnts[e].last for e in self.evnts]).sum()
        
    def clear(self):
        [self.evnts[e].clear() for e in self.evnts]
    
    def save(self):
        return {e:self.evnts[e].save() for e in self.evnts}

    def load(self,dct):
        for e in dct:
            self.evnts[e] = evnt(1)
            self.evnts[e].on = False
            self.evnts[e].load(dct[e])

# class tme(): #a fake class for timer, doing nothing
#     def __init__(self):
#         return
    
#     def __getitem__(self,name):
#         return

#     def __call__(self,name,on=None):
#         return
    
#     def sum(self):
#         return 0
        
#     def clear(self):
#         return
    
#     def dct(self):
#         return {}