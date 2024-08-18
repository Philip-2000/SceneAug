import numpy as np
from Obje import *
from Scne import *

DE = [0.9**2,0.5**2,0.9**2,0.9**2,0.5**2,0.9**2,0.5**2]
DEN = [DE]*10
SIGMA2 = 4.0
MDE = True
def giveup(B,A,C):
    return (B is None) or (len(B) < 100) #or (len(B)/C < 0.3) or (len(B)/A < 0.1)

class bnch():
    def __init__(self,obj,exp=None,dev=None):
        self.exp = exp if obj is None else obj.flat()
        self.dev = dev if obj is None else np.array(DE)
        self.obs = []  if obj is None else [obj]
    
    def __len__(self):
        return len(self.obs)

    def loss(self,obj):
        return ((obj.flat()-self.exp)**2/self.dev).sum()

    def add(self,obj,f=False):
        if self.accept(obj) or f:
            self.exp = (self.exp * len(self.obs) + obj.flat()) / (len(self.obs)+1)
            self.obs.append(obj)
            self.dev = np.average(np.array([(o.flat()-self.exp)**2 for o in self.obs]+DEN),axis=0) if MDE else self.dev
            return True
        return False

    def accept(self,obj):#,add=True):
        return np.min(self.dev*SIGMA2 - (obj.flat()-self.exp)**2) > 0
    
    def refresh(self):
        self.exp = np.average(np.array([o.flat() for o in self.obs]),axis=0) if len(self.obs) > 0 else self.exp
        self.dev = np.average(np.array([(o.flat()-self.exp)**2 for o in self.obs]+DEN),axis=0) if MDE and (len(self.obs) > 0) else self.dev
        return len(self.obs) > 0

    def enable(self,nid):
        for o in self.obs:#if o.scne.OBJES[o.idx].nid != -1: print(o.scne.scene_uid+" "+str(o.idx)+" "+o.scne.OBJES[o.idx].class_name()) #assert 1 == 0
            o.scne.OBJES[o.idx].nid = nid

class bnches():
    def __init__(self):
        self.bunches = []

    def __len__(self):
        return len(self.bunches)

    def mx(self):
        return None if len(self.bunches)==0 else sorted(self.bunches,key=lambda x:-len(x))[0]

    def all(self):
        return sum([len(b) for b in self.bunches])
    
    def refresh(self):
        obs = []
        for b in self.bunches:
            obs += b.obs
            b.obs = [] 
        for o in obs:
            for b in self.bunches:
                if b.add(o):
                    break
        for b in self.bunches:
            if not b.refresh():
                self.bunches.remove(b)

    def accept(self,obj,create=True):
        for b in range(len(self.bunches)):
            if self.bunches[b].add(obj):
                return b
        if create:
            self.bunches.append(bnch(obj))
            return len(self.bunches)-1
        return -1
