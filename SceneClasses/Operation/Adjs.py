import numpy as np
class adj():
    def __init__(self,T=np.array([.0,.0,.0]),S=np.array([.0,.0,.0]),R=np.array([.0]),o=None):
        from copy import deepcopy
        self.T,self.S,self.R= deepcopy(T),deepcopy(S),deepcopy(R)
        self.o = o

    def __getitem__(self,k):
        assert k in ["T","S","R"]
        return self.T if k=="T" else (self.S if k=="S" else self.R)
        
    def __call__(self,rt=1.0):        
        self.o.translation += self.T*rt
        self.o.size += self.S*rt
        self.o.orientation += self.R*rt

    def __add__(self,a):
        return adj(T=self.T+a.T,S=self.S+a.S,R=self.R+a.R,o=self.o)
    
    def __sub__(self,a):
        return self.normed() @ a.normed()
    
    def flat(self):
        return np.concatenate([self.T,self.S,self.R])

    def norm(self):
        return np.linalg.norm(self.flat())

    def normed(self):
        return self.flat()/np.linalg.norm(self.flat())


    def dct(self):
        return {"T":np.round(self.T,4).tolist(),"S":np.round(self.S,4).tolist(),"R":np.round(self.R,4).tolist()}
    
class adjs():
    def __init__(self,objes,lst=None):
        if lst is None:
            self.adjusts = [adj(o.adjust.T,o.adjust.S,o.adjust.R,o) for o in objes]
            self.objes = objes
        else:
            assert len(lst) == len(objes)
            self.adjusts = lst
            self.objes = objes

    def __add__(self,os):
        assert len(os) == len(self)
        return adjs(self.objes, [self[i]+os[i] for i in range(len(self))])

    def __sub__(self,os):
        assert len(os) == len(self)
        return sum([self[i]-os[i] for i in range(len(self))]) / float(len(self))
    
    def __len__(self):
        return len(self.adjusts)
    
    def __getitem__(self,i):
        return self.adjusts[i]

    def __iter__(self):
        return iter(self.adjusts)
    
    def dct(self):
        return [a.dct() for a in self]
