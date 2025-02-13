import numpy as np
from copy import deepcopy
INERTIA = 0.0       # inertia unused 0.0 <----> 1.0 fixed
DECAY_RATE = 200    # objects independent 100 <-------> 0.0 everything move together

class adj():
    def __init__(self, T=np.array([.0, .0, .0]), S=np.array([.0, .0, .0]), R=np.array([.0]), o=None, call=True):
        self.T, self.S, self.R = deepcopy(T), deepcopy(S), deepcopy(R)
        #self.t, self.s, self.r = deepcopy(T), deepcopy(S), deepcopy(R)
        self.o,self.v = o,False
        _ = self() if call else None

    def __str__(self):
        return "%s id=%d, T:(%.3f,%.3f,%.3f), S:(%.3f,%.3f,%.3f), R:(%.3f)" % (self.o.class_name()[:10], self.o.idx, self.T[0], self.T[1], self.T[2], self.S[0], self.S[1], self.S[2], self.R[0])

    def __getitem__(self, k):
        assert k in ["T", "S", "R"]
        return self.T if k == "T" else (self.S if k == "S" else self.R)

    def __call__(self, rt=1.0):
        self.o.translation += self.T * rt
        self.o.size += self.S * rt
        self.o.orientation += self.R * rt

    def __add__(self, a):
        return adj(self.T + a.T, self.S + a.S, self.R + a.R, self.o, call=False)

    def __sub__(self, a):
        return self.Normed() @ a.Normed()
    
    def clear(self):
        self.T, self.S, self.R = np.array([.0, .0, .0]), np.array([.0, .0, .0]), np.array([.0])
        #self.t, self.s, self.r = np.array([.0, .0, .0]), np.array([.0, .0, .0]), np.array([.0])

    def Flat(self):
        return np.concatenate([self.T, self.S, self.R])

    def Norm(self):
        return max(np.linalg.norm(self.Flat()),1e-8)

    def Normed(self):
        return self.Flat() / self.Norm()

    # def flat(self):
    #     return np.concatenate([self.t, self.s, self.r])

    # def norm(self):
    #     return max(np.linalg.norm(self.flat()),1e-8)

    # def normed(self):
    #     return self.flat() / self.norm()

    def dct(self):
        return {"T": np.round(self.T, 4).tolist(), "S": np.round(self.S, 4).tolist(), "R": np.round(self.R, 4).tolist()}

    def update(self, T, S, R, inertia=INERTIA, v=True, call=True): #v = True: this modification can be updated with inertia; v = False: this modification should be covered by the next one
        i = 0#inertia * int(self.v) # set inertia to 0 if it's the first time of optimzation
        self.T, self.S, self.R, self.v = i*self.T+(1-i)*T, i*self.S+(1-i)*S, i*self.R+(1-i)*R, v
        #self.t, self.s, self.r = deepcopy(T), deepcopy(S), deepcopy(R)
        _ = self() if call else None

    def toward(self, o, rt, inertia=INERTIA, v=True,call=True):
        from ...Basic import angleNorm
        self.update(T=(o.translation-self.o.translation)*rt, S=(o.size-self.o.size)*rt,
                    R=np.array([angleNorm((o.orientation-self.o.orientation)[0])])*rt, inertia=inertia, v=v, call=call)

class adjs():
    def __init__(self, objes, lst=None):
        self.objes = objes
        if lst is None:
            self.adjusts = [adj(o.adjust.T, o.adjust.S, o.adjust.R, o, call=False) for o in objes]
        else:
            assert len(lst) == len(objes)
            self.adjusts = lst

    def snapshot(self):
        return adjs(self.objes, [adj(a.T, a.S, a.R, a.o, call=False) for a in self.adjusts])
    
    def __str__(self):
        return "\n".join([str(a) for a in self.adjusts])
        
    def __add__(self, os):
        return adjs(self.objes, [a + b for a, b in zip(self.adjusts, os.adjusts)])

    def __sub__(self, os):
        return sum([a - b for a, b in zip(self.adjusts, os.adjusts)])

    def __call__(self, rt=1.0):
        [a(rt) for a in self.adjusts]

    def flat(self):
        return np.concatenate([a.flat() for a in self.adjusts])

    def Norm(self):
        return sum([a.Norm() for a in self.adjusts]) / len(self.adjusts)

    def Normed(self):
        return self.flat() / np.linalg.Norm(self.flat())

    def dct(self):
        return [a.dct() for a in self.adjusts]

    def apply_influence(self):
        if DECAY_RATE < 100:
            raise NotImplementedError("viscosity is not used right now")
            buffer = [np.array([adj_j.flat()*np.exp(-DECAY_RATE*np.linalg.norm(adj_i.o.translation-adj_j.o.translation)) for j, adj_j in enumerate(self.adjusts) if i != j]).mean(axis=0) for i, adj_i in enumerate(self.adjusts)]
            [adj.update(buffer[i][:3], buffer[i][3:6], buffer[i][6:]) for i, adj in enumerate(self.adjusts)]
