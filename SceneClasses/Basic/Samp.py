import numpy as np
class samp():
    def __init__(self,o,s,debug=True):
        assert abs(s[0])==1 or abs(s[2])==1
        self.TRANSL = s
        self.transl = o + s
        self.radial = self.transl - o.translation
        self.radian = self.radial/np.linalg.norm(self.radial)
        self.tangen = np.cross(self.radian,[0,1,0])
        self.debug  = debug
        if debug:
            self.F, self.WO, self.WI, self.DR, self.OB = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0])
        self.t, self.s, self.r = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])

    def __call__(self, o, config):
        wo, wi, dr = o.scne.WALLS.optFields(self,config)
        ob = o.scne.OBJES.optFields(self, o, config["object"])
        self.t = wo+wi+dr+ob
        self.s = np.dot(self.F,self.radian)*self.TRANSL
        self.r = np.cross(self.F,self.tangen)[1]/np.linalg.norm(self.radial)
        if self.debug:
            self.F, self.WO, self.WI, self.DR, self.OB = self.t, wo, wi, dr, ob

class samps():
    def __init__(self,o,ss,debug=True):
        assert ss.sum() < 0.000001
        self.o, self.samps = o, [samp(o,s,debug) for s in ss]

    def __call__(self, config, ut=False):
        [s(self.o,config) for s in self.samps]
        T = np.average([_.t for _ in self.samps],axis=0)
        S = np.average([_.s for _ in self.samps],axis=0)
        R = np.average([_.r for _ in self.samps],axis=0)
        if ut:
            self.o.translation, self.o.size, self.o.orientation += T, S, R
        return T, S, R
    