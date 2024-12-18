import numpy as np
from numpy.linalg import norm
class samp():
    def __init__(self,s,debug=True):
        assert np.abs(s[0])==1 or np.abs(s[2])==1
        self.TRANSL = np.array(s)
        self.debug  = debug
        self.component = {}
        self.t, self.s, self.r = np.array([.0,.0,.0]), np.array([.0,.0,.0]), np.array([.0])

    def __update(self,o):
        self.transl = o + self.TRANSL
        self.radial = self.transl - o.translation#self.tangen = np.cross(self.radial/norm(self.radial),[0,1,0])

    def __call__(self, o, config, timer):
        self.__update(o)
        timer("wfield",1)
        wo, wi, dr = o.scne.WALLS.optFields(self,config)
        timer("wfield",0)
        timer("ofield",1)
        ob = o.scne.OBJES.optFields(self, o, config["object"]) if o.class_name().find("Lamp") < 0 else np.array([.0,.0,.0])#
        timer("ofield",0)
        timer("syn",1)
        self.t = wo+wi+dr+ob
        self.s = np.dot(self.t,self.radial/norm(self.radial))*self.TRANSL
        self.r = np.cross(self.t,self.radial)[1:2]/norm(self.radial) #self.tangen
        timer("syn",0)
        if self.debug:
            self.component["al"], self.component["wo"], self.component["wi"], self.component["dr"], self.component["ob"] = self.t, wo, wi, dr, ob

    def draw(self,colors={}):
        assert self.debug
        from matplotlib import pyplot as plt
        st = [self.transl[0], self.transl[2]]
        for k in colors:
            ed = [st[0]+self.component[k][0], st[1]+self.component[k][2]]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], marker=".", color=colors[k])
            st = [ed[0],ed[1]]
        plt.plot( [self.transl[0],self.transl[0]], [-self.transl[2],-self.transl[2]], marker=".", color=(0.33,0.33,0.33))

class samps():
    def __init__(self,o,s4,debug=True):#assert sr%4 == 0 np.max(np.abs(np.array(ss)[:,0].sum()),np.abs(np.array(ss)[:,2].sum())) < 0.000001
        ss = [[1.,.0,1.-i*2.0/s4] for i in range(s4)] + [[1.-i*2.0/s4,.0,-1.] for i in range(s4)] + [[-1.,.0,-1.+i*2.0/s4] for i in range(s4)] + [[-1.+i*2.0/s4,.0,1.] for i in range(s4)]
        self.o, self.samps, self.debug = o, [samp(s,debug) for s in ss], debug

    def __iter__(self):
        return iter(self.samps)
    
    def __call__(self, config, timer, ut=-1):
        [s(self.o,config,timer) for s in self]
        timer("syn",1)
        T,S,R = np.average([_.t for _ in self],axis=0)*config["syn"]["T"], np.average([_.s for _ in self],axis=0)*config["syn"]["S"], np.average([_.r for _ in self],axis=0)*config["syn"]["R"]
        if ut>0:
            self.o.adjust.update(T*ut,S*ut,R*ut)#self.o.adjust()
        timer("syn",0)
        return T, S, R
    
    def draw(self,way,colors):
        assert self.debug
        [s.draw(colors) for s in self]

    def violate(self):
        return np.array([norm(s.t) for s in self]).sum() #????