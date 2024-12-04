import numpy as np
class samp():
    def __init__(self,o,s,debug=True):
        assert np.abs(s[0])==1 or np.abs(s[2])==1
        self.TRANSL = np.array(s)
        self.transl = o + s
        self.radial = self.transl - o.translation
        self.radian = self.radial/np.linalg.norm(self.radial)
        self.tangen = np.cross(self.radian,[0,1,0])
        self.debug  = debug
        self.component = {}
        self.t, self.s, self.r = np.array([.0,.0,.0]), np.array([.0,.0,.0]), np.array([.0,.0,.0])

    def __call__(self, o, config):
        wo, wi, dr = o.scne.WALLS.optFields(self,config)
        ob = o.scne.OBJES.optFields(self, o, config["object"]) #np.array([.0,.0,.0])#
        self.t = wo+wi+dr+ob
        self.s = np.dot(self.t,self.radian)*self.TRANSL
        self.r = np.cross(self.t,self.tangen)[1]/np.linalg.norm(self.radial)
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
    def __init__(self,o,ss,debug=True):
        assert np.max(np.abs(np.array(ss)[:,0].sum()),np.abs(np.array(ss)[:,2].sum())) < 0.000001
        self.o, self.samps, self.debug = o, [samp(o,s,debug) for s in ss], debug

    def __call__(self, config, ut=-1):
        [s(self.o,config) for s in self.samps]
        T = np.average([_.t for _ in self.samps],axis=0)
        S = np.average([_.s for _ in self.samps],axis=0)
        R = np.average([_.r for _ in self.samps],axis=0)
        if ut>0:
            if self.debug:
                self.o.adjust["T"], self.o.adjust["S"], self.o.adjust["R"] = T*ut,S*ut,R*ut
            self.o.translation, self.o.size, self.o.orientation = self.o.translation+T*ut, self.o.size+S*ut, self.o.orientation+R*ut
        return T, S, R
    
    def draw(self,way="pnt",colors={"al":(0,0,0)}):
        assert self.debug
        [s.draw(colors) for s in self.samps]