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
        self.o = o
        self.transl = o + self.TRANSL
        self.radial = self.transl - o.translation

    def __call__(self, o, config, timer):
        self.__update(o)
        wo, wi, dr = o.scne.WALLS.optFields(self,o,config)
        wi = wi if config["object"][o.class_name()][-1] else np.zeros_like(wi)
        ob = o.scne.OBJES.optFields(self, o, config["object"]) if o.class_name().find("Lamp") < 0 else np.array([.0,.0,.0])#
        self.t = wo+wi+dr+ob
        self.s = np.dot(self.t,self.radial/norm(self.radial))*np.abs(self.TRANSL)
        self.r =-np.cross(self.t,self.radial)[1:2]/norm(self.radial) #self.tangen
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

    def toSampleJson(self):
        return {"t":np.round(self.t,5).tolist()}
    
    def fromSampleJson(self,js,o):
        self.__update(o)
        self.t = np.array(js["t"])
class samps():
    def __init__(self,o,s4,debug=True):#assert sr%4 == 0 np.max(np.abs(np.array(ss)[:,0].sum()),np.abs(np.array(ss)[:,2].sum())) < 0.000001
        ss = [[1.,.0,1.-i*2.0/s4] for i in range(s4)] + [[1.-i*2.0/s4,.0,-1.] for i in range(s4)] + [[-1.,.0,-1.+i*2.0/s4] for i in range(s4)] + [[-1.+i*2.0/s4,.0,1.] for i in range(s4)]
        self.o, self.samps, self.debug = o, [samp(s,debug) for s in ss], debug

    def __iter__(self):
        return iter(self.samps)
    
    def __call__(self, config, timer, ut=-1):
        timer("fld")
        [s(self.o,config,timer) for s in self]
        timer("fld")
        timer("syn")
        #T,S,R = np.average([_.t for _ in self],axis=0)*config["syn"]["T"], np.average([_.s for _ in self],axis=0)*config["syn"]["S"], np.average([_.r for _ in self],axis=0)*config["syn"]["R"]
        T = np.sum([_.t*norm(_.t) for _ in self],axis=0)/np.sum([norm(_.t) for _ in self]+[1e-6]) *config["syn"]["T"]
        S = np.average([_.s for _ in self],axis=0)*config["syn"]["S"]
        R = np.average([_.r for _ in self],axis=0)*config["syn"]["R"]
        if ut>-1e-5:
            self.o.adjust.update(T*ut,S*ut,R*ut)#self.o.adjust()
        timer("syn")
        return T, S, R
    
    def toSamplesJson(self):
        return [s.toSampleJson() for s in self]

    @classmethod
    def fromSamplesJson(self,o,js):
        a = samps(o, len(js)//4)
        [s.fromSampleJson(js[i],o) for i,s in enumerate(a)]
        return a
    
    def renderables(self, scene_render):
        bound = 1e-2
        if len([s for s in self if norm(s.t)>bound]) == 0: return
        vertices=np.array([ [.2,.0,.0],[ .2,.0,.8],[-.2,.0, .8], 
                            [.2,.0,.0],[-.2,.0,.8],[-.2,.0, .0],
                            [.4,.0,.8],[-.4,.0,.8],[ .0,.0,1.0],]) 
        from simple_3dviz import Mesh
        arrow = Mesh(vertices=vertices, normals=np.array([[.0,1.,.0]]*len(vertices)), colors=np.array([[.6, .6,.6]]*len(vertices)))
        
        mean_t = np.mean([np.abs(s.t) for s in self],axis=0)
        arrow.scale(np.array([norm(mean_t)]*3))
        arrow.rotate_y(np.arctan2(mean_t[0],mean_t[2]))
        mean_transl = np.mean([np.abs(s.transl) for s in self if norm(s.t)>bound],axis=0)
        mean_transl[1] = 3.0
        arrow.affine_transform(t=mean_transl)
        scene_render.add(arrow)

    def draw(self,way,colors):
        assert self.debug
        [s.draw(colors) for s in self]

    def violate(self):
        return np.array([norm(s.t) for s in self]).mean() #????