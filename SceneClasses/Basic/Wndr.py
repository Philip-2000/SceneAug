import numpy as np
from numpy.linalg import norm
class wndr():
    def __init__(self, center=None, size=None, ori=None, wls=None, wid=None, rate=None, width=None, height=None, y=None):
        if center is not None:
            ws = [w for w in wls if w.on(center,0.2)]
            assert len(ws)==1
            self.w = ws[0]
            assert (np.abs(([ori[0],0,ori[1]] if len(ori)==2 or ori.shape == (2,) else [np.cos(ori[0]),0,np.sin(ori[0])]) - w.n)).sum() < 0.001
            self.height= 2*size[1]
            self.width = 2*max(size[0],size[2]) #FIXME: this might not be correct for not vertical or horizontal wall
            self.rate  = self.w.rate(center)
            self.center= np.array([0,center[1],0])+self.rate*self.w.q+(1-self.rate)*self.w.p
        else:
            self.w = wls[wid]
            self.rate  = rate
            self.width = width
            self.height= height
            self.center= np.array([0,y,0])+rate*self.w.q+(1-rate)*self.w.p
            raise NotImplementedError

    def field(self,sp,config):
        return np.array([0,0,0])

class widw(wndr): #暂时没啥用
    def __init__(self, **kwargs):
        super(wndr,self).__init__(**kwargs)
        assert self.center[1]-self.height/2.0 > 0.1
        
class door(wndr):
    def __init__(self, **kwargs):
        super(wndr,self).__init__(**kwargs)
        assert self.center[1]-self.height/2.0 < 0.1
        self.optArea, self.LI, self.RI = None, None, None

    def getOptArea(self,config):
        if self.optArea is None:
            from shapely.geometry import Polygon
            WI = np.cross(self.w.n,np.array([0,1,0]))*self.width*0.5*config["expand"]
            LO = self.center - WI - self.w.n*config["out"]
            RO = self.center + WI - self.w.n*config["out"]
            LI = self.center - WI + self.w.n*config["in"]
            RI = self.center + WI + self.w.n*config["in"]
            self.optArea, self.LI, self.RI = Polygon([[LO[0],LO[2]],[LI[0],LI[2]],[RI[0],RI[2]],[RO[0],RO[2]]]), LI, RI

    def optField(self,sp,config):
        self.getOptArea(config)
        from shapely.geometry import Point
        return min(self.LI - sp.transl, self.RI-sp.transl, key=lambda x:norm(x)) if self.optArea.contains(Point(sp.transl[0],sp.transl[2])) else np.array([0,0,0])
    
class wndrs():
    def __init__(self,wls,cont=None,hint=None,jsn=None):
        if cont is not None:
            self.WNDRS = [door(center=c[:3],size=c[3:5],ori=c[6:],wls=wls) if abs(c[1]-c[4])<0.1 else widw(center=c[:3],size=c[3:5],ori=c[6:],wls=wls) for c in cont]
        elif jsn is not None:
            raise NotImplementedError
            self.WNDRS = []#其实不应该这样的，应该去设那些bbox啥的#door(center=obj["translate"],size=obj["scale"],ori=obj["rotate"],wls=wls) if obj["coarseSemantic"].lower()=="door" else widw(center=c[:3],size=c[3:5],ori=c[6:],wls=wls) for obj in jsn]
        else:
            raise NotImplementedError
            self.WNDRS = [door(wls=wls,**h) if abs(c[1]-c[4])<0.1 else widw(wls=wls,**h) for h in hint] #wid, rate, width, height, y

    def optFields(self,sp,config):
        return np.array([wr.optField(sp,config) for wr in self.WNDRS]).sum(axis=0)