import numpy as np
from numpy.linalg import norm
class wndr():
    def __init__(self, center=None, size=None, ori=None, c_e=None, wls=None, wid=None, rate=None, width=None, height=None, y=None, bbox=None):
        if center is not None: #from cont.npz
            center = center-c_e
            ws = [w for w in wls if w.on(center,0.2)]
            assert len(ws)==1
            self.w = ws[0]
            assert (np.abs(([ori[1],0,ori[0]] if len(ori)==2 or ori.shape == (2,) else [np.sin(ori[0]),0,np.cos(ori[0])]) - self.w.n)).sum() < 0.001
            self.height= 2*size[1]
            self.width = 2*max(size[0],size[2]) #FIXME: this might not be correct for not vertical or horizontal wall
            self.rate  = self.w.rate(center)
            self.center= np.array([0,center[1],0])+self.rate*self.w.q+(1-self.rate)*self.w.p
        elif bbox is not None: #from scene.json
            bmin = bbox["min"]
            bmax = bbox["max"]
            self.center = (np.array(bmin)+np.array(bmax)) / 2.0
            ws = [w for w in wls if w.on(self.center,0.2)]
            assert len(ws)==1
            self.w = ws[0]
            self.height= 2*size[1]
            self.width = 0 #FIXME: ?????????????????????????????
            self.rate  = self.w.rate(self.center)
            #
        else: #from a random hint 
            self.w = wls[wid]
            self.rate  = rate
            self.width = width
            self.height= height
            self.center= np.array([0,y,0])+rate*self.w.q+(1-rate)*self.w.p
            raise NotImplementedError
        from .Obje import bx2d
        self.block = bx2d(self.center,np.array([self.width/2.0,self.height/2.0,0.03]),np.array([np.math.atan2(self.w.n[0],self.w.n[2])]))

    def field(self,sp,config):
        return np.array([0,0,0])

    def toBlockJson(self): #真的太崩溃了。写了半天，暂时还是一个跑不了的状态。跑起来贼麻烦。
        bb = 0             #今天晚上肯定是没有心气去尝试前前后后的东西把他跑起来了还是先去做另一个项目吧
        bj = {
            "bbox":{
                "min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],
                "max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]
                },
            "translate":[float(self.center[0]),float(self.center[1]),float(self.center[2])],
            "scale":[1,1,1],
            "orient":float(self.orientation[0]),
            "rotate":[0,float(self.orientation[0]),0],
            "rotateOrder": "XYZ"}
        return bj
    
    def __str__(self):
        return "center:[%.3f,%.3f] with width=%.3f height=%.3f±%.3f on wall %d"%(self.center[0],self.center[2],self.width,self.center[1],self.height/2.0,self.w.idx) + (":[%.3f±%.3f,%.3f]"%(self.center[0],self.width/2.0,self.center[2]) if abs(self.w.n[2])<0.0001 else (":[%.3f,%.3f±%.3f]"%(self.center[0],self.center[2],self.width/2.0) if abs(self.w.n[0])<0.0001 else "") ) 

    def optField(self,sp,config):
        return np.array([0,0,0])

class widw(wndr):
    def __init__(self, **kwargs):
        super(widw,self).__init__(**kwargs)
        assert self.center[1]-self.height/2.0 > 0.1

    def toBlockJson(self,rid,idx):
        return {**(super(widw,self).toBlockJson()), "id":"window"+idx, "coarseSemantic":"window", "roomId":rid, "inDatabase":False}
        
    def __str__(self):
        return "widw  %s"%(super(widw,self).__str__())
    
    def draw(self):
        from matplotlib import pyplot as plt
        self.block.draw(color="black")
        A,a,B,b = self.block + np.array([1,0,-1]), self.block + np.array([0.2,0,-6]), self.block + np.array([-1,0,-1]), self.block + np.array([-0.2,0,-6])
        plt.plot( [A[0],a[0]], [-A[2],-a[2]], marker="*", color="black")
        plt.plot( [B[0],b[0]], [-B[2],-b[2]], marker="*", color="black")

class door(wndr):
    def __init__(self, **kwargs):
        super(door,self).__init__(**kwargs)
        assert self.center[1]-self.height/2.0 < 0.1
        self.optArea, self.LI, self.RI = None, None, None

    def toBlockJson(self,rid,idx):
        return {**(super(door,self).toBlockJson()), "id":"door"+idx, "coarseSemantic":"door", "roomId":rid, "inDatabase":False}
        
    def __str__(self):
        return "door  %s"%(super(door,self).__str__())
    
    def draw(self):
        from matplotlib import pyplot as plt
        self.block.draw(color="black")
        A,a = self.block + np.array([1,0,1]), self.block + np.array([-0.7,0,10])
        plt.plot( [A[0],a[0]], [-A[2],-a[2]], marker="*", color="black")

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
    def __init__(self,wls,cont=None,c_e=0):
        try: #for numpy.npz, those ori is in [z,x] and arctan is 
            self.WNDRS = [door(center=c[:3],size=c[3:6],ori=c[6:],c_e=c_e,wls=wls) if abs(c[1]-c[4])<0.1 else widw(center=c[:3],size=c[3:6],ori=c[6:],c_e=c_e,wls=wls) for c in cont]
        except:
            try:
                self.WNDRS = [door(wls=wls,bbox=j["bbox"]) if j["coarseSemantic"].lower() == "door" else widw(wls=wls,bbox=j["bbox"]) for j in cont]#其实不应该这样的，应该去设那些bbox啥的#door(center=obj["translate"],size=obj["scale"],ori=obj["rotate"],wls=wls) if obj["coarseSemantic"].lower()=="door" else widw(center=c[:3],size=c[3:5],ori=c[6:],wls=wls) for obj in jsn]
            except:
                self.WNDRS = [door(wls=wls,**h) if abs(h["y"]-h["height"]/2.0)<0.1 else widw(wls=wls,**h) for h in cont] #wid, rate, width, height, y

    def toBlocksJson(self,rid=-1):
        return []#[wd.toBlockJson(rid,str(idx)) for idx,wd in enumerate(self.WNDRS)]

    def optFields(self,sp,config):
        return np.array([wr.optField(sp,config) for wr in self.WNDRS]).sum(axis=0)
    
    def __str__(self):
        return '\n'.join([str(wd) for wd in self])
    
    def __iter__(self):
        return iter(self.WNDRS)

    def draw(self):
        [wd.draw() for wd in self.WNDRS]