from .Scne import scne, scneDs
from .Patn import patternManager
from .Plan import plans
from .Obje import object_types, obje
from .Link import objLink

class agmt():
    def __init__(self,pmVersion,scene,nm="test",v=0):
        self.verb=v
        self.scene = scene
        self.pm = patternManager(pmVersion)
        self.name = nm

    def augment(self,sdev=0.2,cdev=2.,cnt=8):
        if not self.scene.grp:
            plans(self.scene,self.pm,v=0).recognize(use=True,draw=False,opt=False,show=False)
        from numpy.random import rand as R
        from numpy.random import randint as Ri 
        import numpy as np
        from math import pi as PI 
        from copy import copy
    
        result,logs = [],[]
        for _ in range(cnt):
            scene = copy(self.scene)
            if len(scene.GRUPS) == 1:
                l  = (R(3,)-0.5)*cdev
                R0 = (R(3,)-0.5)*sdev+1.0
                Ri0= (Ri(4)/2.0-1)*PI
                scene.GRUPS[0].adjust(l,R0,Ri0)
                logs.append({"t":0,"l":l,"Rs":[R0],"Ris":[Ri0]})
            elif len(scene.GRUPS) == 2:
                t,l = (R()*2-1)*PI,max([scene.GRUPS[0].size[0],scene.GRUPS[0].size[2],scene.GRUPS[1].size[0],scene.GRUPS[1].size[2]]) - R()*0.1
                d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
                Ri0, Ri1 = (Ri(4)/2.0-1)*PI,(Ri(4)/2.0-1)*PI
                R0, R1 = (R(3,)-0.5)*sdev+1.0,(R(3,)-0.5)*sdev+1.0
                scene.GRUPS[0].adjust( d*l,R0,Ri0)
                scene.GRUPS[1].adjust(-d*l,R1,Ri1)
                logs.append({"t":t,"l":l,"Rs":[R0,R1],"Ris":[Ri0,Ri1]})
            elif len(scene.GRUPS) == 3:
                raise NotImplementedError
                t,l = (R()*2-1)*PI,max([scene.GRUPS[0].size[0],scene.GRUPS[0].size[2],self.scene.GRUPS[1].size[0],self.scene.GRUPS[1].size[2]]) - R()*0.1
                d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
                scene.GRUPS[0].adjust( d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
                scene.GRUPS[1].adjust(-d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
                logs.append({"t":t,"l":l,"Rs":[R0,R1,R2],"Ris":[Ri0,Ri1,Ri2]})
            else:
                raise NotImplementedError
        
            scene.draftRoomMask()
            result.append(scene)
        return result, self.scene, logs
    
    def show():
        return

class syth():
    def __init__(self,pmVersion,scene,nm,v):
        self.verb=v
        self.scene = scene
        self.pm = patternManager(pmVersion)
        self.name = nm
    
    def uncond(self):
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene

    def show():
        return

class gnrt(syth):
    def __init__(self,pmVersion,scene=None,nm="test",v=0):
        super(gnrt,self).__init__(pmVersion,(scne.empty(nm) if scene is None else scene),self.__class__.__name__,nm,v)

    def uncond(self):
        import numpy as np
        N = self.nods[0]
        while len(N.edges)>0:
            cs = 0
            for ed in N.edges:
                cs += ed.confidence
                if np.random.rand() < ed.confidenceIn:
                    N,m = ed.endNode,ed.startNode
                    while not (N.idx in m.bunches):
                        m = m.source.startNode
                    r = m.bunches[N.idx].sample()
                    a = [o for o in self.scene.OBJES if o.nid == m.idx] if m.idx > 0 else [obje(np.array([0,0,0]),np.array([1,1,1]),np.array([0]))]
                    o = a[0].rely(obje.fromFlat(r,j=object_types.index(N.type)),self.scaled)
                    self.scene.addObject(o)
                    o.nid = N.idx
                    if m.idx > 0:
                        self.scene.LINKS.append(objLink(a[0].idx,o.idx,len(self.scene.LINKS),self.scene))
                    cs = 0
                    break
            
                if np.random.rand() < cs:
                    break
        return self.scene

    def textcond(self):
        raise NotImplementedError
        return self.scene

    def roomcond(self):
        raise NotImplementedError
        return self.scene

class copl(syth):
    def __init__(self,pmVersion,scene,nm="test",v=0):
        super(copl,self).__init__(pmVersion,scene,self.__class__.__name__,nm,v)
        raise NotImplementedError

    def uncond(self):
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene

class rarg(syth):
    def __init__(self,pmVersion,scene,nm="test",v=0):
        super(rarg,self).__init__(pmVersion,scene,self.__class__.__name__,nm,v)
        raise NotImplementedError

    def uncond(self):
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene