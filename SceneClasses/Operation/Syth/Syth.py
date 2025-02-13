#from .Scne import scne
#from ..Basic.Obje import object_types, obje
import os
SYN_IMG_BASE_DIR = "./pattern/syth/"

class agmt():
    def __init__(self,pmVersion,scene,nm="test",v=0):
        from .Patn import patternManager as PM
        self.verb=v
        self.scene = scene
        self.scene.imgDir = os.path.join(SYN_IMG_BASE_DIR,pmVersion,"agmt")
        os.makedirs(self.scene.imgDir,exist_ok=True)
        self.pm = PM(pmVersion)
        self.name = nm

    def augment(self,sdev=0.2,cdev=2.,cnt=8,draw=False):
        if not self.scene.grp:
            from .Plan import plans
            plans(self.scene,self.pm,v=0).recognize(use=True,draw=False,show=False)
        from numpy.random import rand as R
        from numpy.random import randint as Ri 
        import numpy as np
        from math import pi as PI 
        from copy import copy
    
        result,logs = [],[]
        for _ in range(cnt):
            scene = copy(self.scene)
            scene.scene_uid = scene.scene_uid+str(_)
            os.makedirs(scene.imgDir,exist_ok=True)
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
            if draw:
                scene.draw()
        return result, self.scene, logs
    
    def show(self):
        return #not sure yet

class syth():
    def __init__(self,pm,scene,appli,nm,v, txcond=True, rmcond=True):
        self.verb=v
        self.scene = scene
        self.scene.imgDir = os.path.join(SYN_IMG_BASE_DIR,pm.version,appli) if nm != "test" else self.scene.imgDir
        os.makedirs(self.scene.imgDir,exist_ok=True)
        self.pm = pm
        self.name = nm

        self.txcond, self.rmcond = txcond, rmcond
     
    def uncond(self):
        raise NotImplementedError("virtual function")

    def textcond(self):
        raise NotImplementedError("virtual function")

    def roomcond(self):
        raise NotImplementedError("virtual function")
    
    def txrmcond(self):
        raise NotImplementedError("virtual function")

    def __call__(self):
        if (self.txcond and self.scene.text) and (self.rmcond and self.scene.room):
            return self.txrmcond()
        elif self.txcond and hasattr(self.scene, "TEXTS") and len(self.scene.TEXTS):
            return self.textcond()
        elif self.rmcond and len(self.scene.WALLS):
            return self.roomcond()
        else:
            return self.uncond()

    def show():
        return

