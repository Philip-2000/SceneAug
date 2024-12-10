import numpy as np

class optm():
    def __init__(self,pmVersion=None,scene=None,PatFlag=False,PhyFlag=True,rand=False,config={},exp=False):
        self.scene = scene
        from ..Experiment.Tmer import tmer,tme
        self.timer = tmer() if exp else tme()
        self.exp = exp
        self.PatOpt = None if not PatFlag else PatOpt(pmVersion,scene,self.timer,config=config["pat"],exp=exp) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,self.timer,config=config["phy"],exp=exp)
        _           = None if not rand    else self.__random()

    def __random(self):
        for o in self.scene.OBJES:
            if o.idx % 2 == 0:
                o.translation -= 0.8*o.direction()

    def __call__(self, s, iRate=-1, jRate=-1): #timer, adjs, vio, fit, cos(PhyAdjs,PatAdjs), Over
        if self.PhyOpt and self.PatOpt:
            self.timer("all",1)
            PhyRet = self.PhyOpt(s,iRate) #adjs,vio,self.over(adjs,vio)
            PatRet = self.PatOpt(s,jRate) #adjs,fit,self.over(adjs,vio)
            self.timer("all",0)
            if self.exp:#单次操作耗时，adjust数值，violate数值，recognize的fit，pat操作趋势和phy操作趋势冲突？
                return {"timer":self.timer, "adjs":PhyRet["adjs"]+PatRet["adjs"], "vio":PhyRet["vio"], "fit":PatRet["fit"], "cos":PhyRet["adjs"]-PatRet["adjs"], "over":PhyRet["over"] and PatRet["over"]}
            else:
                return {"over":PhyRet["over"] and PatRet["over"]}
        elif self.PhyOpt:
            return {"timer":self.timer,**(self.PhyOpt(s,iRate))} #adjs,vio,self.over(adjs,vio)
        elif self.PatOpt:
            return {"timer":self.timer,**(self.PatOpt(s,jRate))} #adjs,fit,self.over(adjs,vio)
    
    def loop(self, steps=100, iRate=-1, jRate=-1): #an example of loop, but it's recommended to call the __call__ directly
        if steps>0:
            for s in range(steps):
                self(s,iRate,jRate)
                print(s)
        else:
            ret,s = [False],0
            while (not ret[-1]): #the over criterion
                ret =self(s,iRate,jRate)
                s=s+1
        _ = (self.PhyOpt.show() if self.PhyOpt else None, self.PatOpt.show() if self.PatOpt else None)
        
class PhyOpt():
    def __init__(self,scene,timer,config={},exp=False):
        self.scene = scene
        self.config= config
        self.iRate = config["rate"]
        self.s4 = config["s4"]
        self.timer = timer
        self.exp = exp
        
        self.configVis = config["vis"] if "vis" in config else None
        if (not exp) and self.configVis and (self.configVis["fiv"] or self.configVis["fih"] or self.configVis["fip"] or self.configVis["fiq"]):
            from ..Semantic.Fild import fild
            self.scene.fild = fild(scene,config["grid"],config)
        self.shows = {"res":[],"syn":[],"pnt":[],"pns":[],"fiv":[],"fih":[],"fip":[],"fiq":[]}
        self.steps = 0

    def draw(self,s):
        if self.exp:
            return
        r = self.scene.fild() if self.scene.fild else None
        for nms in self.shows:
            if nms in self.configVis:# and (nms[:2] != "fi"):#
                self.shows[nms].append(self.scene.drao(nms, self.configVis[nms],s))

    def show(self):
        from moviepy.editor import ImageSequenceClip
        import os
        for nms in self.shows:
            if len(self.shows[nms]):
                ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)
    
    def over(self,adjs,vio):
        return False

    def __call__(self,s,ir=-1):
        self.timer("phy_opt",1)
        adjs= self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=(self.iRate if ir<0 else ir))
        self.timer("phy_opt",0)
        vio = self.scene.OBJES.violates() if self.exp else None #[SumOfNorm(s.t),SumOfNorm(s.t),SumOfNorm(s.t),......]
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":adjs,"vio":vio,"over":self.over(adjs,vio)}


class PatOpt():
    def __init__(self,pmVersion,scene,timer,config={},exp=False):
        from SceneClasses.Operation.Patn import patternManager as PM 
        self.PM = PM(pmVersion)
        self.scene = scene
        self.rerec = False if "rerec"  not in config else config["rerec"]
        self.prerec=(False if "prerec" not in config else config["prerec"]) and not self.rerec
        self.rand  = False if "rand"   not in config else config["rand"]
        self.iRate = config["rate"]

        self.configVis = config["vis"] if "vis" in config else None

        self.timer = timer
        self.exp = exp
        self.steps = 0
        from .Plan import plans
        if self.rand and self.prerec:
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
            self.__random()
        elif self.rand:
            self.__random()
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        elif self.prerec:
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        self.shows = {"pat":[]}

    def __random(self):
        for o in self.scene.OBJES:
            if o.idx % 2 == 0:
                o.translation -= 0.8*o.direction()

    def draw(self,s):
        if not self.exp:
            self.shows["pat"].append(self.scene.drao("pat", self.configVis["pat"],s))

    def show(self):
        from moviepy.editor import ImageSequenceClip
        import os
        ImageSequenceClip(self.shows["pat"], fps=3).write_videofile(os.path.join(self.scene.imgDir,"pat.mp4"),logger=None)
    
    def over(self,adjs,fit):
        return False
    
    def __call__(self,s,ir=-1):
        if self.rerec:
            self.timer("pat_rec",1)
            from .Plan import plans
            plans(self.scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
            self.timer("pat_rec",0)
        
        self.timer("pat_opt",1)
        adjs= self.scene.plan.optimize((self.iRate if ir <0 else ir),self.exp)
        self.timer("pat_opt",0)
        fit = self.scene.plan.update_fit() if self.exp else 0
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":adjs,"fit":fit,"over":self.over(adjs,fit)}
    