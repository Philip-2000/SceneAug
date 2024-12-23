from ..Experiment.ExOp import EXOP_BASE_DIR
class optm():
    def __init__(self,pmVersion=None,scene=None,PatFlag=False,PhyFlag=True,rand=False,config={},exp=False):
        self.scene = scene
        from ..Experiment.Tmer import tmer,tme
        self.timer = tmer() if exp else tme()
        self.exp = exp
        from . import Adjs
        Adjs.INERTIA, Adjs.DECAY_RATE = config["adjs"]["inertia"], config["adjs"]["decay"]
        self.PatOpt = None if not PatFlag else PatOpt(pmVersion,scene,self.timer,config=config["pat"],exp=exp) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,self.timer,config=config["phy"],exp=exp)
        _           = None if not rand    else self.__random()

    def __random(self):
        self.dev = 0.05
        import numpy as np
        from ..Operation.Adjs import adjs,adj
        for o in self.scene.OBJES:
            o.adjust = adj(T=np.random.randn((3))*self.dev,S=np.random.randn((3))*self.dev * 0.1,R=np.random.randn((1))*self.dev,o=o)#o.adjust()
        return adjs(self.scene.OBJES)
        #for o in self.scene.OBJES: o.translation = o.translation if o.idx%2 else o.translation - 0.8*o.direction()
    
    def __over(self,ad,fit,vio):
        return False#ad.Norm()<0.1 and self.PatOpt.over(fit) and self.PhyOpt.over(vio)
    
    def __call__(self, s): #timer, adjs, vio, fit, cos(PhyAdjs,PatAdjs), Over

        if self.PhyOpt and self.PatOpt:
            self.timer("all",1)
            self.timer("accum",1) #from .Adjs import adjs #ad = adjs(self.scene.OBJES) #print("zero") #print(ad)
            #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d--.png"%(self.scene.scene_uid[:10],s))#return #
            PatRet = self.PatOpt(s) #adjs,fit,self.over(adjs,vio)
            #print("no") #print(PhyRet["adjs"])
            self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-.png"%(self.scene.scene_uid[:10],s))#return #
            PhyRet = self.PhyOpt(s) #adjs,vio,self.over(adjs,vio)
            #print("yes")#print(PatRet["adjs"])
            self.timer("all",0)
            self.timer("accum",0)
            
            fit = self.scene.plan.update_fit() if self.exp else 0
            #,"over":self.over(adjs,fit)
            vio = self.scene.OBJES.violates() if self.exp else None
            ad = PhyRet["adjs"]+PatRet["adjs"]
            #,"over":self.over(adjs,vio))



            return {"timer":self.timer,
                    "adjs":ad,
                    "vio":vio,
                    "fit":fit,
                    "cos":PhyRet["adjs"]-PatRet["adjs"],
                    "over": self.__over(ad,fit,vio)
                } if self.exp else {"over":PhyRet["over"] and PatRet["over"]}
        elif self.PhyOpt:
            assert not self.exp
            PhyRet = self.PhyOpt(s) #adjs,vio,self.over(adjs,vio)
            return {"timer":self.timer,**(PhyRet)} #adjs,vio,self.over(adjs,vio)
        elif self.PatOpt:
            assert not self.exp
            PatRet = self.PatOpt(s) #adjs,vio,self.over(adjs,vio)
            return {"timer":self.timer,**(PatRet)} #adjs,fit,self.over(adjs,vio)
    
    def loop(self, steps=100, iRate=-1, jRate=-1): #an example of loop, but it's recommended to call the __call__ directly
        if steps>0:
            for s in range(steps):
                self(s,iRate,jRate)
                print(s)
        else:
            self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d.png"%(self.scene.scene_uid[:10],0))
            ret,step,time = {"over":False},0,0
            while (not ret["over"]): 
                ret = self(step)
                step += 1
                self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d.png"%(self.scene.scene_uid[:10],step))
                if step > 8:
                    break
        _ = (self.PhyOpt.show() if self.PhyOpt else None, self.PatOpt.show() if self.PatOpt else None)

class PhyOpt():
    def __init__(self,scene,timer,config={},exp=False):
        from .Shdl import shdl_factory
        self.scene = scene
        self.config= config
        self.iRate = shdl_factory(**config["rate"])
        self.s4 = config["s4"]
        self.timer = timer
        self.exp = exp
        
        self.configVis = config["vis"] if "vis" in config else None
        if (not exp) and self.configVis:
            if ("fiv" in self.configVis or "fih" in self.configVis or "fip" in self.configVis or "fiq" in self.configVis):
                from ..Semantic.Fild import fild
                self.scene.fild = fild(scene,config["grid"],config)
        self.shows = {"res":[],"syn":[],"pnt":[],"pns":[],"fiv":[],"fih":[],"fip":[],"fiq":[]}
        self.steps = 0

    def draw(self,s):
        if not self.exp:
            r = self.scene.fild() if self.scene.fild else None
            for nms in self.shows:
                if nms in self.configVis:# and (nms[:2] != "fi"):#
                    self.shows[nms].append(self.scene.drao(nms, self.configVis[nms],s))

    def show(self):
        from moviepy.editor import ImageSequenceClip
        import os
        for nms in self.shows:
            _ = ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None) if len(self.shows[nms]) else None
                
    def over(self,vio):
        return False #vio < 0.001

    def __call__(self,s,ir=-1):
        #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-opt.png"%(self.scene.scene_uid[:10],s))#return #
        self.timer("phy_opt",1)
        adjs = self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=(self.iRate(s) if ir<0 else ir))
        self.timer("phy_opt",0)
        #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-snp.png"%(self.scene.scene_uid[:10],s))#return #
        bdjs = adjs.snapshot()
        self.timer("phy_inf",1)
        #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-inf.png"%(self.scene.scene_uid[:10],s))#return #
        adjs.apply_influence()
        #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d+inf.png"%(self.scene.scene_uid[:10],s))#return #
        self.timer("phy_inf",0)
        #vio = self.scene.OBJES.violates() if self.exp else None #[SumOfNorm(s.t),SumOfNorm(s.t),SumOfNorm(s.t),......]
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":bdjs+adjs}

class PatOpt():
    def __init__(self,pmVersion,scene,timer,config={},exp=False):
        from SceneClasses.Operation.Patn import patternManager as PM 
        from .Shdl import shdl_factory
        self.PM = PM(pmVersion)
        self.scene = scene
        self.rerec = False if "rerec"  not in config else config["rerec"]
        self.prerec=(False if "prerec" not in config else config["prerec"]) and not self.rerec
        self.rand  = False if "rand"   not in config else config["rand"]
        self.iRate = shdl_factory(**config["rate"])
        self.configVis = config["vis"] if "vis" in config else None

        self.timer = timer
        self.exp = exp
        self.steps = 0
        if not exp:
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
    
    def over(self,fit):
        return False#adjs.norm() < 0.1
    
    def __call__(self,s,ir=-1):
        # if self.rerec:
        #     self.timer("pat_rec",1)
        #     from .Plan import plans
        #     plans(self.scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        #     self.timer("pat_rec",0)
        
        self.timer("pat_opt",1)
        adjs= self.scene.plan.optimize((self.iRate(s) if ir <0 else ir))
        #print(adjs)
        self.timer("pat_opt",0)
        bdjs = adjs.snapshot()
        self.timer("phy_inf",1)
        adjs.apply_influence()
        self.timer("phy_inf",0)
        #fit = self.scene.plan.update_fit() if self.exp else 0
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":bdjs+adjs}
    