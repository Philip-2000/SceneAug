from ..Experiment.ExOp import EXOP_BASE_DIR
default_optm_config = {
    "vis":{
        "res":{"res":(.5,.5,.5),},
        "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
        "pns":{"wo":(1.0,0,0),"wi":(0,0,1.0),"ob":(.33,.33,.33),"dr":(0,1.0,0),},
        "fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(0,1.0,0),},
        "pat":True,
    },
    "pat":{
        #"rerec":False,
        "prerec":True,
        #"rand":False, #/5.0
        "rate":{"mode":"exp_dn","r0":0.9,"lda":0.05,"rinf":0.4},#{"mode":"static","v":0.8}, #
        "vis":{
            "pat":True,
            "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
        }
    },
    "phy":{
        "rate":{"mode":"exp_up","rinf":0.1*10,"lda":0.5,"r0":0.1/100.0},#{"mode":"static","v":rate/500},#
        "s4": 4,
        "door":{"expand":0.9,"in":0.5,"rt":2.0}, #"out":0.1,
        "wall":{"bound":0.5,},
        "object":{
            "Pendant Lamp":[.0,.01,.01,False],#
            "Ceiling Lamp":[.0,.01,.01,False],#
            "Bookcase / jewelry Armoire":[.2,1., 1.,True],#
            "Round End Table":[.0,.5, .5,False],#
            "Dining Table":[.0,.5, .5,False],#
            "Sideboard / Side Cabinet / Console table":[.0,.9, .9,True],#
            "Corner/Side Table":[.0,.9, .9,True],#
            "Desk":[.0,.9, .9,True],#
            "Coffee Table":[.0,1.,1.1,False],#
            "Dressing Table":[.0,.9, .9,True],#
            "Children Cabinet":[.2,1., .9,True],#
            "Drawer Chest / Corner cabinet":[.2,1., 1.,True],#
            "Shelf":[.2,1., 1.,True],#
            "Wine Cabinet":[.2,1., 1.,True],#
            "Lounge Chair / Cafe Chair / Office Chair":[.0,.5, .5,False],#
            "Classic Chinese Chair":[.0,.5, .5,False],#
            "Dressing Chair":[.0,.5, .5,False],#
            "Dining Chair":[.0,.8, .8,False],#
            "armchair":[.0,.5, .5,False],#
            "Barstool":[.0,.5, .5,False],#
            "Footstool / Sofastool / Bed End Stool / Stool":[.0,.5, .5,False],#
            "Three-seat / Multi-seat Sofa":[.2,1., 1.,True],#
            "Loveseat Sofa":[.2,1., 1.,True],#
            "L-shaped Sofa":[.0,.6, 1.,True],#
            "Lazy Sofa":[.2,1., 1.,True],#
            "Chaise Longue Sofa":[.2,1., 1.,True],#
            "Wardrobe":[.2,1., 1.,True],#
            "TV Stand":[.2,1., 1.,True],#
            "Nightstand":[.0,.5, .5,True],#
            "King-size Bed":[.2,1.2,1.2,True],#
            "Kids Bed":[.2,1.2,1.2,True],#
            "Bunk Bed":[.2,1.2,1.2,True],#
            "Single bed":[.2,1.2,1.2,True],#
            "Bed Frame":[.2,1.2,1.2,True],#
        },
        "syn":{"T":1.1,"S":0.6,"R":1.0,},
        "grid":{"L":3.2,"d":0.1,"b":10,},
        "vis":{
            "res":{"res":(.5,.5,.5),},
            "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
            "pns":{"wo":(1.0,0,0),"wi":(0,0,1.0),"ob":(.33,.33,.33),"dr":(0,1.0,0),},
            # "fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            # "fih":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            # "fiq":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            # #"fip":{"res":(0.33,0.33,0.33)},
            # #"pnt":{"al":(.0,.0,.0),},
        }
    },
    "adjs":{"inertia":0.0,"decay":200.0,}
}

class optm_mcmc():
    def __init__(self,pmVersion,scene,timer=None):
        self.scene=scene
        from SceneClasses.Operation.Patn import patternManager as PM 
        from .Shdl import shdl_factory
        from ..Experiment.Tmer import tme
        self.PM = PM(pmVersion)
        self.E,self.T = -1e3, shdl_factory(T=1e4,a=0.1,mode="hamonic")
        self.timer = timer if timer else tme()

    def __eval(self):
        from .Plan import plans
        from ..Experiment.Tmer import tme
        fit,_,__ = plans(self.scene,self.PM,v=0).recognize(use=False,draw=False,show=False)
        _, vio = self.scene.OBJES.optimizePhy(default_optm_config["phy"],tme()), self.scene.OBJES.violates()
        return fit - 100*vio

    def __call__(self,T):
        import random, numpy as np
        from .Adjs import adj
        I = random.choice([o.idx for o in self.scene.OBJES if o.v])
        self.scene.OBJES[I].adjust = adj(np.random.normal(0,1,3),np.random.normal(0,0.01,3),np.random.normal(0,1,1),self.scene.OBJES[I])
        E = self.__eval()
        if E > self.E or np.random.rand() < np.exp((E-self.E)/T):#if the result is bette, accept it; if it is worse, still accept it with a probability of exp(-deltaE/T) 
            self.state = E
        else: #if this modification is not accepted, undo it
            a = self.scene.OBJES[I].adjust
            self.scene.OBJES[I].adjust = adj(-a.T,-a.S,-a.R,self.scene.OBJES[I])

    def loop(self,s=1000):
        self.timer("")
        for t in range(s):
            self(self.T(t))
        self.timer("")

class optm():
    def __init__(self,pm=None,scene=None,PatFlag=True,PhyFlag=True,rand=-1,config={},exp=False, timer=None):
        self.scene = scene
        from ..Experiment.Tmer import tme
        self.timer = tme() if timer is None else timer
        self.exp, self.state, self.config = exp, -1e3, config
        from . import Adjs
        Adjs.INERTIA, Adjs.DECAY_RATE = config["adjs"]["inertia"], config["adjs"]["decay"]
        self.PatOpt = None if not PatFlag else PatOpt(pm,scene,self.timer,config=config["pat"],exp=exp) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,self.timer,config=config["phy"],exp=exp)
        _           = None if rand < 0    else self.__random(rand)

        self.over_bounds, self.over_states, self.over_len = {"ads":0.5,"vios":2.0,"fits":35.0}, {"ads":[],"vios":[],"fits":[]}, 4
        self.error_bounds = {"vios":10.0,"fits":80.0}

    def __random(self,rand,use=True):
        import numpy as np,os
        a,b = self.scene.randomize(dev=rand,cen=True,hint=np.load(os.path.join(self.scene.imgDir,"rand.npy")) if use and os.path.exists(os.path.join(self.scene.imgDir,"rand.npy")) else None)#
        np.save(os.path.join(self.scene.imgDir,"rand.npy"), b)
        return a
        
    def __over(self,ad,fit,vio):
        import numpy as np
        self.over_states["ads"].append(ad.Norm())
        self.over_states["vios"].append(vio)
        self.over_states["fits"].append(fit)
        
        # if fit > self.error_bounds["fits"]*4 or vio > self.error_bounds["vios"]:
        #     print("initial error: fit=%f, vio=%f"%(fit,vio))
        #     return None
        if len(self.over_states["ads"]) > self.over_len:
            if fit > self.error_bounds["fits"] or vio > self.error_bounds["vios"]:
                print("error: fit=%f, vio=%f"%(fit,vio))
                return None
            self.over_states["ads"].pop(0)
            self.over_states["vios"].pop(0)
            self.over_states["fits"].pop(0)
        return False if len(self.over_states["ads"]) < self.over_len else bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["vios"]).mean() < self.over_bounds["vios"] and np.array(self.over_states["fits"]).mean() < self.over_bounds["fits"])
        
    def qualitative(self):
        import json,os
        fit, vio = self.PatOpt.eval(), self.PhyOpt.eval(True)
        self.scene.eval={"adjs":0.0,"vio":vio,"fit":fit, "s":0.0, "time":0.00}
        for nms in ["pns","syn","pat","fiv"]: self.scene.drao(nms, self.config["vis"][nms], 0.0)
        open(os.path.join(self.scene.imgDir,"%.1f.json"%(0.0)),"w").write(json.dumps(self.scene.toSceneJson()))

        from ..Experiment.Tmer import tmer
        timer = tmer()
        [o.adjust.clear() for o in self.scene.OBJES]
        for s in range(6):
            timer("")
            adjs= self.PatOpt.opt(s)
            timer("")
            fit, vio = self.PatOpt.eval(), self.PhyOpt.eval(True)
            self.scene.eval={"adjs":adjs.Norm(),"vio":vio,"fit":fit, "s":s+0.5, "time":timer[""].last}
            for nms in ["pns","syn","pat"]: self.scene.drao(nms, self.config["vis"][nms], s+0.5)
            open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))

            timer("")
            adjs= self.PhyOpt.opt(s)
            timer("")
            fit, vio = self.PatOpt.eval(), self.PhyOpt.eval()
            self.scene.eval={"adjs":adjs.Norm(),"vio":vio,"fit":fit, "s":s+1.0, "time":timer[""].last}
            for nms in ["pns","syn","pat"]: self.scene.drao(nms, self.config["vis"][nms], s+1.0)
            open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+1.0)),"w").write(json.dumps(self.scene.toSceneJson()))

    def __call__(self, s, debugdraw=None): #timer, adjs, vio, fit, cos(PhyAdjs,PatAdjs), Over
        if self.PhyOpt and self.PatOpt:
            self.timer("") #from .Adjs import adjs #ad = adjs(self.scene.OBJES) #print("zero") #print(ad)
            #debugdraw(self.scene,s,"--") if debugdraw else None #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d--.png"%(self.scene.scene_uid[:10],s))#return #
            PatRet = self.PatOpt.opt(s) #adjs,fit,self.over(adjs,vio)
            #print("no") #print(PhyRet["adjs"])
            #debugdraw(self.scene,s,"-") if debugdraw else None ##self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-.png"%(self.scene.scene_uid[:10],s))#return #
            PhyRet = self.PhyOpt.opt(s) #adjs,vio,self.over(adjs,vio)
            #print("yes")#print(PatRet["adjs"])
            self.timer("")
            
            fit = self.PatOpt.eval()#,fitmax = self.scene.plan.update_fit() #if self.exp else (0,0)
            #print(fit)
            #,"over":self.over(adjs,fit)
            vio = self.PhyOpt.eval()#self.scene.OBJES.violates() #if self.exp else None
            ad = PhyRet+PatRet #PhyRet["adjs"]+PatRet["adjs"]
            #,"over":self.over(adjs,vio))
            
            return {"adj":ad, "vio":vio, "fit":fit,
                    "over": self.__over(ad,fit,vio)
                } if self.exp else {"over": self.__over(ad,fit,vio)}
        elif self.PhyOpt:
            assert not self.exp
            PhyRet = self.PhyOpt(s) #adjs,vio,self.over(adjs,vio)
            return {"timer":self.timer,**(PhyRet)} #adjs,vio,self.over(adjs,vio)
        elif self.PatOpt:
            assert not self.exp
            PatRet = self.PatOpt(s) #adjs,vio,self.over(adjs,vio)
            return {"timer":self.timer,**(PatRet)} #adjs,fit,self.over(adjs,vio)
    
    def loop(self, steps=-1, pbar=None): #an example of loop, but it's recommended to call the __call__ directly
        [o.adjust.clear() for o in self.scene.OBJES]
        if steps<0:
            while True:
                ret,s = {"over":False},0
                while (ret["over"] is False) and s <= 20: #the over criterion #assert ret["over"] is False and ret["over"] is not None
                    ret, s = self(s), s+1
                if ret["over"] is not None and s <= 20:
                    break
                else:
                    print("restart",self.scene.scene_uid)
                    adjs0 = self.__random(2.0,use=False)
        else:
            for s in range(steps):
                self(s)
                if pbar:
                    pbar.set_description("optimizing %s %d"%(self.scene.scene_uid[:20], s))
        
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
        for nms in [nms for nms in self.shows if len(self.shows[nms])]:
            from moviepy.editor import ImageSequenceClip
            import os
            _ = ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)
                
    def over(self,vio):
        return False #vio < 0.001
    
    def eval(self,safe=True):
        if safe:
            self.scene.OBJES.optimizePhy(self.config,self.timer,debug=True,ut=-1)
        return self.scene.OBJES.violates()
    
    def opt(self,s,ir=-1):
        return self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=(self.iRate(s) if ir<0 else ir))

    def __call__(self,s,ir=-1):
        #self.scene.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d-opt.png"%(self.scene.scene_uid[:10],s))#return #
        self.timer("phy_opt")
        adjs = self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=(self.iRate(s) if ir<0 else ir))
        self.timer("phy_opt")
        
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":adjs}# bdjs+adjs} #

class PatOpt():
    def __init__(self,pm,scene,timer,config={},exp=False):
        #from SceneClasses.Operation.Patn import patternManager as PM 
        from .Shdl import shdl_factory
        self.PM = pm #PM(pmVersion)
        self.scene = scene
        
        self.prerec=(False if "prerec" not in config else config["prerec"])# and not self.rerec
        #self.rand  = False if "rand"   not in config else config["rand"]
        self.iRate = shdl_factory(**config["rate"])
        self.configVis = config["vis"] if "vis" in config else None

        self.timer = timer
        self.exp = exp
        self.steps = 0
        if not exp: # exp did random by itself (because it has to record the result of randomization) , not from us
            if self.prerec: #for recognition: when we know how the original scene is, we use recognition
                from .Plan import plans
                plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
                #if self.rand: self.__random()
            else: #for rearrangement: when we don't know how the original scene is, we use rearrangement to guess how the semantic of the objects are
                from .Syth_Rarg import rarg
                rarg(self.PM,scene,v=0).uncond(use=True,move=True,draw=False)

        self.shows = {"pat":[]}

    def draw(self, s):
        if not self.exp:
            if "pat" in self.configVis:# and (nms[:2] != "fi"):#
                self.shows["pat"].append(self.scene.drao("pat", self.configVis["pat"], s))

    def show(self):
        for nms in [nms for nms in self.shows if len(self.shows[nms])]:
            from moviepy.editor import ImageSequenceClip
            import os
            ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)
    
    def over(self,fit):
        return False#adjs.norm() < 0.1
    
    def eval(self):
        fit,fitmax = self.scene.plan.update_fit()
        return fitmax-fit

    def opt(self,s,ir=-1):
        return self.scene.plan.optimize((self.iRate(s) if ir <0 else ir),s)

    def __call__(self,s,ir=-1):        
        self.timer("pat_opt")
        adjs = self.scene.plan.optimize((self.iRate(s) if ir <0 else ir),s)
        self.timer("pat_opt")
        
        self.draw(s)
        self.steps = max(s,self.steps)
        return {"adjs":adjs}#bdjs+adjs}#
    