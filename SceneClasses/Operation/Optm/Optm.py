from ...Experiment import EXOP_BASE_DIR
def tme(*args,**kwargs): pass
default_optm_config = {
    "vis":{
        "res":{"res":(.5,.5,.5),},
        "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
        "pns":{"wo":(1.0,0,0),"wi":(0,0,1.0),"ob":(.33,.33,.33),"dr":(0,1.0,0),},
        #"fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(0,1.0,0),},
        "save":True,
        "pat":True,
    },
    "pat":{
        #"rerec":False,
        "prerec":True,
        #"rand":False, #/5.0
        "rate":{"mode":"exp_dn","r0":0.9,"lda":0.02,"rinf":0.4},#{"mode":"static","v":0.8}, #
        "vis":{
            "pat":True,
            "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
        },
        "stop":{ "over_bounds":{"ads":0.5, "fits":4.0}, "error_bounds":{"fits":50.0}, "over_len":4, "max_len":20, },
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
        "grid":{"L":4.6,"d":0.1,"b":10,},
        "vis":{
            "res":{"res":(.5,.5,.5),},
            "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
            "pns":{"wo":(1.0,0,0),"wi":(0,0,1.0),"ob":(.33,.33,.33),"dr":(0,1.0,0),},
            #"fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            #"fih":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            #"fiq":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            #"fip":{"res":(0.33,0.33,0.33)},
            #"pnt":{"al":(.0,.0,.0),},
        },
        "stop":{ "over_bounds":{"ads":0.5, "vios":2.0}, "error_bounds":{"vios":10.0}, "over_len":4, "max_len":20 },
    },
    #"stop":{ "over_bounds":{"ads":0.5}, "error_bounds":{}, "over_len":4, },
    "adjs":{"inertia":0.0,"decay":200.0,}
}

class optm_mcmc():
    def __init__(self,pmVersion,scene,timer=None):
        self.scene=scene
        from SceneClasses.Operation.Patn import patternManager as PM 
        from .Shdl import shdl_factory
        from ...Experiment import tme
        self.PM = PM(pmVersion)
        self.E,self.T = -1e3, shdl_factory(T=1e4,a=0.1,mode="hamonic")
        self.timer = timer if timer else tme()

    def __eval(self):
        from .. import rgnz
        from ...Experiment import tme
        fit,_,__ = rgnz(self.scene,self.PM,v=0).recognize(use=False,draw=False,show=False)
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
        for t in range(s): self(self.T(t))
        self.timer("")

class optm_utils():
    def __init__(self,scene=None,config={},timer=None):
        self.timer = tme if timer is None else timer
        self.scene,self.config = scene,config
        self.shows = { nms:[] for nms in config["vis"] }

    def _random(self,rand,use=True):
        import numpy as np,os
        a,b = self.scene.randomize(dev=rand,cen=True,hint=np.load(os.path.join(self.scene.imgDir,"rand.npy")) if use and os.path.exists(os.path.join(self.scene.imgDir,"rand.npy")) else None)#
        np.save(os.path.join(self.scene.imgDir,"rand.npy"), b)
        return a

    def _draw(self,s):
        import os, json
        for nms in self.config["vis"]: self.shows[nms].append(self.scene.drao(nms, self.config["vis"][nms], s)) if nms != "save" else open(os.path.join(self.scene.imgDir,"%.1f.json"%(s)),"w").write(json.dumps(self.scene.toSceneJson()))

    def _show(self):
        import os, moviepy.editor.ImageSequenceClip as ImageSequenceClip
        for nms in [nms for nms in self.shows if len(self.shows[nms])]: ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)        

    def _over(self,*args):
        raise AssertionError("virtual function")

    def eval(self,**kwargs):
        raise AssertionError("virtual function")

    def opt(self):
        raise AssertionError("virtual function")

    def __call__(self,s):
        ad = self.opt(s) #adjs,vio,self.over(adjs,vio)
        self._draw(s+1.0)
        e = self.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        return {"over": self._over(s,ad,e)}

    def loop(self, steps=-1, pbar=None): #an example of loop, but it's recommended to call the __call__ directly
        self.scene.OBJES.adjust_clear()
        if steps<0:
            while True:
                ret,s = {"over":False},0
                while (ret["over"] is False) and s <= self.max_len: #the over criterion #assert ret["over"] is False and ret["over"] is not None
                    ret, s = self(s), s+1
                if ret["over"] is True and s <= self.max_len: break #if the over criterion is met, break
                else:                                               #if the restart criterion is met, restart
                    print("restart",self.scene.scene_uid)
                    adjs0 = self.__random(2.0,use=False)
        else:
            for s in range(steps):
                self(s)
                if pbar: pbar.set_description("optimizing %s %d"%(self.scene.scene_uid[:20], s))
        self._show()

class PhyOpt(optm_utils):
    def __init__(self,scene,config,timer=None):
        super().__init__(scene,config,timer)
        from . import Shdl
        self.iRate = Shdl.shdl_factory(**config["rate"])
        self.s4 = config["s4"]
        self.over_bounds, self.over_states, self.over_len, self.max_len = config["stop"]["over_bounds"], {"ads":[],"vios":[]}, config["stop"]["over_len"], config["stop"]["max_len"]
        self.error_bounds = config["stop"]["error_bounds"]
    
    def _over(self,s,ad,vio):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["vios"].append(vio)
        if len(self.over_states["ads"]) > self.over_len:
            if vio > self.error_bounds["vios"] or s > self.max_len:
                print("error: vio=%f,s=%d"%(vio,s))
                return None
            self.over_states["ads"].pop(0), self.over_states["vios"].pop(0)
        return len(self.over_states["ads"]) == self.over_len and bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["vios"]).mean() < self.over_bounds["vios"])

    def eval(self,safe=True):
        if safe: self.scene.OBJES.optimizePhy(self.config,self.timer,debug=True,ut=-1)
        return self.scene.OBJES.violates()
    
    def opt(self,s):
        return self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.config["vis"]),ut=self.iRate(s))

class PatOpt(optm_utils):
    def __init__(self,pm,scene,config,timer=None,exp=False):
        super().__init__(scene,config,timer)
        from . import Shdl
        self.iRate = Shdl.shdl_factory(**config["rate"])
        self.PM = pm

        self.prerec = config.get("prerec", True)
        if self.prerec and not exp: #for recognition: when we know how the original scene is, we use recognition
            from .. import rgnz
            rgnz(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        elif not exp:               #for rearrangement: when we don't know how the original scene is, we use rearrangement to guess how the semantic of the objects are
            from .. import rarg
            rarg(self.PM,scene,v=0).uncond(use=True,move=True,draw=False)

        self.over_bounds, self.over_states, self.over_len, self.max_len = config["stop"]["over_bounds"], {"ads":[],"fits":[]}, config["stop"]["over_len"], config["stop"]["max_len"]
        self.error_bounds = config["stop"]["error_bounds"]

    def _over(self,s,ad,fit):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["fits"].append(fit)
        if len(self.over_states["ads"]) > self.over_len:
            if fit > self.error_bounds["fits"] or s > self.max_len:
                print("error: fit=%f,s=%d"%(fit,s))
                return None
            self.over_states["ads"].pop(0), self.over_states["fits"].pop(0)
        return len(self.over_states["ads"]) == self.over_len and bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["fits"]).mean() < self.over_bounds["fits"])

    def eval(self,safe=True):
        fit,fitmax = self.scene.PLAN.update_fit() #print(fit,fitmax)
        return fitmax-fit

    def opt(self,s):
        return self.scene.PLAN.optimize(self.iRate(s),s)
    
class optm(optm_utils):
    def __init__(self,pm,scene,config,timer=None,exp=False,rand=-1):
        super().__init__(scene,config,timer)
        self.PatOpt = PatOpt(pm,scene,config["pat"],timer,exp)
        self.PhyOpt = PhyOpt(scene,config["phy"],timer)

        self.over_states = {"ads":[],"vios":[],"fits":[]}
        self.over_bounds = {**config["pat"]["stop"]["over_bounds"], **config["phy"]["stop"]["over_bounds"]}
        self.over_len, self.max_len = config["pat"]["stop"]["over_len"], config["pat"]["stop"]["max_len"]
        self.error_bounds = {**config["pat"]["stop"]["error_bounds"], **config["phy"]["stop"]["error_bounds"]}
        if rand > 0: self._random(rand)

    def _over(self,s,ad,fit,vio):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["vios"].append(vio),  self.over_states["fits"].append(fit)
        
        if len(self.over_states["ads"]) > self.over_len:
            if fit > self.error_bounds["fits"] or vio > self.error_bounds["vios"] or s > self.max_len:
                print("error: fit=%f, vio=%f, s=%d"%(fit,vio,s))
                return None
            self.over_states["ads"].pop(0), self.over_states["vios"].pop(0), self.over_states["fits"].pop(0)
        return len(self.over_states["ads"]) == self.over_len and bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["vios"]).mean() < self.over_bounds["vios"] and np.array(self.over_states["fits"]).mean() < self.over_bounds["fits"])
    
    def __call__(self,s):
        PatRet = self.PatOpt.opt(s) 
        PhyRet = self.PhyOpt.opt(s)
        self._draw(s+1.0)
        
        fit = self.PatOpt.eval()#,fitmax = self.scene.PLAN.update_fit() #if self.exp else (0,0)
        vio = self.PhyOpt.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        ad = PhyRet+PatRet #PhyRet["adjs"]+PatRet["adjs"]

        return {"over": self._over(s,ad,fit,vio)}

    def qualitative(self): #for qualitative show
        #import json,os
        self.scene.EVALS={"adjs":0.0,"vio":self.PhyOpt.eval(True),"fit":self.PatOpt.eval(), "s":0.0, "time":0.00}
        from ...Semantic.Fild import fild
        self.scene.fild = fild(self.scene,self.config["phy"]["grid"],self.config["phy"])
        self.scene.drao("fiv", {"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(0,1.0,0),}, 0.0)
        self._draw(0.0)
        
        from ...Experiment import tmer
        timer = tmer()
        self.scene.OBJES.adjust_clear()
        for s in range(6):
            timer("",1)
            adjs= self.PatOpt.opt(s)
            timer("",0)
            self.scene.EVALS={"adjs":adjs.Norm(),"vio":self.PhyOpt.eval(True),"fit":self.PatOpt.eval(), "s":s+0.5, "time":timer[""].last}
            self._draw(s+0.5)

            timer("",1)
            adjs= self.PhyOpt.opt(s)
            timer("",0)
            self.scene.EVALS={"adjs":adjs.Norm(),"vio":self.PhyOpt.eval(),"fit":self.PatOpt.eval(), "s":s+1.0, "time":timer[""].last}
            self._draw(s+1.0)
            print(s)

    def exps(self,s): #for experiment, just write new functions for new usage, don't change any existing functions, please
        self.timer("",1) #from .Adjs import adjs #ad = adjs(self.scene.OBJES) #print("zero") #print(ad)
        
        self.timer("pat_opt",1)
        PatRet = self.PatOpt.opt(s) #adjs,fit,self.over(adjs,vio)
        self.timer("pat_opt",0)
        self.timer("phy_opt",1)
        PhyRet = self.PhyOpt.opt(s) #adjs,vio,self.over(adjs,vio)
        self.timer("phy_opt",0)
        self.timer("",0)
        
        fit = self.PatOpt.eval()#,fitmax = self.scene.PLAN.update_fit() #if self.exp else (0,0)
        vio = self.PhyOpt.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        ad = PhyRet+PatRet #PhyRet["adjs"]+PatRet["adjs"]
        
        return {"adj":ad, "vio":vio, "fit":fit, "over": self._over(s,ad,fit,vio)}

    
"""
class optm():
    def __init__(self,pm=None,scene=None,PatFlag=True,PhyFlag=True,rand=-1,config={},exp=False, timer=None):
        self.scene = scene
        from ...Experiment import tme
        self.timer = tme() if timer is None else timer
        self.exp, self.state, self.config = exp, -1e3, config
        from . import Adjs
        Adjs.INERTIA, Adjs.DECAY_RATE = config["adjs"]["inertia"], config["adjs"]["decay"]
        self.PatOpt = None if not PatFlag else PatOpt(pm,scene,self.timer,config=config["pat"],exp=exp) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,self.timer,config=config["phy"],exp=exp)
        _           = None if rand < 0    else self.__random(rand)
        
        self.shows = { nms:[] for nms in config["vis"] }

        #self.over_bounds, self.over_states, self.over_len = {"ads":0.5,"vios":2.0,"fits":5.0}, {"ads":[],"vios":[],"fits":[]}, 4
        #self.error_bounds = {"vios":10.0,"fits":50.0}
        self.over_states = {"ads":[],"vios":[],"fits":[]}
        self.over_bounds = {**config["pat"]["stop"]["over_bounds"], **config["phy"]["stop"]["over_bounds"]}
        self.over_len, self.max_len = config["pat"]["stop"]["over_len"], config["pat"]["stop"]["max_len"]
        self.error_bounds = {**config["pat"]["stop"]["error_bounds"], **config["phy"]["stop"]["error_bounds"]}

    def __random(self,rand,use=True):
        import numpy as np,os
        a,b = self.scene.randomize(dev=rand,cen=True,hint=np.load(os.path.join(self.scene.imgDir,"rand.npy")) if use and os.path.exists(os.path.join(self.scene.imgDir,"rand.npy")) else None)#
        np.save(os.path.join(self.scene.imgDir,"rand.npy"), b)
        return a
        
    def __over(self,ad,fit,vio):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["vios"].append(vio),  self.over_states["fits"].append(fit)
        
        if len(self.over_states["ads"]) > self.over_len:
            if fit > self.error_bounds["fits"] or vio > self.error_bounds["vios"]:
                print("error: fit=%f, vio=%f"%(fit,vio))
                return None
            self.over_states["ads"].pop(0), self.over_states["vios"].pop(0), self.over_states["fits"].pop(0)
        return False if len(self.over_states["ads"]) < self.over_len else bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["vios"]).mean() < self.over_bounds["vios"] and np.array(self.over_states["fits"]).mean() < self.over_bounds["fits"])
        
    def qualitative(self): #for qualitative show
        import json,os
        fit, vio = self.PatOpt.eval(), self.PhyOpt.eval(True)
        self.scene.EVALS={"adjs":0.0,"vio":vio,"fit":fit, "s":0.0, "time":0.00}
        for nms in ["pns","syn","pat","fiv"]: self.scene.drao(nms, self.config["vis"][nms], 0.0)
        open(os.path.join(self.scene.imgDir,"%.1f.json"%(0.0)),"w").write(json.dumps(self.scene.toSceneJson()))

        from ...Experiment import tmer
        timer = tmer()
        [o.adjust.clear() for o in self.scene.OBJES]
        for s in range(6):
            timer("",1)
            adjs= self.PatOpt.opt(s)
            timer("",0)
            fit, vio = self.PatOpt.eval(), self.PhyOpt.eval(True)
            self.scene.EVALS={"adjs":adjs.Norm(),"vio":vio,"fit":fit, "s":s+0.5, "time":timer[""].last}
            for nms in ["pns","syn","pat"]: self.scene.drao(nms, self.config["vis"][nms], s+0.5)
            open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))

            timer("",1)
            adjs= self.PhyOpt.opt(s)
            timer("",0)
            fit, vio = self.PatOpt.eval(), self.PhyOpt.eval()
            self.scene.EVALS={"adjs":adjs.Norm(),"vio":vio,"fit":fit, "s":s+1.0, "time":timer[""].last}
            for nms in ["pns","syn","pat"]: self.scene.drao(nms, self.config["vis"][nms], s+1.0)
            open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+1.0)),"w").write(json.dumps(self.scene.toSceneJson()))

    def exps(self,s): #for experiment
        self.timer("",1) #from .Adjs import adjs #ad = adjs(self.scene.OBJES) #print("zero") #print(ad)
        
        self.timer("pat_opt",1)
        PatRet = self.PatOpt.opt(s) #adjs,fit,self.over(adjs,vio)
        self.timer("pat_opt",0)
        self.timer("phy_opt",1)
        PhyRet = self.PhyOpt.opt(s) #adjs,vio,self.over(adjs,vio)
        self.timer("phy_opt",0)
        self.timer("",0)
        
        fit = self.PatOpt.eval()#,fitmax = self.scene.PLAN.update_fit() #if self.exp else (0,0)
        vio = self.PhyOpt.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        ad = PhyRet+PatRet #PhyRet["adjs"]+PatRet["adjs"]
        
        return {"adj":ad, "vio":vio, "fit":fit, "over": self.__over(ad,fit,vio)}

    def __call__(self,s):
        import json,os
        PatRet = self.PatOpt.opt(s) #adjs,fit,self.over(adjs,vio)
        for nms in self.config["vis"]: self.shows[nms].append(self.scene.drao(nms, self.config["vis"][nms], s+0.5)) if nms != "save" else open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))
        PhyRet = self.PhyOpt.opt(s) #adjs,vio,self.over(adjs,vio)
        for nms in self.config["vis"]: self.shows[nms].append(self.scene.drao(nms, self.config["vis"][nms], s+1.0)) if nms != "save" else open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+1.0)),"w").write(json.dumps(self.scene.toSceneJson()))
        
        fit = self.PatOpt.eval()#,fitmax = self.scene.PLAN.update_fit() #if self.exp else (0,0)
        vio = self.PhyOpt.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        ad = PhyRet+PatRet #PhyRet["adjs"]+PatRet["adjs"]

        #open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))

        return {"over": self.__over(ad,fit,vio)}
    
    def loop(self, steps=-1, pbar=None): #an example of loop, but it's recommended to call the __call__ directly
        [o.adjust.clear() for o in self.scene.OBJES]
        if steps<0:
            while True:
                ret,s = {"over":False},0
                while (ret["over"] is False) and s <= self.max_len: #the over criterion #assert ret["over"] is False and ret["over"] is not None
                    ret, s = self(s), s+1
                if ret["over"] is True and s <= self.max_len: break #if the over criterion is met, break
                else:                                               #if the restart criterion is met, restart
                    print("restart",self.scene.scene_uid)
                    adjs0 = self.__random(2.0,use=False)
        else:
            for s in range(steps):
                self(s)
                if pbar: pbar.set_description("optimizing %s %d"%(self.scene.scene_uid[:20], s))
        self.show()
    
    def show(self):
        import os, moviepy.editor.ImageSequenceClip as ImageSequenceClip
        for nms in [nms for nms in self.shows if len(self.shows[nms])]: ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)

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
                from ...Semantic import fild
                self.scene.fild = fild(scene,config["grid"],config)

        self.over_states = {"ads":[],"vios":[],"fits":[]}
        self.over_bounds = config["stop"]["over_bounds"]
        self.over_len, self.max_len = config["stop"]["over_len"], config["stop"]["max_len"]
        self.error_bounds = config["stop"]["error_bounds"]

        self.shows = {"res":[],"syn":[],"pnt":[],"pns":[],"fiv":[],"fih":[],"fip":[],"fiq":[]}
        
    # def draw(self,s):
    #     if not self.exp:
    #         r = self.scene.fild() if self.scene.fild else None
    #         for nms in self.shows:
    #             if nms in self.configVis:# and (nms[:2] != "fi"):#
    #                 self.shows[nms].append(self.scene.drao(nms, self.configVis[nms],s))

    def __over(self,ad,vio):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["vios"].append(vio)
        
        if len(self.over_states["ads"]) > self.over_len:
            if vio > self.error_bounds["vios"]:
                print("error: vio=%f"%(vio))
                return None
            self.over_states["ads"].pop(0), self.over_states["vios"].pop(0)
        return False if len(self.over_states["ads"]) < self.over_len else bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["vios"]).mean() < self.over_bounds["vios"])
        
    def eval(self,safe=True):
        if safe: self.scene.OBJES.optimizePhy(self.config,self.timer,debug=True,ut=-1)
        return self.scene.OBJES.violates()
    
    def opt(self,s):
        return self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=self.iRate(s))

    def __call__(self,s):
        import json,os
        PhyRet = self.opt(s) #adjs,vio,self.over(adjs,vio)
        for nms in self.config["vis"]: self.shows[nms].append(self.scene.drao(nms, self.config["vis"][nms], s+1.0)) if nms != "save" else open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+1.0)),"w").write(json.dumps(self.scene.toSceneJson()))
        
        vio = self.eval(safe=False)#self.scene.OBJES.violates() #if self.exp else None
        ad = PhyRet #PhyRet["adjs"]+PatRet["adjs"]

        #open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))

        return {"over": self.__over(ad,vio)}
    
    def show(self):
        import os, moviepy.editor.ImageSequenceClip as ImageSequenceClip
        for nms in [nms for nms in self.shows if len(self.shows[nms])]: ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)

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
        if not exp: # exp did random by itself (because it has to record the result of randomization) , not from us
            if self.prerec: #for recognition: when we know how the original scene is, we use recognition
                from .. import rgnz
                rgnz(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
                #if self.rand: self.__random()
            else: #for rearrangement: when we don't know how the original scene is, we use rearrangement to guess how the semantic of the objects are
                from .. import rarg
                rarg(self.PM,scene,v=0).uncond(use=True,move=True,draw=False)
            for o in [_ for _ in self.scene.OBJES if _.gid == 0]: o.v = False 

        self.over_states = {"ads":[],"fits":[]}
        self.over_bounds = config["stop"]["over_bounds"]
        self.over_len, self.max_len = config["stop"]["over_len"], config["stop"]["max_len"]
        self.error_bounds = config["stop"]["error_bounds"]

        self.shows = {"pat":[]}

    # def draw(self, s):
    #     if not self.exp:
    #         if "pat" in self.configVis:# and (nms[:2] != "fi"):#
    #             self.shows["pat"].append(self.scene.drao("pat", self.configVis["pat"], s))

    # def show(self):
    #     for nms in [nms for nms in self.shows if len(self.shows[nms])]:
    #         from moviepy.editor import ImageSequenceClip
    #         import os
    #         ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)
    
    def __over(self,ad,fit):
        import numpy as np
        self.over_states["ads"].append(ad.Norm()), self.over_states["fits"].append(fit)
        
        if len(self.over_states["ads"]) > self.over_len:
            if fit > self.error_bounds["fits"]:
                print("error: fit=%f"%(fit))
                return None
            self.over_states["ads"].pop(0), self.over_states["fits"].pop(0)
        return False if len(self.over_states["ads"]) < self.over_len else bool(np.array(self.over_states["ads"]).mean() < self.over_bounds["ads"] and np.array(self.over_states["fits"]).mean() < self.over_bounds["fits"])
        
    def eval(self):
        fit,fitmax = self.scene.PLAN.update_fit() #print(fit,fitmax)
        return fitmax-fit

    def opt(self,s):
        return self.scene.PLAN.optimize(self.iRate(s),s)
    
    def __call__(self,s):
        import json,os
        PatRet = self.PatOpt.opt(s) #adjs,fit,self.over(adjs,vio)
        for nms in self.config["vis"]: self.shows[nms].append(self.scene.drao(nms, self.config["vis"][nms], s+0.5)) if nms != "save" else open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))
        
        fit = self.eval()#,fitmax = self.scene.PLAN.update_fit() #if self.exp else (0,0)
        ad = PatRet #PhyRet["adjs"]+PatRet["adjs"]

        #open(os.path.join(self.scene.imgDir,"%.1f.json"%(s+0.5)),"w").write(json.dumps(self.scene.toSceneJson()))

        return {"over": self.__over(ad,fit)}
    
    def loop(self, steps=-1, pbar=None): #an example of loop, but it's recommended to call the __call__ directly
        [o.adjust.clear() for o in self.scene.OBJES]
        if steps<0:
            while True:
                ret,s = {"over":False},0
                while (ret["over"] is False) and s <= self.max_len: #the over criterion #assert ret["over"] is False and ret["over"] is not None
                    ret, s = self(s), s+1
                if ret["over"] is True and s <= self.max_len: break #if the over criterion is met, break
                else:                                               #if the restart criterion is met, restart
                    print("restart",self.scene.scene_uid)
                    adjs0 = self.__random(2.0,use=False)
        else:
            for s in range(steps):
                self(s)
                if pbar: pbar.set_description("optimizing %s %d"%(self.scene.scene_uid[:20], s))
        self.show()
    
    def show(self):
        import os, moviepy.editor.ImageSequenceClip as ImageSequenceClip
        for nms in [nms for nms in self.shows if len(self.shows[nms])]: ImageSequenceClip(self.shows[nms], fps=3).write_videofile(os.path.join(self.scene.imgDir,nms+".mp4"),logger=None)
"""
