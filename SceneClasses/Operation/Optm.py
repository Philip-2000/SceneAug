class optm():
    def __init__(self,pm=None,scene=None,PatFlag=False,PhyFlag=True,rand=False,rec=True,config={}):
        self.scene = scene
        assert PhyFlag 
        assert not PatFlag
        self.PatOpt = None if not PatFlag else PatOpt(pm,scene,rec=rec,config=config["pat"]) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,config=config["phy"])
        _           = None if not rand    else self.__random()

    def __random(self):
        for o in self.scene.OBJES:
            if o.idx % 2 == 0:
                o.translation -= 0.8*o.direction()
    
    def __call__(self, steps=100, iRate=0.01, jRate=0.01):
        [(self.PhyOpt(iRate,s) if self.PhyOpt else None, self.PatOpt(jRate,s) if self.PatOpt else None) for s in range(steps)]
        self.PhyOpt.show()
        #well we shouldn't go this way? 
        
class PhyOpt():
    def __init__(self,scene,config={}):
        self.scene = scene
        from ..Semantic.Fild import fild
        self.scene.fild = fild(scene,config["grid"],config)
        self.config = config
        self.configVis = config["vis"] if config["vis"] else None
        
        #要给这PhyOpt和PatOpt配置可视化方案
        self.shows = {"res":[],"syn":[],"pnt":[],"pns":[]}

    def draw(self,s):
        _ = self.scene.drao("res", self.configVis["res"],s) if self.configVis["res"] else None
        if _:
            self.shows["res"].append(_)
        _ = self.scene.drao("syn", self.configVis["syn"],s) if self.configVis["syn"] else None
        if _:
            self.shows["syn"].append(_)
        _ = self.scene.drao("pnt", self.configVis["pnt"],s) if self.configVis["pnt"] else None
        if _:
            self.shows["pnt"].append(_)
        _ = self.scene.drao("pns", self.configVis["pns"],s) if self.configVis["pns"] else None
        if _:
            self.shows["pns"].append(_)
        _ = self.scene.drao("fiv", self.configVis["fiv"],s) if self.configVis["fiv"] else None
        assert False
        # _ = self.scene.drao("fih", self.configVis["fih"],s) if self.configVis["fih"] else None
        # _ = self.scene.drao("fip", self.configVis["fip"],s) if self.configVis["fip"] else None
        # _ = self.scene.drao("fiq", self.configVis["fiq"],s) if self.configVis["fiq"] else None

    def show(self):
        from moviepy.editor import ImageSequenceClip
        import os
        if "res" in self.shows:
            ImageSequenceClip(self.shows["res"], fps=2).write_videofile(os.path.join(".","res.mp4"),logger=None)
        if "syn" in self.shows:
            ImageSequenceClip(self.shows["syn"], fps=2).write_videofile(os.path.join(".","syn.mp4"),logger=None)
        if "pnt" in self.shows:
            ImageSequenceClip(self.shows["pnt"], fps=2).write_videofile(os.path.join(".","pnt.mp4"),logger=None)
        if "pns" in self.shows:
            ImageSequenceClip(self.shows["pns"], fps=2).write_videofile(os.path.join(".","pns.mp4"),logger=None)
       
        
        pass

    def __call__(self,iRate,s):
        self.scene.OBJES.optimizePhy(
            self.config,
            debug=bool(self.configVis),
            ut=iRate
        )
        #r = self.scene.fild() if self.scene.fild else None            
        self.draw(s)

class PatOpt():
    def __init__(self,pm,scene,rec=True,config={}):
        from .Patn import patternManager as PM
        from .Plan import plans
        self.PM = pm
        self.scene = scene
        if rec:
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)

    def __call__(self,iRate,s):
        #handly search the father and son relation among objects,

        self.scene.grp=True
        j = -1
        for p in self.plas:
            if len(p)<=1:
                continue
            j += 1
            for oid,nid in p.nids[1:]:
                assert self.scene[oid].nid == nid
                assert self.scene[oid].gid == j
                son = self.scene[oid]


                m = self.pm.nods[self.pm.mid(nid)]

                fid = p.searchOid(m.idx)[0]
                assert self.scene[fid].nid == m.idx
                fat = self.scene[fid]
                fat_son = fat - son #？？？？？
                fat_son = m.bunches[nid].optimize(fat_son)
                new_son = fat + fat_son
                son.translation,son.size,son.orientation = new_son.translation,new_son.size,new_son.orientation

        pass
