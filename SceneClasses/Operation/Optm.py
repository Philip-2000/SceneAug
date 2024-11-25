class optm():
    def __init__(self,pm=None,scene=None,PatFlag=False,PhyFlag=True,rand=False,rec=True,config={}):
        self.scene = scene #assert PhyFlag or PatFlag
        self.PatOpt = None if not PatFlag else PatOpt(pm,scene,rec=rec,config=config["pat"]) 
        self.PhyOpt = None if not PhyFlag else PhyOpt(scene,config=config["phy"])
        if rand:
            self.random()

    def random(self):
        for o in self.scene.OBJES:
            if o.idx % 3 == 0:
                o.translation -= 1.0*o.direction()
    
    def __call__(self, steps=100, iRate=0.01, jRate=0.01):
        [(self.PhyOpt(iRate,s) if self.PhyOpt else None, self.PatOpt(jRate,s) if self.PatOpt else None) for s in range(steps)]
        
class PhyOpt():
    def __init__(self,scene,config={}):
        self.scene = scene
        from ..Semantic.Fild import fild
        self.scene.fild = fild(scene,config["grids"],config)
        
        from ..Semantic.Fild import fild
        self.config = config #config for operations on optimization

        self.configVis = config if config["vis"] else None
        
        #要给这PhyOpt和PatOpt配置可视化方案

    def draw(self):

        self.scene.OBJES.drao("res", self.configVis["res"])
        self.scene.OBJES.drao("syn", self.configVis["syn"])
        self.scene.OBJES.drao("pnt", self.configVis["pnt"])
        self.scene.OBJES.drao("pns", self.configVis["pns"])

        self.scene.fild.draw("fild", self.configVis["fild"])
        self.scene.fild.draw("pot", self.configVis["pot"])
        #res中需要把各个成分场都给出来吗？
        #采样点是物体的语义？有点道理？
        #物体位移是物体的语义？
        #场是场景的语义？
        #在哪里？在self.vis中操作这些对象的对应可视化操作，将他们操作出来。
        pass

    #region: -hyper-parameter--------#
    
    #endregion: -hyper-parameter-----#


    def __call__(self,iRate,s):
        self.scene.OBJES.optimizePhy(self,self.config,debug=(self.ObjectVis or self.FieldsVis),ut=iRate)
        r = self.scene.fild() if self.scene.fild else None            
        self.draw()

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