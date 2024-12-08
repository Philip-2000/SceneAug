import numpy as np
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
    
    def __call__(self, steps=100, iRate=-1, jRate=-1):
        if steps>0:
            for s in range(steps):
                if self.PhyOpt:
                    self.PhyOpt(s,iRate)
                if self.PatOpt:
                    self.PatOpt(s,jRate)
        else:
            PhyRet,PatRet,s = (False),(False),0
            while (PhyRet[-1] and PatRet[-1]): #the over criterion
                if self.PhyOpt:
                    PhyRet = self.PhyOpt(s,iRate)
                if self.PatOpt:
                    PatRet = self.PhyOpt(s,iRate)
                s=s+1
        self.PhyOpt.show()
        #well we shouldn't go this way? 
        
class PhyOpt():
    def __init__(self,scene,config={},exp=False):
        self.scene = scene
        self.config= config
        self.iRate = config["rate"]
        self.s4 = config["s4"]
        
        self.configVis = config["vis"] if config["vis"] else None
        if self.configVis and (self.configVis["fiv"] or self.configVis["fih"] or self.configVis["fip"] or self.configVis["fiq"]):
            from ..Semantic.Fild import fild
            self.scene.fild = fild(scene,config["grid"],config)
        self.shows = {"res":[],"syn":[],"pnt":[],"pns":[],"fiv":[],"fih":[],"fip":[],"fiq":[]}
        self.steps = 0

        from ..Experiment.Tmer import tmer,tme
        self.timer = tmer() if exp else tme()
        self.exp = exp
        #还有一个就是“timer”吗？因为之前的都是离线操作、要么就是离线的“场景识别”、要么就是场景生成但没来及写计时器的
        #所以其实我从来也没写过计时器

    def draw(self,s):
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
    
    def over(self,mod,vio):
        return False

    def __call__(self,s,ir=-1):
        mod = self.scene.OBJES.optimizePhy(self.config,self.timer,debug=bool(self.configVis),ut=(self.iRate if ir<0 else ir))
        #[(T,S,R),(T,S,R),(T,S,R),......]
        vio = self.scene.OBJES.violates() if self.exp else None #[SumOfNorm(s.t),SumOfNorm(s.t),SumOfNorm(s.t),......]
        self.draw(s)
        self.steps = max(s,self.steps)
        return mod,vio,self.timer,self.over(mod,vio)

    def log(self):
        #我觉得像咱们的PhyOpt对象，要对此次的场景、学习率、config等超参数有所感知
            #学习率？
            #采样点采样率
        #并且具备对外输出某些评估指标的能力
            #逐帧的：
                #物体调整的尺度
                #总的violate，即各个采样点的受场力情况
                #耗时
            #总体的：
                #收敛总步数
                #收敛总时间
                                                            #不过它没必要管理各大超参数的关系，
        pass


class PatOpt():
    def __init__(self,pm,scene,config={},rerec=False,prerec=False,rand=False,exp=False): #while in use: rerec=False, prerec=True?, rand=False
        self.PM = pm
        self.scene = scene
        self.rerec = rerec
        self.prerec= prerec and not rerec
        self.iRate = config["rate"]

        from ..Experiment.Tmer import tmer,tme
        self.timer = tmer() if exp else tme()
        self.exp = exp
        self.steps = 0
        from .Plan import plans
        if rand and prerec:
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
            self.__random()
        elif rand:
            self.__random()
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        elif prerec:
            plans(scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
        self.shows = {"pat":[]}

    def __random(self):
        for o in self.scene.OBJES:
            if o.idx % 2 == 0:
                o.translation -= 0.8*o.direction()

    def draw(self,s):
        self.shows["pat"].append(self.scene.drao("pat", self.configVis["pat"],s))

    def show(self):
        from moviepy.editor import ImageSequenceClip
        import os
        ImageSequenceClip(self.shows["pat"], fps=2).write_videofile(os.path.join(self.scene.imgDir,"pat.mp4"),logger=None)
    
    def over(self,mod,score):
        return False
    
    def optimize(self,ir):
        self.scene.grp=True
        j = -1
        for p in self.plas:
            if len(p)<=1:
                continue
            j += 1
            for oid,nid in p.nids[1:]:
                assert self.scene[oid].nid == nid and self.scene[oid].gid == j
                son = self.scene[oid]
                m = self.pm.nods[self.pm.mid(nid)]

                fid = p.searchOid(m.idx)[0]
                assert self.scene[fid].nid == m.idx
                fat = self.scene[fid]
                fat_son = fat - son #？？？？？
                fat_son = m.bunches[nid].optimize(fat_son,(self.iRate if ir <0 else ir))
                new_son = fat + fat_son
                if self.exp:
                    son.adjust["T"] = new_son.translation - son.translation
                    son.adjust["S"] = new_son.size - son.size
                    son.adjust["R"] = new_son.orientation - son.orientation
                son.translation,son.size,son.orientation = new_son.translation,new_son.size,new_son.orientation
        return [(son.adjust["T"],son.adjust["S"],son.adjust["R"]) for son in self.scene.OBJES]
    
    def __call__(self,s,ir=-1):
        #handly search the father and son relation among objects,
        if self.rerec:
            self.timer("rec",1)
            from .Plan import plans
            plans(self.scene,self.PM,v=0).recognize(use=True,draw=False,show=False)
            self.timer("rec",0)
        
        self.timer("",1)
        mod = self.optimize(ir)
        self.timer("opt",0)
        fit = self.scene.plan.update_fit() if self.exp else 0
        self.draw(s)
        self.steps = max(s,self.steps)
        return mod,fit,self.timer,self.over(mod,fit)
    

    #沟槽的周日计划：
    #（0）把PhyOpt的文件系统核对一下，把obje结构调整之后以及rela和rely函数删去之后的那个函数场景识别方法甚至是模式构建方法核对一下？这可麻烦了
    #（1）把PatOpt做得ok了
    #（2）把PatExop做得ok了
    #（3）开启早期实验，确认实验数据的生成、存储和可视化全过程ok
    #沟槽的周一计划：
    #（4）逐步向PhyExop和双向的Exop发展
    #

    def run(self):
        #我觉得像咱们的PatOpt对象，要对此次的场景、学习率、config等超参数有所感知
            #学习率？
            #场景识别方式？
        #并且具备对外输出某些评估指标的能力
            #逐帧的：
                #物体调整的尺度
                #recognize评分
                #耗时
            #总体的：
                #收敛总步数
                #收敛总时间
                                                            #不过它没必要管理各大超参数的关系，

        
        pass