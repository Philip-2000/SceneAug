from ...Semantic import plan
class rgnz():
    def __init__(self,scene,pm,v=0):
        self.verb=v
        self.iRate=0.6
        self.scene = scene
        self.pm = pm
        self.currentPlan = plan(scene,pm)
        self.segments = len(self.currentPlan.PLAN)
        if self.verb > 1:
            print(self.segments)
        self.plans = [self.currentPlan]
        self.showTitles = []

    def addPlan(self,Plan): #bounded insert sort
        for _ in range(20):
            if _ >= len(self.plans) or Plan.fit > self.plans[_].fit:
                self.plans.insert(_,Plan)
                if len(self.plans) > 20:
                    self.plans.pop(-1)

    def spatiallyRecorrect(self,Plan):
        from ...Semantic import grup
        j = -1
        orphans = []
        for o in self.scene.OBJES:
            ar = [p for p in Plan.PLAN if o.idx in [on[0] for on in p.nids]]
            assert len(ar) <= 1
            if len(ar) == 0 or len(ar[0].nids)==1:
                orphans.append(o.idx)
        tmpGrups = {}
        for ip in range(len(Plan.PLAN)):
            p = Plan.PLAN[ip]
            if len(p)<=1:
                continue
            tmpGrups[ip] = grup([on[0] for on in p.nids],idx=j+1,scne=self.scene)#,{"sz":self.scene.roomMask.shape[-1],"rt":16},j+1,scne=self.scene)

        for oid in orphans:
            o = self.scene[oid]
            oPolygon = o.shape()
            oArea = oPolygon.area
            vs = []
            for k,v in tmpGrups.items():
                vPolygon = v.shape()
                iArea = vPolygon.intersection(oPolygon).area
                if (iArea / oArea)>self.iRate:
                    vs.append(k)
            
            if len(vs) == 1:
                Plan.PLAN[vs[0]].nids.append((oid,Plan.PLAN[vs[0]].nids[-1][1]))

        #a more aggresive version of spatially recorrection will drag the not orphans in other segments, but not done yet
        pass

    def recognize(self,use=True,draw=True,show=False):        
        if show:
            self.show()
            return None,None,None
        if self.segments > 4:
            return 0,0,None
        import math
        if self.verb > 1:
            print("start recognize, fit is %.5f"%(self.currentPlan.fit))
            print(self.currentPlan)
        if len(self.currentPlan) == 0:
            return 0.0,0,None
        self.currentPlan.singleExtend(0)
        if self.verb > 1:
            print("recognize 0, fit is %.5f"%(self.currentPlan.fit))
            print(self.currentPlan)

        R = 1
        for s in range(1,self.segments):
            #assert s<3
            if self.verb > 1:
                print("recognize %d"%(s))
            R = math.prod([len(self.currentPlan.PLAN[t].nids) for t in range(s)])
            if self.verb>2:
                print("R=%d"%(R))

            hint = [-1]+[0 for j in range(s-1)]
            for r in range(R): #-1
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlan.PLAN[i].nids):
                    if self.verb>2:
                        print("in while before")
                        print(hint)
                        print("i=%d"%(i))
                    hint[i],i=0,i+1
                    hint[i]=hint[i]+1
                    if self.verb>2:
                        print("in while after")
                        print(hint)
                        print("i=%d"%(i))

                nowPlan = plan.retreats(self.currentPlan,hint)
                if self.verb > 2:
                    print(hint)
                    print("before, fit is %.5f"%(nowPlan.fit))
                    print(nowPlan)
                    nowPlan.printIds()
                nowPlan.singleExtend(s)
                if self.verb > 2:
                    print("after, fit is %.5f"%(nowPlan.fit))
                    print(nowPlan)
                self.addPlan(nowPlan)

                #assert not (len(hint)==2 and hint[0] == 2 and hint[1] == 0)

            self.currentPlan.PLAN[s] = self.plans[0].PLAN[s]
            if self.verb > 1:
                self.currentPlan.fit = sum([sum(p.fits) for p in self.currentPlan.PLAN])
                print("extend %d over, fit is %.5f"%(s, self.currentPlan.fit))
                print(self.currentPlan)
        
            fixedOids = [on[0] for on in self.currentPlan.PLAN[s].nids]
            for t in range(0,s):
                onId = 0
                while onId < len(self.currentPlan.PLAN[t].nids):
                    on = self.currentPlan.PLAN[t].nids[onId]
                    oid,nids= on[0],[on[1]]
                    if oid in fixedOids:
                        onJd = onId+1
                        while onJd < len(self.currentPlan.PLAN[t].nids):
                            jN = self.currentPlan.PLAN[t].nids[onJd]
                            delete=False
                            for nid in nids:
                                if (jN[1] in self.pm[nid].bunches):
                                    self.currentPlan.PLAN[t].nids.pop(onJd)
                                    self.currentPlan.PLAN[t].fits.pop(onJd)
                                    delete=True
                                    nids.append(jN[1])
                                    break
                            if not delete:
                                onJd += 1
                        self.currentPlan.PLAN[t].nids.pop(onId)
                        self.currentPlan.PLAN[t].fits.pop(onId)
                    else:
                        onId += 1
            
            self.currentPlan.fit = sum([sum(p.fits) for p in self.currentPlan.PLAN])
            if self.verb > 1:
                print("eliminate %d over, fit is %.5f"%(s, self.currentPlan.fit))
                print(self.currentPlan)
        
        if self.verb > 0:
            print("recognize over, fit is %.5f"%(self.currentPlan.fit))
            print(self.currentPlan)
    
        if self.iRate < 1.0:
            self.spatiallyRecorrect(self.currentPlan)

        if self.verb > 0:
            print("recorrect over, fit is %.5f"%(self.currentPlan.fit))
            print(self.currentPlan)
        
        if use:# or opt:
            self.currentPlan.utilize()
        if draw:
            self.scene.draw(drawUngroups=True,classText=True,d=True)
        # if opt:
        #     opts = self.currentPlan.optimize()
        #     if draw:
        #         self.scene.draw(drawUngroups=True,classText=True,d=True)
        return self.currentPlan.fit, sum([len(p) for p in self.currentPlan.PLAN]), None
    
    # def optimize(self,draw=True):
    #     self.recognize(use=True,draw=True,opt=False)
    #     #self.currentPlan.singleExtend(0)

    def addFrame(self,Plan,name,dir,idx=-1,stay=1):#for show
        import os
        if idx>=0:
            zeros = [0 for _ in range(len(Plan))]
            for l in range(len(Plan.PLAN[idx])-1,0,-1):
                zeros[idx] = l
                pl = plan.retreats(Plan, zeros)
                pl.utilize(forShow=True)
                self.scene.draw(imageTitle=os.path.join(dir,name+str(l)+".png"))
                self.showTitles.append(name+str(l))
        Plan.utilize(forShow=True)
        self.scene.draw(imageTitle=os.path.join(dir,name+".png"))
        for _ in range(stay):
            self.showTitles.append(name)

    def show(self): #show the recognizing process with several frames,
        from moviepy.editor import ImageSequenceClip
        import math,os
        thisDir = os.path.join(self.scene.imgDir,self.scene.scene_uid)
        os.makedirs(thisDir,exist_ok=True)
        #self.showTitles = ['first3', 'first2', 'first1', 'first', 'first', 'first', 'second02', 'second01', 'second0', 'second0', 'second0', 'second12', 'second11', 'second1', 'second1', 'second1', 'second22', 'second21', 'second2', 'second2', 'second2', 'over1', 'over1', 'over1', 'spatial', 'spatial']
        #ImageSequenceClip([os.path.join(thisDir,t+".png") for t in self.showTitles], fps=5).write_videofile(os.path.join(thisDir,"recognize.mp4"),logger=None)#return

        self.currentPlan.singleExtend(0)
        self.addFrame(self.currentPlan,"first",thisDir,0,3)              #--------------------------go first

        for s in range(1,self.segments):
            #assert s<3
            R = math.prod([len(self.currentPlan.PLAN[t].nids) for t in range(s)])

            hint = [0 for j in range(s)]
            for r in range(R-1):
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlan.PLAN[i].nids):
                    hint[i],i=0,i+1
                    hint[i]=hint[i]+1

                nowPlan = plan.retreats(self.currentPlan,hint)
                nowPlan.singleExtend(s)
                self.addPlan(nowPlan)
                if s==1:
                    self.addFrame(nowPlan,"second%d"%(r),thisDir,1,3)    #--------------------------squeeze the first and get second
            
            self.currentPlan.PLAN[s] = self.plans[0].PLAN[s]
        
            fixedOids = [on[0] for on in self.currentPlan.PLAN[s].nids]
            for t in range(0,s):
                onId = 0
                while onId < len(self.currentPlan.PLAN[t].nids):
                    on = self.currentPlan.PLAN[t].nids[onId]
                    oid,nid= on[0],on[1]
                    if oid in fixedOids:
                        onJd = onId+1
                        while onJd < len(self.currentPlan.PLAN[t].nids):
                            jN = self.currentPlan.PLAN[t].nids[onJd]
                            if (jN[1] in self.pm[nid].bunches):
                                self.currentPlan.PLAN[t].nids.pop(onJd)
                            else:
                                onJd += 1
                        self.currentPlan.PLAN[t].nids.pop(onId)
                    else:
                        onId += 1
            
            self.addFrame(self.currentPlan,"over%d"%(s),thisDir,-1,3)             #--------------------------squeeze the existing and get next, (directly get the best one, ok)

        if self.iRate < 1.0:
            self.spatiallyRecorrect(self.currentPlan)
            self.addFrame(self.currentPlan,"spatial",thisDir,-1,2)                #--------------------------spatially recorrect,
        self.showTitles = self.showTitles[:1]*2+self.showTitles #print(len(self.showTitles))
        ImageSequenceClip([os.path.join(thisDir,t+".png") for t in self.showTitles], fps=2).write_videofile(os.path.join(thisDir,"recognize.mp4"),logger=None)#
       