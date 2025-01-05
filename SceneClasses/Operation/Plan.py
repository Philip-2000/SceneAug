from itertools import chain
from copy import deepcopy

class pla():
    def __init__(self,nids,fits=[0]): #nids should not be empty, even when it is been sent in
        self.fits,self.nids=fits,nids #[(oid,nid),(oid,nid)......]

    def Str(self,s,pm):
        return "(%s,%d,%d)\n"%(s[self.nids[0][0]].class_name()[:10],self.nids[0][0],self.nids[0][1]) + \
                "\n".join(
                        [
                            " ".join(["\t"]*(pid+1)) + "(%s,%d,%d->(%s,%d,%d))"%(
                            s[p[0]].class_name()[:10],p[0],p[1],
                            s[self.searchOid(pm[p[1]].mid)].class_name()[:10],self.searchOid(pm[p[1]].mid),pm[p[1]].mid
                            ) for pid, p in enumerate(self.nids[1:])
                        ]
                    )

    def __len__(self):
        return len(self.nids)
    
    def searchOid(self,nid):
        return [on[0] for on in self.nids if on[1] == nid][0]

    @classmethod
    def extend(cls,a,oid,nid,v):
        return pla(a.nids+[(oid,nid)],a.fits+[v])
    
    @classmethod
    def retreat(cls,a):
        return pla(deepcopy(a.nids[:-1]),deepcopy(a.fits[:-1]))
    
    def occupied(self):
        return [a[0] for a in self.nids]

    def utilize(self, g, scene, pm):
        from ..Semantic.Link import objLink
        from ..Semantic.Grup import grup
        scene[self.nids[0][0]].nid = self.nids[0][1]
        scene[self.nids[0][0]].gid = g
        for i, (oid,nid) in enumerate(self.nids[1:]):
            scene[oid].nid,scene[oid].gid = nid, g
            scene.LINKS.append(objLink(oid,self.nids[i][0],len(scene.LINKS),scene,"lightblue"))
            scene.LINKS.append(objLink(oid,self.searchOid(pm[nid].mid),len(scene.LINKS),scene,"pink"))

        scene.GRUPS.append(grup([on[0] for on in self.nids],{"sz":scene.roomMask.shape[-1],"rt":16},g,scne=scene))

    def optimize(self,g,scene,pm,ir,s): #print(self.Str(scene,pm))
        from .Bnch import bnch_tree
        return bnch_tree(self,pm,scene).optimize(ir,s) 
        # bt = bnch_tree(self,pm,scene)
        # print(bt)
        # return bt.optimize(ir)

    def update_fit(self, g, scene, pm):
        from .Bnch import singleMatch
        for i,(oid,nid) in enumerate(self.nids[1:]):
            if nid == self.nids[i][1]: #it came from spatially recorrect
                continue
            m, fid = pm[pm[nid].mid], self.searchOid(pm[nid].mid) #m, fid = pm.nods[pm.mid(nid)], self.searchOid(pm.mid(nid))
            assert scene[oid].nid == nid and scene[oid].gid == g and scene[fid].nid == m.idx
            loss = m.bunches[nid].loss(scene[fid] - scene[oid])
            self.fits[i+1] = singleMatch(loss,None,None,None,None)

class plas(): #it's a recognize plan
    def __init__(self,scene=None,pm=None,base=None):
        if base is None:
            self.scene,self.pm=scene,pm
            self.plas = [pla([(o.idx, [ed.endNode.idx for ed in pm[0].edges if ed.endNode.type == pm.merging[o.class_name()]].pop() )]) for o in scene.OBJES if (pm.merging[o.class_name()] in pm.rootNames)]
        else:
            self.scene,self.pm=base.scene,base.pm
            self.plas = deepcopy(base.plas)
        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.plas])))
        self.fit = sum([sum(p.fits) for p in self.plas])
        self.myPM = None

    def __str__(self):
        return "\n".join([ "%d\t"%(pid) + p.Str(self.scene,self.pm) for pid,p in enumerate(self.plas)])

    def __iter__(self):
        return iter(self.plas)

    def formPM(self):
        import numpy as np
        from .Bnch import bnch
        from .Patn import patternManager as PM
        from ..Basic.Obje import obje
        
        pm = PM(vers="tmp",new=True)
        pm.rootNames = [ pm.merging[self.scene[p.nids[0][0]].class_name()] for p in self.plas ]
        for p in self.plas:
            newNids, fid = {0:0},0
            for oid,nid in p.nids:
                o,mid = self.scene[oid], self.pm[nid].mid#m.idx
                O = obje(o.translation,np.array([1,1,1]),o.orientation,i=0) if p.nids.index((oid,nid))==0 else self.scene[p.searchOid(mid)]
                fid = pm.createNode(pm[fid],self.pm.merging[o.class_name()],1,1)
                #pm.nods[newNids[mid]].bunches[fid], newNids[nid] = bnch(None,O.rela(o,self.pm.scaled).flat(),np.abs(O.rela(o,self.pm.scaled).flat()-self.pm.nods[mid].bunches[nid].exp)), fid
                pm[newNids[mid]].bunches[fid], newNids[nid] = bnch(None,(O-o).flat(),np.abs((O-o).flat()-self.pm[mid].bunches[nid].exp)), fid
        return pm

    def diff(self,scene,ref=None,v=0):
        if self.myPM is None:
            self.myPM = self.formPM()
            if v>0:
                print(self.myPM)
        fit,ass,_ = plans(scene,self.myPM,v=v).recognize(use=False,draw=False,show=False)#assert fit < ref
        if v>0:
            print(fit)
            print(self.fit)
        return abs(self.fit - fit)

    def printIds(self):
        return
        print("occupied" +"+".join(["(%s,%d)"%(self.scene[id].class_name(),id) for id in self.occupied]) )

    def __len__(self):
        return len(self.plas)

    def plaExtend(self,p,selections):
        from .Bnch import singleMatch
        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.plas])))
        cs=0
        for ed in self.pm[p.nids[-1][1]].edges:
            m = self.pm[self.pm[ed.endNode.idx].mid]
            a = self.scene[p.searchOid(m.idx)]
            self.printIds()
            losses = [(oo,m.bunches[ed.endNode.idx].loss(a-oo)) for oo in [o for o in self.scene.OBJES if (self.pm.merging[o.class_name()]==ed.endNode.type and (o.idx not in self.occupied and o.idx not in [on[0] for on in p.nids] ))]] #print(str(lev)+" loop: " + ed.endNode.type + " nid=" + str(ed.endNode.idx) + " idx=" + str(o.idx) + " mid=" + str(m.idx))
            if m.idx==1 and ed.endNode.idx == 254 and False:#
                print("m.idx=%d,m.oid=%d,new.idx=%d"%(m.idx,a.idx,ed.endNode.idx))
                print("exp")
                print(m.bunches[ed.endNode.idx].exp)
                for oolosses in losses:
                    oo,loss = oolosses[0],oolosses[1]#[0] 
                    print("id=%d"%(oo.idx))
                    print("absolute")
                    print(oo.flat())
                    

                    print("(%s,%d):%.3f"%(oolosses[0].class_name(),oolosses[0].idx,oolosses[1]))
                    pass
                    
                
                print("\t".join([ "(%s,%d):%.3f"%(oolosses[0].class_name(),oolosses[0].idx,oolosses[1]) for oolosses in losses]))
            self.printIds()
            for oolosses in sorted(losses,key=lambda x:x[1]):
                oo,loss = oolosses[0],oolosses[1]#[0] 
                #print(oo.idx)
                v = singleMatch(loss,ed.confidence,ed.confidenceIn,self.pm[p.nids[-1][1]].edges.index(ed),cs)
                selections.append(pla.extend(p,oo.idx,ed.endNode.idx,v)) 
                break
            cs += ed.confidence

    def singleExtend(self,pid):
        #self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.plas])))
        #print(self.occupied)
        #print(self)
        self.printIds()
        selections,id = [deepcopy(self.plas[pid])],0
        while id<len(selections):
            self.plaExtend(selections[id],selections)
            id+=1
        self.plas[pid] = sorted(selections,key=lambda x:-sum(x.fits))[0]

        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.plas])))
        self.fit = sum([sum(p.fits) for p in self.plas])
        return self.plas[pid]
    
    @classmethod
    def retreats(cls,self,hint):
        newPlas = cls(base=self)
        for i in range(len(hint)):
            for j in range(hint[i]):
                newPlas.plas[i] = pla.retreat(newPlas.plas[i])
        newPlas.occupied = chain(*[p.occupied() for p in newPlas.plas])
        newPlas.fit = sum([sum(p.fits) for p in newPlas.plas])
        return newPlas

    def utilize(self,clear=True,forShow=False):
        if clear:
            self.scene.LINKS.clear()
            self.scene.GRUPS.clear()
            for o in self.scene.OBJES:
                o.gid,o.nid=0,0
        self.scene.grp, self.scene.plan = True,self
        [p.utilize(g+1,self.scene,self.pm) for g,p in enumerate([p for p in self.plas if len(p)>1 or forShow])]

    def optimize(self, ir, s):
        #_ = self.optinit() if s==0 else None
        Is = [p.optimize(g+1,self.scene,self.pm,ir,s) for g,p in enumerate([_ for _ in self.plas if len(_) > 1])]
        from ..Operation.Adjs import adjs
        return adjs(self.scene.OBJES), Is#[(son.adjust["T"],son.adjust["S"],son.adjust["R"]) for son in self.scene.OBJES]
    
    def update_fit(self):
        [p.update_fit(g+1,self.scene,self.pm) for g,p in enumerate([_ for _ in self.plas if len(_) > 1])]
        self.fit = sum([sum(p.fits) for p in self.plas])
        return self.fit
    
    def __getitem__(self, k):
        return self.plas[k]

    def __iter__(self):
        return iter(self.plas)
    
    def optinit(self):
        #this operation is right after the randomization, so it's operation could be regarded as the part of the randomization
        #during ExOp, this operation is recorded with the randomization result to be the so called, "ground truth"
        #but however, during real utilization where we want to optimize a real scene, we can also use this function to find a better initialization state
        #so the thing is a little bit complex anyway
        import numpy as np
        xs,zs = [w.p[0] for w in self.scene.WALLS],[w.p[2] for w in self.scene.WALLS]
        assert abs(np.max(xs)+np.min(xs))<0.01 and abs(np.max(zs)+np.min(zs))<0.01
        if len([p for p in self.plas if len(p)>1]) == 1:
            p = [p for p in self.plas if len(p)>1][0]
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c = (wmax.center()*(wmax.length+1.0) + wmin.center()*(wmin.length+1.0))/(wmax.length + wmin.length + 2.0)#np.array([np.max(xs), 0.0, (wmax.p[2]+wmax.q[2])/2.0])
            M = self.scene[p.nids[0][0]].translation.copy() - c 
            [self.scene[oid].adjust.update(T=-M, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p.nids ]
        elif len([p for p in self.plas if len(p)>1]) == 2:
            p1,p2 = [p for p in self.plas if len(p)>1]
            hint = [2,3,1,4,5] #I'm done with it ok? this hint is only for 'losy', I would never coding like this shit if I don't have to finish papers
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            if (wmin.length > wmax.length) ^ (hint.index(p1.nids[0][1])>hint.index(p2.nids[0][1])): #wmin's area is larger than wmax's 
                p1,p2 = p2,p1 #then p1 should be at wmax, p2 at wmin 
            # if np.max(xs) > np.max(zs):
            #     c1,c2 = wmax.center() * 0.4, wmin.center()*0.4 #np.array([np.max(xs)*0.3, 0.0, (wmax.p[2]+wmax.q[2])/2.0]), np.array([np.min(xs)*0.3, 0.0, (wmin.p[2]+wmin.q[2])/2.0])
            # else:
            #     c1,c2 = wmax.center() * 0.4, wmin.center()*0.4 #np.array([(wmax.p[0]+wmax.q[0])*0.5, 0.0, np.max(zs)*0.3]), np.array([(wmin.p[0]+wmin.q[0])*0.5, 0.0, np.min(zs)*0.3])
            #print(wmax, wmax.center(), wmax.center()*0.5)
            #print(wmin, wmin.center(), wmin.center()*0.5)
            

            M1,M2 = self.scene[p1.nids[0][0]].translation.copy() - wmax.center() * ((wmax.length+1.0)/(wmax.length + wmin.length + 2.0)), self.scene[p2.nids[0][0]].translation.copy() - wmin.center()*((wmin.length+1.0)/(wmax.length + wmin.length + 2.0))
            [self.scene[oid].adjust.update(T=-M1, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p1.nids]
            [self.scene[oid].adjust.update(T=-M2, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p2.nids]
        else:
            print(self.scene)
            raise Exception(self.scene.scene_uid +  "plas.optinit: the number of groups is %d not 1 or 2"%(len([p for p in self.plas if len(p)>1])))
        from .Adjs import adjs
        return adjs(self.scene.OBJES)

class plans():
    def __init__(self,scene,pm,v=0):
        self.verb=v
        self.iRate=0.6
        self.scene = scene
        self.pm = pm
        self.currentPlas = plas(scene,pm)
        self.segments = len(self.currentPlas.plas)
        if self.verb > 1:
            print(self.segments)
        self.plases = [self.currentPlas]
        self.showTitles = []

    def addPlas(self,Plas): #bounded insert sort
        for _ in range(20):
            if _ >= len(self.plases) or Plas.fit > self.plases[_].fit:
                self.plases.insert(_,Plas)
                if len(self.plases) > 20:
                    self.plases.pop(-1)

    def spatiallyRecorrect(self,plas):
        from ..Semantic.Grup import grup
        j = -1
        orphans = []
        for o in self.scene.OBJES:
            ar = [p for p in plas.plas if o.idx in [on[0] for on in p.nids]]
            assert len(ar) <= 1
            if len(ar) == 0 or len(ar[0].nids)==1:
                orphans.append(o.idx)
        tmpGrups = {}
        for ip in range(len(plas.plas)):
            p = plas.plas[ip]
            if len(p)<=1:
                continue
            tmpGrups[ip] = grup([on[0] for on in p.nids],{"sz":self.scene.roomMask.shape[-1],"rt":16},j+1,scne=self.scene)

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
                plas.plas[vs[0]].nids.append((oid,plas.plas[vs[0]].nids[-1][1]))

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
            print("start recognize, fit is %.5f"%(self.currentPlas.fit))
            print(self.currentPlas)
        if len(self.currentPlas) == 0:
            return 0.0,0,None
        self.currentPlas.singleExtend(0)
        if self.verb > 1:
            print("recognize 0, fit is %.5f"%(self.currentPlas.fit))
            print(self.currentPlas)

        R = 1
        for s in range(1,self.segments):
            #assert s<3
            if self.verb > 1:
                print("recognize %d"%(s))
            R = math.prod([len(self.currentPlas.plas[t].nids) for t in range(s)])
            if self.verb>2:
                print("R=%d"%(R))

            hint = [-1]+[0 for j in range(s-1)]
            for r in range(R): #-1
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlas.plas[i].nids):
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

                nowPlas = plas.retreats(self.currentPlas,hint)
                if self.verb > 2:
                    print(hint)
                    print("before, fit is %.5f"%(nowPlas.fit))
                    print(nowPlas)
                    nowPlas.printIds()
                nowPlas.singleExtend(s)
                if self.verb > 2:
                    print("after, fit is %.5f"%(nowPlas.fit))
                    print(nowPlas)
                self.addPlas(nowPlas)

                #assert not (len(hint)==2 and hint[0] == 2 and hint[1] == 0)

            self.currentPlas.plas[s] = self.plases[0].plas[s]
            if self.verb > 1:
                self.currentPlas.fit = sum([sum(p.fits) for p in self.currentPlas.plas])
                print("extend %d over, fit is %.5f"%(s, self.currentPlas.fit))
                print(self.currentPlas)
        
            fixedOids = [on[0] for on in self.currentPlas.plas[s].nids]
            for t in range(0,s):
                onId = 0
                while onId < len(self.currentPlas.plas[t].nids):
                    on = self.currentPlas.plas[t].nids[onId]
                    oid,nids= on[0],[on[1]]
                    if oid in fixedOids:
                        onJd = onId+1
                        while onJd < len(self.currentPlas.plas[t].nids):
                            jN = self.currentPlas.plas[t].nids[onJd]
                            delete=False
                            for nid in nids:
                                if (jN[1] in self.pm[nid].bunches):
                                    self.currentPlas.plas[t].nids.pop(onJd)
                                    self.currentPlas.plas[t].fits.pop(onJd)
                                    delete=True
                                    nids.append(jN[1])
                                    break
                            if not delete:
                                onJd += 1
                        self.currentPlas.plas[t].nids.pop(onId)
                        self.currentPlas.plas[t].fits.pop(onId)
                    else:
                        onId += 1
            
            self.currentPlas.fit = sum([sum(p.fits) for p in self.currentPlas.plas])
            if self.verb > 1:
                print("eliminate %d over, fit is %.5f"%(s, self.currentPlas.fit))
                print(self.currentPlas)
        
        if self.verb > 0:
            print("recognize over, fit is %.5f"%(self.currentPlas.fit))
            print(self.currentPlas)
    
        if self.iRate < 1.0:
            self.spatiallyRecorrect(self.currentPlas)

        if self.verb > 0:
            print("recorrect over, fit is %.5f"%(self.currentPlas.fit))
            print(self.currentPlas)
        
        if use:# or opt:
            self.currentPlas.utilize()
        if draw:
            self.scene.draw(drawUngroups=True,classText=True,d=True)
        # if opt:
        #     opts = self.currentPlas.optimize()
        #     if draw:
        #         self.scene.draw(drawUngroups=True,classText=True,d=True)
        return self.currentPlas.fit, sum([len(p) for p in self.currentPlas.plas]), None
    
    # def optimize(self,draw=True):
    #     self.recognize(use=True,draw=True,opt=False)
    #     #self.currentPlas.singleExtend(0)

    def addFrame(self,plas,name,dir,idx=-1,stay=1):#for show
        import os
        if idx>=0:
            zeros = [0 for _ in range(len(plas))]
            for l in range(len(plas.plas[idx])-1,0,-1):
                zeros[idx] = l
                pl = plas.retreats(plas, zeros)
                pl.utilize(forShow=True)
                self.scene.draw(imageTitle=os.path.join(dir,name+str(l)+".png"))
                self.showTitles.append(name+str(l))
        plas.utilize(forShow=True)
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

        self.currentPlas.singleExtend(0)
        self.addFrame(self.currentPlas,"first",thisDir,0,3)              #--------------------------go first

        for s in range(1,self.segments):
            #assert s<3
            R = math.prod([len(self.currentPlas.plas[t].nids) for t in range(s)])

            hint = [0 for j in range(s)]
            for r in range(R-1):
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlas.plas[i].nids):
                    hint[i],i=0,i+1
                    hint[i]=hint[i]+1

                nowPlas = plas.retreats(self.currentPlas,hint)
                nowPlas.singleExtend(s)
                self.addPlas(nowPlas)
                if s==1:
                    self.addFrame(nowPlas,"second%d"%(r),thisDir,1,3)    #--------------------------squeeze the first and get second
            
            self.currentPlas.plas[s] = self.plases[0].plas[s]
        
            fixedOids = [on[0] for on in self.currentPlas.plas[s].nids]
            for t in range(0,s):
                onId = 0
                while onId < len(self.currentPlas.plas[t].nids):
                    on = self.currentPlas.plas[t].nids[onId]
                    oid,nid= on[0],on[1]
                    if oid in fixedOids:
                        onJd = onId+1
                        while onJd < len(self.currentPlas.plas[t].nids):
                            jN = self.currentPlas.plas[t].nids[onJd]
                            if (jN[1] in self.pm[nid].bunches):
                                self.currentPlas.plas[t].nids.pop(onJd)
                            else:
                                onJd += 1
                        self.currentPlas.plas[t].nids.pop(onId)
                    else:
                        onId += 1
            
            self.addFrame(self.currentPlas,"over%d"%(s),thisDir,-1,3)             #--------------------------squeeze the existing and get next, (directly get the best one, ok)

        if self.iRate < 1.0:
            self.spatiallyRecorrect(self.currentPlas)
            self.addFrame(self.currentPlas,"spatial",thisDir,-1,2)                #--------------------------spatially recorrect,
        self.showTitles = self.showTitles[:1]*2+self.showTitles #print(len(self.showTitles))
        #FIXME:the first several frames are black
        ImageSequenceClip([os.path.join(thisDir,t+".png") for t in self.showTitles], fps=2).write_videofile(os.path.join(thisDir,"recognize.mp4"),logger=None)#
       