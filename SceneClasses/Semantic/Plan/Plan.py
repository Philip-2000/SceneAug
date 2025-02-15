from itertools import chain
from copy import deepcopy

class pla():
    def __init__(self,nids,fits=[0]): #nids should not be empty, even when it is been sent in
        self.fits,self.nids=fits,nids #[(oid,nid),(oid,nid)......]

    def Str(self,s,pm):
        return "(%s,%d,%d)\n"%(s[self.nids[0][0]].class_name[:10],self.nids[0][0],self.nids[0][1]) + \
                "\n".join(
                        [
                            " ".join(["\t"]*(pid+1)) + "(%s,%d,%d->(%s,%d,%d))"%(
                            s[p[0]].class_name[:10],p[0],p[1],
                            s[self(pm[p[1]].mid)].class_name[:10],self(pm[p[1]].mid),pm[p[1]].mid
                            ) for pid, p in enumerate(self.nids[1:])
                        ]
                    )

    def __len__(self):
        return len(self.nids)
    
    def __iter__(self):
        return iter(self.nids)
    
    def __getitem__(self,i):
        return self.nids[i]

    def __call__(self, nid):
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
        from ...Semantic import objLink
        for i, (oid,nid) in enumerate(self.nids):
            scene[oid].nid,scene[oid].gid = nid, g
            if i>0:
                scene.LINKS.append(objLink(oid,self[i-1][0],color="lightblue"))
                scene.LINKS.append(objLink(oid,self(pm[nid].mid),color="pink"))
        from ...Semantic import grup
        scene.GRUPS.append(grup([on[0] for on in self.nids]))#,{"sz":scene.roomMask.shape[-1],"rt":16},g,scne=scene))
        
    def optimize(self,g,scene,pm,ir,s): #print(self.Str(scene,pm))
        from ...Operation import bnch_tree # bt = bnch_tree(self,pm,scene) print(bt) return bt.optimize(ir)
        return bnch_tree(self,pm,scene).optimize(ir,s) 

    def update_fit(self, g, scene, pm):
        from ...Operation import singleMatch
        for i,(oid,nid) in enumerate(self.nids[1:]):
            if nid == self.nids[i][1]: #it came from spatially recorrect
                continue
            m, fid = pm[pm[nid].mid], self(pm[nid].mid) #m, fid = pm.nods[pm.mid(nid)], self(pm.mid(nid))
            assert scene[oid].nid == nid and scene[oid].gid == g and scene[fid].nid == m.idx
            loss = m.bunches[nid].loss(scene[fid] - scene[oid])
            self.fits[i+1] = singleMatch(loss,None,None,None,None)

    def fromPlaJson(js,g,scene,pm):
        a = pla(js["nids"],js["fits"])
        a.utilize(g,scene,pm)
        return a
    
    def toPlaJson(self):
        return {"nids":self.nids,"fits":self.fits}
    
    def renderables(self, scene_render, scene, pm):
        from simple_3dviz import Lines
        from numpy.linalg import norm
        for i in range(1,len(self.nids)):
            dst_o,src_o = scene[self.nids[i][0]], scene[self(pm[self.nids[i][1]].mid)]
            exp = pm[src_o.nid].bunches[dst_o.nid].exp
            act = (src_o - dst_o).flat()
            dif = norm(act-exp)
            src,dst = src_o.translation, dst_o.translation
            if norm(src-dst) > norm(exp[:3]): #the distance is larger than the expected distance 
                colors,width = (max(0.5-0.1*dif,0.0),0,0), max(0.003 - 0.002*dif,0.0001)
            else: #the distance is smaller than the expected distance
                colors,width = (min(0.5+0.1*dif,1.0),0,0), 0.003 + 0.002*dif
            src[1],dst[1] = 2.8,2.8
            scene_render.add(Lines([scene[self.nids[i][0]].translation,scene[self(pm[self.nids[i][1]].mid)].translation],colors=colors,width=width))

class plan(): #it's a recognize plan
    def __init__(self,scene=None,pm=None,base=None):
        if base is None:
            self.scene,self.pm=scene,pm #pm.merging[o.class_name]
            self.PLAN = [pla([(o.idx, [ed.endNode.idx for ed in pm[0].edges if ed.endNode.label == o.label].pop() )]) for o in scene.OBJES if (o.label("mrg") in pm.rootNames)]
        else:
            self.scene,self.pm=base.scene,base.pm
            self.PLAN = deepcopy(base.PLAN)
        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.PLAN])))
        self.fit = sum([sum(p.fits) for p in self.PLAN])
        self.myPM = None

    @classmethod
    def fromPlanJson(cls,js,scene):
        from ...Operation import patternManager as PM
        a = cls(scene=scene,pm=PM(vers=js["vers"]))
        a.PLAN = [ pla.fromPlaJson(p, i+1, scene, a.pm) for i,p in enumerate(js["plan"])]
        a.occupied = list(chain(*[p.occupied() for p in a.PLAN]))
        a.fit = sum([sum(p.fits) for p in a.PLAN])
        return a
        
    def toPlanJson(self,rsj):
        dct = {"vers":self.pm.version}
        dct["plan"] = []
        for p in self.PLAN:
            if len(p)>1: dct["plan"].append(p.toPlaJson())
        rsj["plan"] = dct
        return rsj
    
    def renderables(self, scene_render):
        for p in self.PLAN: p.renderables(scene_render, self.scene, self.pm)

    def __str__(self):
        return "\n".join([ "%d\t"%(pid) + p.Str(self.scene,self.pm) for pid,p in enumerate(self.PLAN)])

    def __iter__(self):
        return iter(self.PLAN)

    def formPM(self):
        import numpy as np
        from ...Operation import bnch
        from ...Operation import patternManager as PM
        from ...Basic import obje
        
        pm = PM(vers="tmp",new=True) #pm.merging[self.scene[p.nids[0][0]].class_name]
        pm.rootNames = [ self.scene[p.nids[0][0]].label("mrg") for p in self.PLAN ]
        for p in self.PLAN:
            newNids, fid = {0:0},0
            for oid,nid in p.nids:
                o,mid = self.scene[oid], self.pm[nid].mid#m.idx
                O = obje(o.translation,np.array([1,1,1]),o.orientation,i=0) if p.nids.index((oid,nid))==0 else self.scene[p(mid)]
                fid = pm.createNode(pm[fid],o.label("mrg"),1,1)
                #pm.nods[newNids[mid]].bunches[fid], newNids[nid] = bnch(None,O.rela(o,self.pm.scaled).flat(),np.abs(O.rela(o,self.pm.scaled).flat()-self.pm.nods[mid].bunches[nid].exp)), fid
                pm[newNids[mid]].bunches[fid], newNids[nid] = bnch(None,(O-o).flat(),np.abs((O-o).flat()-self.pm[mid].bunches[nid].exp)), fid
        return pm

    def diff(self,scene,ref=None,v=0):
        if self.myPM is None:
            self.myPM = self.formPM()
            if v>0:print(self.myPM)
        from ...Operation import rgnz  
        fit,ass,_ = rgnz(scene,self.myPM,v=v).recognize(use=False,draw=False,show=False)#assert fit < ref
        if v>0:print(fit,self.fit)
        return abs(self.fit - fit)

    def printIds(self):
        return
        print("occupied" +"+".join(["(%s,%d)"%(self.scene[id].class_name,id) for id in self.occupied]) )

    def __len__(self):
        return len(self.PLAN)

    def plaExtend(self,p,selections):
        from ...Operation import singleMatch
        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.PLAN])))
        cs=0
        for ed in self.pm[p.nids[-1][1]].edges:
            m = self.pm[self.pm[ed.endNode.idx].mid]
            a = self.scene[p(m.idx)]
            self.printIds() #self.pm.merging[o.class_name]
            losses = [(oo,m.bunches[ed.endNode.idx].loss(a-oo)) for oo in [o for o in self.scene.OBJES if (o.label==ed.endNode.label and (o.idx not in self.occupied and o.idx not in [on[0] for on in p.nids] ))]] #print(str(lev)+" loop: " + ed.endNode.label.n + " nid=" + str(ed.endNode.idx) + " idx=" + str(o.idx) + " mid=" + str(m.idx))
            if m.idx==1 and ed.endNode.idx == 254 and False:#
                print("m.idx=%d,m.oid=%d,new.idx=%d"%(m.idx,a.idx,ed.endNode.idx))
                print("exp")
                print(m.bunches[ed.endNode.idx].exp)
                for oolosses in losses:
                    oo,loss = oolosses[0],oolosses[1]#[0] 
                    print("id=%d"%(oo.idx))
                    print("absolute")
                    print(oo.flat())
                    

                    print("(%s,%d):%.3f"%(oolosses[0].class_name,oolosses[0].idx,oolosses[1]))
                    pass
                    
                
                print("\t".join([ "(%s,%d):%.3f"%(oolosses[0].class_name,oolosses[0].idx,oolosses[1]) for oolosses in losses]))
            self.printIds()
            for oolosses in sorted(losses,key=lambda x:x[1]):
                oo,loss = oolosses[0],oolosses[1]#[0] 
                #print(oo.idx)
                v = singleMatch(loss,ed.confidence,ed.confidenceIn,self.pm[p.nids[-1][1]].edges.index(ed),cs)
                selections.append(pla.extend(p,oo.idx,ed.endNode.idx,v)) 
                break
            cs += ed.confidence

    def singleExtend(self,pid):
        #self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.PLAN])))
        #print(self.occupied)
        #print(self)
        self.printIds()
        selections,id = [deepcopy(self.PLAN[pid])],0
        while id<len(selections):
            self.plaExtend(selections[id],selections)
            id+=1
        self.PLAN[pid] = sorted(selections,key=lambda x:-sum(x.fits))[0]

        self.occupied = deepcopy(list(chain(*[p.occupied() for p in self.PLAN])))
        self.fit = sum([sum(p.fits) for p in self.PLAN])
        return self.PLAN[pid]
    
    @classmethod
    def retreats(cls,self,hint):
        newPlan = cls(base=self)
        for i in range(len(hint)):
            for j in range(hint[i]):
                newPlan.PLAN[i] = pla.retreat(newPlan.PLAN[i])
        newPlan.occupied = chain(*[p.occupied() for p in newPlan.PLAN])
        newPlan.fit = sum([sum(p.fits) for p in newPlan.PLAN])
        return newPlan

    def utilize(self,clear=True,forShow=False):
        if clear:
            self.scene.LINKS.clear()
            self.scene.GRUPS.clear()
            for o in self.scene.OBJES:
                o.gid,o.nid=0,0
        self.scene.grp, self.scene.PLAN = True,self
        [p.utilize(g+1,self.scene,self.pm) for g,p in enumerate([p for p in self.PLAN if len(p)>1 or forShow])]

    def optimize(self, ir, s):
        #_ = self.optinit() if s==0 else None
        [p.optimize(g+1,self.scene,self.pm,ir,s) for g,p in enumerate([_ for _ in self.PLAN if len(_) > 1])]
        from ...Operation import adjs
        return adjs(self.scene.OBJES)#[(son.adjust["T"],son.adjust["S"],son.adjust["R"]) for son in self.scene.OBJES]
    
    def update_fit(self):
        [p.update_fit(g+1,self.scene,self.pm) for g,p in enumerate([_ for _ in self.PLAN if len(_) > 1])]
        self.fit = sum([sum(p.fits) for p in self.PLAN if len(p)>1])
        #if self.scene.scene_uid.startswith("339e4d0f-096a"): print(self)
        N = sum([ len([i for i in range(1,len(p)) if p.nids[i][1] != p.nids[i-1][1]]) for p in self.PLAN if len(p) > 1 ])
        return self.fit, N*15.0 # sum([len(p.fits)-1 for p in self.PLAN if len(p)>1])*15.0
    
    def __getitem__(self, k):
        return self.PLAN[k]

    def __iter__(self):
        return iter(self.PLAN)
    
    def optinit(self):
        #this operation is right after the randomization, so it's operation could be regarded as the part of the randomization
        #during ExOp, this operation is recorded with the randomization result to be the so called, "ground truth"
        #but however, during real utilization where we want to optimize a real scene, we can also use this function to find a better initialization state
        #so the thing is a little bit complex anyway
        import numpy as np
        xs,zs = [w.p[0] for w in self.scene.WALLS],[w.p[2] for w in self.scene.WALLS]
        assert abs(np.max(xs)+np.min(xs))<0.01 and abs(np.max(zs)+np.min(zs))<0.01
        if len([p for p in self.PLAN if len(p)>1]) == 1:
            p = [p for p in self.PLAN if len(p)>1][0]
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c = (wmax.center*(wmax.length+1.0) + wmin.center*(wmin.length+1.0))/(wmax.length + wmin.length + 2.0)#np.array([np.max(xs), 0.0, (wmax.p[2]+wmax.q[2])/2.0])
            M = self.scene[p.nids[0][0]].translation.copy() - c 
            [self.scene[oid].adjust.update(T=-M, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p.nids ]
        elif len([p for p in self.PLAN if len(p)>1]) == 2:
            p1,p2 = [p for p in self.PLAN if len(p)>1]
            hint = [2,3,1,4,5] #I'm done with it ok? this hint is only for 'losy', I would never coding like this shit if I don't have to finish papers
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            if (wmin.length > wmax.length) ^ (hint.index(p1.nids[0][1])>hint.index(p2.nids[0][1])): #wmin's area is larger than wmax's 
                p1,p2 = p2,p1 #then p1 should be at wmax, p2 at wmin 
            # if np.max(xs) > np.max(zs):
            #     c1,c2 = wmax.center * 0.4, wmin.center*0.4 #np.array([np.max(xs)*0.3, 0.0, (wmax.p[2]+wmax.q[2])/2.0]), np.array([np.min(xs)*0.3, 0.0, (wmin.p[2]+wmin.q[2])/2.0])
            # else:
            #     c1,c2 = wmax.center * 0.4, wmin.center*0.4 #np.array([(wmax.p[0]+wmax.q[0])*0.5, 0.0, np.max(zs)*0.3]), np.array([(wmin.p[0]+wmin.q[0])*0.5, 0.0, np.min(zs)*0.3])
            #print(wmax, wmax.center, wmax.center*0.5)
            #print(wmin, wmin.center, wmin.center*0.5)
            #print(wmax.center * ((wmax.length+1.0)/(wmax.length + wmin.length + 2.0)))
            #print(wmin.center * ((wmin.length+1.0)/(wmax.length + wmin.length + 2.0)))
            
            M1,M2 = self.scene[p1.nids[0][0]].translation.copy() - wmax.center * ((wmax.length+1.0)/(wmax.length + wmin.length + 2.0)), self.scene[p2.nids[0][0]].translation.copy() - wmin.center*((wmin.length+1.0)/(wmax.length + wmin.length + 2.0))
            
            [self.scene[oid].adjust.update(T=-M1, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p1.nids]
            [self.scene[oid].adjust.update(T=-M2, S=np.array([0,0,0]), R=np.array([0])) for oid,nid in p2.nids]
        else:
            print(self.scene)
            raise Exception(self.scene.scene_uid +  "plan.optinit: the number of groups is %d not 1 or 2"%(len([p for p in self.PLAN if len(p)>1])))
        from ...Operation import adjs
        return adjs(self.scene.OBJES)
