from .Scne import *
from .Link import *
from .Grup import *
from .Bnch import singleMatch
from itertools import chain

class pla():
    def __init__(self,nids,fits=[]): #nids should not be empty, even when it is been sent in
        self.fits,self.nids=fits,nids #[(oid,nid),(oid,nid)......]
    
    def __len__(self):
        return len(self.nids)
    
    @classmethod
    def extend(cls,a,oid,nid,v):
        return pla(a.nids+[(oid,nid)],a.fits+[v])
    
    @classmethod
    def retreat(cls,a):
        return pla(a.nids[:-1],a.fits[:-1])
    
    def occupied(self):
        return [a[0] for a in self.nids]

class plas(): #it's a recognize plan
    def __init__(self,scene=None,pm=None,base=None):
        if base is None:
            self.scene,self.pm=scene,pm
            self.plas = [pla([(o.idx,pm.rootNames.index(pm.merging[o.class_name()])+1)]) for o in scene.OBJES if (pm.merging[o.class_name()] in pm.rootNames)]
        else:
            self.scene,self.pm=base.scene,base.pm
            self.plas = deepcopy(base.plas)
        self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
        self.fit = sum([sum(p.fits) for p in self.plas])

    def __str__(self):
        return "\n".join([ "%d\t"%(pp) + "→".join(["(%s,%d,%d)"%(self.scene.OBJES[p[0]].class_name(),p[0],p[1]) for p in self.plas[pp].nids]) for pp in range(len(self.plas))])

    def printIds(self):
        return
        print("occupied" +"+".join(["(%s,%d)"%(self.scene.OBJES[id].class_name(),id) for id in self.occupied]) )

    def __len__(self):
        return len(self.plas)

    def plaExtend(self,p,selections):
        self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
        cs=0
        for ed in self.pm.nods[p.nids[-1][1]].edges:
            m = ed.startNode
            while not(ed.endNode.idx in m.bunches):
                m = m.source.startNode

            a = [on[0] for on in p.nids if on[1] == m.idx]
            a = self.scene.OBJES[a[0]]
            #self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
            self.printIds()
            losses = [(oo,m.bunches[ed.endNode.idx].loss(a.rela(oo,self.pm.scaled))) for oo in [o for o in self.scene.OBJES if (self.pm.merging[o.class_name()]==ed.endNode.type and (o.idx not in self.occupied and o.idx not in [on[0] for on in p.nids] ))]] #print(str(lev)+" loop: " + ed.endNode.type + " nid=" + str(ed.endNode.idx) + " idx=" + str(o.idx) + " mid=" + str(m.idx))
            if m.idx==1 and ed.endNode.idx == 254 and False:#
                print("m.idx=%d,m.oid=%d,new.idx=%d"%(m.idx,a.idx,ed.endNode.idx))
                print("exp")
                print(m.bunches[ed.endNode.idx].exp)
                for oolosses in losses:
                    oo,loss = oolosses[0],oolosses[1]#[0] 
                    print("id=%d"%(oo.idx))
                    print("absolute")
                    print(oo.flat())
                    
                    print("relative")
                    print(a.rela(oo,self.pm.scaled).flat())

                    print("(%s,%d):%.3f"%(oolosses[0].class_name(),oolosses[0].idx,oolosses[1]))
                    pass
                    
                
                print("\t".join([ "(%s,%d):%.3f"%(oolosses[0].class_name(),oolosses[0].idx,oolosses[1]) for oolosses in losses]))
            #self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
            self.printIds()
            for oolosses in sorted(losses,key=lambda x:x[1]):
                oo,loss = oolosses[0],oolosses[1]#[0] 
                #print(oo.idx)
                v = singleMatch(loss,ed.confidence,ed.confidenceIn,self.pm.nods[p.nids[-1][1]].edges.index(ed),cs)
                selections.append(pla.extend(p,oo.idx,ed.endNode.idx,v)) 
                break
            cs += ed.confidence

    def singleExtend(self,pid):
        #self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
        #print(self.occupied)
        #print(self)
        self.printIds()
        selections,id = [copy(self.plas[pid])],0
        while id<len(selections):
            self.plaExtend(selections[id],selections)
            id+=1
        self.plas[pid] = sorted(selections,key=lambda x:-sum(x.fits))[0]

        self.occupied = copy(list(chain(*[p.occupied() for p in self.plas])))
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

    def utilize(self):
        self.scene.grp=True
        j = -1
        for p in self.plas:
            if len(p)<=1:
                continue
            j += 1
            for id in range(1,len(p.nids)):
                oid,nid = p.nids[id][0], p.nids[id][1]
                self.scene.OBJES[oid].nid = nid
                self.scene.OBJES[oid].gid = j
                self.scene.LINKS.append(objLink(oid,p.nids[id-1][0],len(self.scene.LINKS),self.scene,"lightblue"))
                
                m = self.pm.nods[nid].source.startNode
                while not(nid in m.bunches):
                    m = m.source.startNode
                for md in range(0,id):
                    if p.nids[md][1] == m.idx:
                        self.scene.LINKS.append(objLink(oid,p.nids[md][0],len(self.scene.LINKS),self.scene,"pink"))
                        break

            self.scene.GRUPS.append(grup([on[0] for on in p.nids],{"sz":self.scene.roomMask.shape[-1],"rt":16},j+1,scne=self.scene))

    def optimize(self,iRate):
        #handly search the father and son relation among objects,

        self.scene.grp=True
        j = -1
        for p in self.plas:
            if len(p)<=1:
                continue
            j += 1
            for id in range(1,len(p.nids)):
                oid,nid = p.nids[id][0], p.nids[id][1]
                assert self.scene.OBJES[oid].nid == nid
                assert self.scene.OBJES[oid].gid == j
                son = self.scene.OBJES[oid]
                m = self.pm.nods[nid].source.startNode
                while not(nid in m.bunches):
                    m = m.source.startNode
                for md in range(0,id):
                    if p.nids[md][1] == m.idx:
                        fid = p.nids[md][0]
                        assert self.scene.OBJES[fid].nid == m.idx
                        fat = self.scene.OBJES[fid]
                        fat_son = fat.rela(son,self.pm.scaled)
                        fat_son = m.bunches[nid].optimize(fat_son)
                        new_son = fat.rely(fat_son,self.pm.scaled)
                        son.translation,son.size,son.orientation = new_son.translation,new_son.size,new_son.orientation

        pass


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

    def addPlas(self,Plas): #bounded insert sort
        for _ in range(20):
            if _ >= len(self.plases) or Plas.fit > self.plases[_].fit:
                self.plases.insert(_,Plas)
                if len(self.plases) > 20:
                    self.plases.pop(-1)

    def spatiallyRecorrect(self,plas):
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
            o = self.scene.OBJES[oid]
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

    def recognize(self,use=True,opt=False,draw=True,show=False):
        import math
        if self.verb > 1:
            print("start recognize")
            print(self.currentPlas)
        if len(self.currentPlas) == 0:
            return 0.0,0
        self.currentPlas.singleExtend(0)
        if self.verb > 1:
            print("recognize 0")
            print(self.currentPlas)

        R = 1
        for s in range(1,self.segments):
            #assert s<3
            if self.verb > 1:
                print("recognize %d"%(s))
            R = math.prod([len(self.currentPlas.plas[t].nids) for t in range(s)])
            if self.verb>2:
                print("R=%d"%(R))

            hint = [0 for j in range(s)]
            for r in range(R-1):
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlas.plas[i].nids):
                    # if self.verb>2:
                    #     print("in while before")
                    #     print(hint)
                    #     print("i=%d"%(i))
                    hint[i],i=0,i+1
                    hint[i]=hint[i]+1
                    # if self.verb>2:
                    #     print("in while after")
                    #     print(hint)
                    #     print("i=%d"%(i))

                nowPlas = plas.retreats(self.currentPlas,hint)
                if self.verb > 2:
                    print(hint)
                    print("before")
                    print(nowPlas)
                    nowPlas.printIds()
                nowPlas.singleExtend(s)
                if self.verb > 2:
                    print("after")
                    print(nowPlas)
                self.addPlas(nowPlas)

                #assert not (len(hint)==2 and hint[0] == 2 and hint[1] == 0)

            
            self.currentPlas.plas[s] = self.plases[0].plas[s]
            if self.verb > 1:
                print("extend %d over"%(s))
                print(self.currentPlas)
        
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
                            if (jN[1] in self.pm.nods[nid].bunches):
                                self.currentPlas.plas[t].nids.pop(onJd)
                            else:
                                onJd += 1
                        self.currentPlas.plas[t].nids.pop(onId)
                    else:
                        onId += 1
            
            if self.verb > 1:
                print("eliminate %d over"%(s))
                print(self.currentPlas)
        
        if self.verb > 0:
            print("recognize over")
            print(self.currentPlas)
    
        if self.iRate < 1.0:
            self.spatiallyRecorrect(self.currentPlas)

        if self.verb > 0:
            print("recorrect over")
            print(self.currentPlas)
        
        if use or opt:
            self.currentPlas.utilize()
        if draw:
            self.scene.draw(drawUngroups=True,classText=True,d=True)
        if opt:
            self.currentPlas.optimize()
            if draw:
                self.scene.draw(drawUngroups=True,classText=True,d=True)
        if show:
            pass
        return self.currentPlas.fit, sum([len(p) for p in self.currentPlas.plas])
    
    def optimize(self,draw=True,show=False):
        self.recognize(use=True,draw=True,show=False)
        #self.currentPlas.singleExtend(0)