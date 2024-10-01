from .Scne import *
from .Link import *
from .Grup import *
from itertools import chain

class pla():
    def __init__(self,nids,fits=[]): #nids should not be empty, even when it is been sent in
        self.fits,self.nids=fits,nids #[(oid,nid),(oid,nid)......]
    
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
        self.occupied = chain(*[p.occupied() for p in self.plas])
        self.fit = sum([sum(p.fits) for p in self.plas])

    def plaExtend(self,p,selections):
        cs=0
        for ed in self.pm.nods[p.nids[-1][1]].edges:
            m = ed.startNode
            while not(ed.endNode.idx in m.bunches):
                m = m.source.startNode

            a = [on[0] for on in p.nids if on[1] == m.idx]
            a = self.scene.OBJES[a[0]]

            losses = [(oo,m.bunches[ed.endNode.idx].loss(a.rela(oo))) for oo in [o for o in self.OBJES if (self.pm.merging[o.class_name()]==ed.endNode.type and (not o.idx in self.occupied))]] #print(str(lev)+" loop: " + ed.endNode.type + " nid=" + str(ed.endNode.idx) + " idx=" + str(o.idx) + " mid=" + str(m.idx))
                
            for oolosses in sorted(losses,key=lambda x:x[1]):
                oo,loss = oolosses[0],oolosses[1]#[0] 
                v = singleMatch(loss,ed.confidence,ed.confidenceIn,self.pm.nods[p.nids[-1][1]].edges.index(ed),cs)
                selections.append(pla.extend(p,oo.idx,ed.endNode.idx,v)) 
                break
            cs += ed.confidence

    def singleExtend(self,pid):
        selections,id = [self.plas[pid]],0
        while id<len(selections):
            self.singleExtend(selections[id],selections)
            id+=1
        self.plas[pid] = sorted(selections,lambda x:-sum(x.fits))[0]

        self.occupied = chain(*[p.occupied() for p in self.plas])
        self.fit = sum([sum(p.fits) for p in self.plas])
        return self.plas[pid]
    
    def recognize(self):
        for i in range(len(self.plas)):
            
            
            pass

            #self.occupied = chain(*[p.occupied() for p in self.plas])
        

        plan = {"nids":[pm.rootNames.index(pm.merging[o.class_name()])+1 if (pm.merging[o.class_name()] in pm.rootNames) else -1 for o in self.OBJES],
                "fit":0}
        for o in self.OBJES:
            o.nid = plan["nids"][o.idx]
        for r in pm.rootNames:
            for o in [o for o in self.OBJES if pm.merging[o.class_name()] == r]:#print(r)
                self.traverse(pm,o,plan)
        #for p in self.plans:
            #print(str(p["fit"])+"\t".join([pm.merging[self.OBJES[i].class_name()]+":"+str(p["nids"][i]) for i in range(len(p["nids"])) if p["nids"][i] != -1]))
            if len(self.plans):
                plan = sorted(self.plans,key=lambda x:-x["fit"])[0]
        pass

    @classmethod
    def retreats(cls,self,hint):
        newPlas = cls(base=self)
        for i in range(len(self.plas)):
            for j in range(hint[i]):
                newPlas.plas[i] = pla.retreat(newPlas.plas[i])
        newPlas.occupied = chain(*[p.occupied() for p in newPlas.plas])
        newPlas.fit = sum([sum(p.fits) for p in newPlas.plas])
        return newPlas

    def utilize(self):
        self.scene.grp=True
        for j in range(len(self.plas)):
            p = self.plas[j]
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

            self.scene.GRUPS.append(grup([on[0] for on in p.nids],{"sz":self.roomMask.shape[-1],"rt":16},j,scne=self.scene))


class plans():
    def __init__(self,scene,pm):
        self.scene = scene
        self.pm = pm
        self.currentPlas = plas(scene,pm)
        self.segments = len(self.currentPlas.plas)
        self.plases = [self.currentPlas]

    def addPlas(self,Plas): #bounded insert sort
        for _ in range(20):
            if Plas.fit > self.plases[_].fit or _ >= len(self.plases):
                self.plases.insert(_,Plas)
                if len(self.plases) > 20:
                    self.plases.pop(-1)

    def recognize(self,use=True,draw=True,show=False):
        self.currentPlas.singleExtend(0)

        for s in range(1,self.segments):
            R = 1
            for p in self.currentPlas.plas[:s]:
                R*=len(p.nids)

            hint = [0 for j in range(s)]
            for r in range(R-1):
                hint[0],i = hint[0]+1,0
                while hint[i]==len(self.currentPlas.plas[i].nids):
                    hint[i],i=0,i+1
                    hint[i]+=1

                nowPlas = plas.retreats(self.currentPlas,hint)
                nowPlas.singleExtend(s)
                self.addPlas(nowPlas)

            self.currentPlas = self.plases[0]

        if use:
            self.currentPlas.utilize()
        if draw:
            self.scene.draw(drawUngroups=True,classText=True,d=True)
        if show:
            pass
        return self.currentPlas.fit
    
    def optimize(self,draw=True,show=False):
        self.recognize(use=True,draw=True,show=False)
        #self.currentPlas.singleExtend(0)