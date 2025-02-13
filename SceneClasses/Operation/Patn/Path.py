class path:
    def __init__(self,pt,pm):
        self.path = pt
        self.pm = pm
    
    def __str__(self):
        return " -> ".join([self.pm.nods[i].label.n + " " + str(i) for i in self.path])

    def __len__(self):
        return len(self.path)

    @classmethod
    def append(cls,p,nid):
        return cls([i for i in p.path]+[nid],p.pm)
    
    @classmethod
    def clone(cls,p):
        return cls([i for i in p.path], p.pm)
    
    def __getitem__(self, idx):
        return self.path[idx]
    
    @classmethod
    def subset(cls,p):
        return cls([i for i in p.path][:-1], p.pm)
    
    def recognize(self,scene):
        from numpy.linalg import norm
        solutions,conflicts = [],0.0
        for i,nid in enumerate(self.path):
            for o in scene.OBJES:
                if o.label == self.pm.nods[nid].label: #if self.pm.merging[o.class_name] == self.pm.nods[nid].type:
                    #print(o)
                    #print(self)
                    from ..Syth import sltn
                    SLTN = sltn(self.pm,scene)
                    o.nid = nid
                    for k in range(i):
                        #print(o.nid,self.path[k])
                        ko = self.pm.rel_object(o,self.path[k])
                        #ko.v = False
                        SLTN = SLTN + (ko,scene.conflict(ko))
                        conflicts += scene.conflict(ko)
                        if conflicts > 1.0: break
                    if conflicts > 1.0:                        
                        o.nid = -1
                        continue

                    SLTN = SLTN + (o,0.0)
                    #print("sltn",SLTN)
                    gaps = [SLTN]
                        
                    for j in range(i+1,len(self.path)):
                        js_p = None
                        jo = self.pm.rel_object(o,self.path[j])
                        for p in scene.OBJES:
                            if p.nid == -1 and p.label == self.pm.nods[self.path[j]].label:#self.pm.merging[p.class_name] == self.pm.nods[self.path[j]].type:
                                #print(p," found")
                                #print(jo.flat() - p.flat(), norm(jo.flat() - p.flat()))
                                if (js_p is None and norm(jo.flat() - p.flat()) < 2.0) or(js_p is not None and norm(jo.flat() - p.flat()) < norm(jo.flat() - js_p.flat())):
                                    js_p = p
                        if js_p is None:
                            #jo.v = False
                            #print(jo,"not chosen", o.idx)
                            
                            SLTN = SLTN + (jo,scene.conflict(jo))
                            gaps.append(SLTN)
                            conflicts = SLTN.conflict#+= scene.conflict(jo)
                            if conflicts > 1.0:break
                        else:
                            #print(js_p," chosen", o.idx)
                            js_p.nid = self.path[j]
                            #gaps = []for g in gaps: SLTN = SLTN + g
                            SLTN = SLTN + (js_p,0.0)
                            gaps = [SLTN]
                    for _ in scene.OBJES: _.nid= -1
                    for g in gaps: solutions.append(SLTN)

                    #print("sltn",SLTN,"\n")
        return solutions

    def matching(self,text,I=0,match=None):
        if len(self.path) == I: return [match]
        #print("\n","I=",I, "type=",self.pm.nods[self.path[I]].label.n)
        from ..Patn import mtch
        if match is None: match = mtch(text, self.pm)
        nid = self.path[I]
        ret = self.matching(text,I+1,match)
        if not (self.pm.nods[nid].label.n in text.objs): return ret
        for o in [_ for _ in text.objs[self.pm.nods[nid].label.n] if _.nid == -1]:
            #print(o.flags, nid, text)
            if match.check_add(o.text, o.idx, nid) == False: continue
            #print("match")
            o.nid = nid
            ret = ret + self.matching(text, I+1, match + (self.pm.nods[nid].label.n, o.idx, nid))
            o.nid = -1
        return ret

class paths:
    def __init__(self,pm,node=None,lst=None):
        import random
        self.paths,self.pm = [],pm
        if node is not None:
            self.recursive_construct(node,path([], pm),pm,lst=lst)
        else:
            for nod in pm.nods[0].edges:
                self.recursive_construct(nod.endNode,path([], pm),pm,lst=lst)
        #random.shuffle(self.paths)

    def __str__(self):
        return "paths:\n" + "\n".join([str(p) for p in self.paths])

    def recursive_construct(self,node,p,pm,lst=None):
        A = [ed for ed in node.edges if ed.endNode.label.n in lst] if lst is not None else node.edges
        if len(A) == 0:
            self.paths.append(path.append(p,node.idx))
        for ed in A:
            self.recursive_construct(ed.endNode,path.append(p,node.idx), pm, lst)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.paths[idx]
    
    def __iter__(self):
        return iter(self.paths)

    def __call__(self):
        pass

    def recognize(self,scene):
        solution,solutions = [],[]
        #cnts = {}
        for p in self.paths:#[90:91]:
            #print(self.paths.index(p))
            #if not p[-1] == 146: continue
            #print(p)
            ret = p.recognize(scene)
            for r in ret:
                f = False
                for s in reversed(solution):
                    if r == s:
                        f = True
                        break
                if not f: solution.append(r)
        for i,s in enumerate(solution):
            if s.cover():
                solutions.append([s])
                continue
            for j in range(i+1,len(solution)):
                if s.cover(solution[j]): solutions.append([s,solution[j]])
        
        
        print(len(solutions))
        return solutions

    def matching(self,text):
        #print(text)
        match,matchs = [],[]
        #cnts = {}
        for p in self.paths:#[90:91]:
            #print(self.paths.index(p))
            if len(p) <= 3 or not p[-3] == 164: continue
            #print(p)
            ret = p.matching(text)
            for r in ret:
                # f = False
                # for s in reversed(match):
                #     if r == s:
                #         f = True
                #         break
                # if not f:
                match.append(r)
        for i,s in enumerate(match):
            if s.cover():
                matchs.append([s])
                continue
            for j in range(i+1,i+1):#len(match)):#
                if s.cover(match[j]): matchs.append([s,match[j]])
        print(len(matchs)) #for m in matchs: print(m[0])
        return matchs

class pathses:
    def __init__(self,pm,scene):
        self.scene, self.pm = scene, pm
        self.tolerate = [3,4]
        A = [o.label("mrg") for o in scene.OBJES if o.label("mrg") in pm.rootNames]#[pm.merging[o.class_name()] for o in scene.OBJES if pm.merging[o.class_name()] in [ed.endNode.type for ed in pm.nods[0].edges]]
        AA= list(set(A))
        #orphans = []
        if len(AA) > 2: #if there are more than two core semantic labels in the scene
            orphans = A
            A = []
            for t in ["Dining Table","Coffee Table","King-size Bed","Desk","Dressing Table"]:
                if t in orphans:
                    A.append(t)
                    #orphans.remove(t)
                    if len(A) == 2:
                        break
        elif len(AA) == 2:
            a = A.count(AA[0])
            while a > 1:
                #orphans.append(AA[0])
                A.remove(AA[0])
                a -= 1
            a = A.count(AA[1])
            while a > 1:
                #orphans.append(AA[1])
                A.remove(AA[1])
                a -= 1
        elif len(AA) == 1:
            a = A.count(AA[0])
            while a > 1:
                #orphans.append(AA[0])
                A.remove(AA[0])
                a -= 1
        else:
            raise Exception("No core object in the scene " + str(scene.scene_uid) + " " + str([o.class_name for o in scene.OBJES]))
        #print([self.pm.merging[o.class_name] for o in self.scene.OBJES])
        self.B = [o.label("mrg") for o in scene.OBJES]#[pm.merging[o.class_name] for o in scene.OBJES]
        #print(self.B)
        self.pathses = [paths(self.pm, ed.endNode, self.B) for ed in pm.nods[0].edges if ed.endNode.label.n in A]
        #print(self)
        
    def __str__(self):
        return "\n".join([str(p) for p in self.pathses])

    def combine(self, pathA, pathB, scene):
        from copy import deepcopy
        res = []
        pA = path.clone(pathA)
        while len(pA) > 0:
            pa = path.clone(pA)
            C = [o.label("mrg") for o in scene.OBJES]#[self.pm.merging[o.class_name] for o in scene.OBJES]
            while len(pa) > 0 and self.pm.nods[pa.path[-1]].label.n in C:
                C.remove(self.pm.nods[pa.path[-1]].label.n)
                pa = path.subset(pa)
            if len(pa) == 0:
                break
            pA = path.subset(pA)
        c_a = deepcopy(C)
        while len(pA) > 0:
            C_A = deepcopy(c_a)
            i=0
            for c in pathB.path:
                if self.pm.nods[c].label.n in C_A:
                    C_A.remove(self.pm.nods[c].label.n)
                    i+=1
                else:
                    break
            if len(C_A) <= self.tolerate[1]:
                res.append([deepcopy(pA.path),deepcopy(pathB.path[:i]),deepcopy(C_A)])
            c_a.append(self.pm.nods[pA.path[-1]].label.n)
            pA = path.subset(pA)
        return res

    def __call__(self):
        #print(self)
        from copy import deepcopy
        res = []
        for pA in self.pathses[0]:
            if len(self.pathses)==2:
                for pB in self.pathses[1]:
                    a = self.combine(pA,pB,self.scene)
                    for A in a:
                        if len(A[0]) > 1 and len(A[1]) > 1:
                            res.append(A)
            else:
                C = [o.label("mrg") for o in self.scene.OBJES]#[self.pm.merging[o.class_name()] for o in self.scene.OBJES]
                i = 0
                for c in pA.path:
                    if self.pm.nods[c].label.n in C:
                        C.remove(self.pm.nods[c].label.n)
                        i+=1
                    else:
                        break
                if len(C) <= self.tolerate[0] and i > 1:
                    res.append([deepcopy(pA.path[:i]),deepcopy(C)])
        return res