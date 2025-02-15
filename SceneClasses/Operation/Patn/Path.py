class path:
    def __init__(self,pt,pm):
        self.path = pt
        self.pm = pm
    
    def __str__(self):
        return " -> ".join([self.pm.nods[i].label.n[:10] + " " + str(i) for i in self.path])

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
    def subset(cls,p,i=-1):
        return cls([i for i in p.path][:i], p.pm)
    
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

    def assign(self,scene,j,center,orient,move=True):
        from ...Basic import obje
        from ...Semantic import pla
        from numpy.linalg import norm
        scene.PLAN.PLAN.append(pla([],[]))
        N = self.pm.nods[0]
        for i in self.path:
            assert i in [ed.endNode.idx for ed in N.edges]
            M = self.pm.nods[i]
            K = M.mid
            b = self.pm[K].bunches[M.idx]

            k = [ o for o in scene.OBJES if o.nid==K ][0] if K > 0 else obje.empty()
            #j=object_types.index(M.type)
            m = k+obje.fromFlat(b.exp,n=M.label("ful")) #self.pm.merging[o.class_name()]
            o = sorted([ o for o in scene.OBJES if o.label == M.label and o.nid==-1 ], key=lambda o: norm(o.size - m.size))[0]
            o.nid, o.gid = M.idx, j+1
            o.translation = (m.translation if K > 0 else center) if move else o.translation
            o.orientation = (m.orientation if K > 0 else orient) if move else o.orientation
            scene.PLAN.PLAN[j] = pla(scene.PLAN[j].nids + [(o.idx, o.nid)], scene.PLAN[j].fits + [0])
            N = M

class paths:
    def __init__(self,*args):
        self.paths = [path.clone(_) for _ in args]
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]

    def __str__(self):
        return "paths:\n" + "\n".join([str(p) for p in self.paths])

    def rarg_init(self, scene): #self.rarg_init(res[0][:-1]) return centers, orients
        import random, numpy as np
        xs,zs = [w.p[0] for w in scene.WALLS],[w.p[2] for w in scene.WALLS]
        assert abs(np.max(xs)+np.min(xs))<0.01 and abs(np.max(zs)+np.min(zs))<0.01
        if len(self) == 1:
            p = self[0]
            walls = sorted(scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c = (wmax.center*(wmax.length+1.0) + wmin.center*(wmin.length+1.0))/(wmax.length + wmin.length + 2.0)#np.array([np.max(xs), 0.0, (wmax.p[2]+wmax.q[2])/2.0])
            o = (np.array([.0]) if np.random.rand() > 0.5 else np.array([np.pi])) if np.random.rand() > 0.5 else (np.array([np.pi/2]) if np.random.rand() > 0.5 else np.array([-np.pi/2])) 
            c = c - (0.3+np.random.rand()*0.7)*np.array([np.sin(o[0]), 0, np.cos(o[0])])
            c[1] = 0
            return [c], [o]
        elif len(self) == 2:
            p1,p2 = self
            hint = [2,3,1,4,5] #I'm done with it ok? this hint is only for 'losy', I would never coding like this shit if I don't have to finish papers
            walls = sorted(scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c1,c2 = wmax.center * ((wmax.length+1.0)/(wmax.length + wmin.length + 2.0)), wmin.center*((wmin.length+1.0)/(wmax.length + wmin.length + 2.0))
            
            i12 =-1 if np.max(xs) > np.max(zs) else 2
            o1 = np.array([np.pi/2])*random.choice([ i12, i12+3 if i12==-1 else i12-1, i12-3 if i12==2 else i12+1 ])
            i21 = 1 if np.max(xs) > np.max(zs) else 0
            o2 = np.array([np.pi/2])*random.choice([ i21, i21-1, i21+1 ])

            c1,c2 = c1 - (0.3+np.random.rand()*0.7)*np.array([np.sin(o1[0]), 0, np.cos(o1[0])]),c2 - (0.3+np.random.rand()*0.7)*np.array([np.sin(o2[0]), 0, np.cos(o2[0])])
            c1[1],c2[1] = 0,0
            if (wmin.length > wmax.length) ^ (hint.index(p1[0])>hint.index(p2[0])): #wmin's area is larger than wmax's 
                if np.random.rand() > (((min(wmin.length,wmax.length) / (wmin.length+wmax.length) )/0.5)**3)*0.5:
                    return [c2,c1], [o2,o1]
                else:
                    return [c1,c2], [o1,o2]
            else:
                if np.random.rand() > (((min(wmin.length,wmax.length) / (wmin.length+wmax.length) )/0.5)**3)*0.5:
                    return [c1,c2], [o1,o2]
                else:
                    return [c2,c1], [o2,o1]

    def assign(self,scene,centers=None,orients=None):
        if centers is None or orients is None: centers,orients = self.rarg_init(scene)
        from ...Semantic import plan
        scene.PLAN = plan(scene, self.paths[0].pm)
        for j,r in enumerate(self.paths): r.assign(scene,j,centers.pop(0),orients.pop(0))        
        for o in scene.OBJES: o.v = (o.nid != -1)
        scene.PLAN.update_fit()

class pathses:
    def __init__(self,pm,node=None,lst=None):
        import random
        self.paths,self.pm = [],pm
        if node is not None:
            self.recursive_construct(node,path([], pm),pm,lst=lst)
        else:
            self.segments = [0]
            for nod in pm.nods[0].edges:
                self.recursive_construct(nod.endNode,path([], pm),pm,lst=lst)
                self.segments.append(len(self.paths))
            self.root_segments = {nod.endNode.label.n:[self.segments[i],self.segments[i+1]] for i,nod in enumerate(pm.nods[0].edges)}

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

    def __combine(self, pathA, pathB, scene):
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
            if len(C_A) <= self.tolerate[1]:#deepcopy(pA.path),deepcopy(pathB.path[:i])
                #res.append([deepcopy(pA.path),deepcopy(pathB.path[:i]),deepcopy(C_A)])
                res.append([paths(pA,path.subset(pathB,i)),deepcopy(C_A)])
            c_a.append(self.pm.nods[pA.path[-1]].label.n)
            pA = path.subset(pA)
        return res


    def search(self,scene): #for rarg - path(s) = [path, path ...]
        self.tolerate = [3,4]
        A = [o.label("mrg") for o in scene.OBJES if o.label("mrg") in self.pm.rootNames]#[pm.merging[o.class_name()] for o in scene.OBJES if pm.merging[o.class_name()] in [ed.endNode.type for ed in pm.nods[0].edges]]
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
        from copy import deepcopy
        res = []
        for pA in self.paths[self.root_segments[AA[0]][0]:self.root_segments[AA[0]][1]]:
            if len(AA)==2:
                for pB in self.paths[self.root_segments[AA[1]][0]:self.root_segments[AA[1]][1]]:
                    m = self.__combine(pA,pB,scene)
                    for M in m:
                        if len(M[0][0]) > 1 and len(M[0][1]) > 1:
                            res.append(M)
            else:
                C = [o.label("mrg") for o in scene.OBJES]#[self.pm.merging[o.class_name()] for o in self.scene.OBJES]
                i = 0
                for c in pA.path:
                    if self.pm.nods[c].label.n in C:
                        C.remove(self.pm.nods[c].label.n)
                        i+=1
                    else:
                        break
                if len(C) <= self.tolerate[0] and i > 1:
                    res.append([paths(path.subset(pA,i)),deepcopy(C)])
                    #res.append([deepcopy(pA.path[:i]),deepcopy(C)])
        return res        

    def recognize(self,scene): #for copl - sltn(s) = [sltn, sltn ...]
        from ..Syth import sltns
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
                solutions.append(sltns(s))
            else:
                for j in range(i+1,len(solution)):
                    if s.cover(solution[j]): solutions.append(sltns(s,solution[j]))
        
        print(len(solutions))
        return solutions

    def matching(self,text): #for textcond - mtch(s) = [mtch, mtch ...]
        from .Mtch import mtchs
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
                matchs.append(mtchs(s))
                continue
            for j in range(i+1,i+1):#len(match)):#
                if s.cover(match[j]): matchs.append(mtchs(s,match[j]))
        print(len(matchs)) #for m in matchs: print(m[0])
        return matchs


# class pathses:
#     def __init__(self,pm,scene):
#         self.scene, self.pm = scene, pm
#         self.tolerate = [3,4]
#         A = [o.label("mrg") for o in scene.OBJES if o.label("mrg") in pm.rootNames]#[pm.merging[o.class_name()] for o in scene.OBJES if pm.merging[o.class_name()] in [ed.endNode.type for ed in pm.nods[0].edges]]
#         AA= list(set(A))
#         #orphans = []
#         if len(AA) > 2: #if there are more than two core semantic labels in the scene
#             orphans = A
#             A = []
#             for t in ["Dining Table","Coffee Table","King-size Bed","Desk","Dressing Table"]:
#                 if t in orphans:
#                     A.append(t)
#                     #orphans.remove(t)
#                     if len(A) == 2:
#                         break
#         elif len(AA) == 2:
#             a = A.count(AA[0])
#             while a > 1:
#                 #orphans.append(AA[0])
#                 A.remove(AA[0])
#                 a -= 1
#             a = A.count(AA[1])
#             while a > 1:
#                 #orphans.append(AA[1])
#                 A.remove(AA[1])
#                 a -= 1
#         elif len(AA) == 1:
#             a = A.count(AA[0])
#             while a > 1:
#                 #orphans.append(AA[0])
#                 A.remove(AA[0])
#                 a -= 1
#         else:
#             raise Exception("No core object in the scene " + str(scene.scene_uid) + " " + str([o.class_name for o in scene.OBJES]))
#         #print([self.pm.merging[o.class_name] for o in self.scene.OBJES])
#         self.B = [o.label("mrg") for o in scene.OBJES]#[pm.merging[o.class_name] for o in scene.OBJES]
#         #print(self.B)
#         self.pathses = [paths(self.pm, ed.endNode, self.B) for ed in pm.nods[0].edges if ed.endNode.label.n in A]
#         #print(self)
        
#     def __str__(self):
#         return "\n".join([str(p) for p in self.pathses])

#     def combine(self, pathA, pathB, scene):
#         from copy import deepcopy
#         res = []
#         pA = path.clone(pathA)
#         while len(pA) > 0:
#             pa = path.clone(pA)
#             C = [o.label("mrg") for o in scene.OBJES]#[self.pm.merging[o.class_name] for o in scene.OBJES]
#             while len(pa) > 0 and self.pm.nods[pa.path[-1]].label.n in C:
#                 C.remove(self.pm.nods[pa.path[-1]].label.n)
#                 pa = path.subset(pa)
#             if len(pa) == 0:
#                 break
#             pA = path.subset(pA)
#         c_a = deepcopy(C)
#         while len(pA) > 0:
#             C_A = deepcopy(c_a)
#             i=0
#             for c in pathB.path:
#                 if self.pm.nods[c].label.n in C_A:
#                     C_A.remove(self.pm.nods[c].label.n)
#                     i+=1
#                 else:
#                     break
#             if len(C_A) <= self.tolerate[1]:
#                 res.append([deepcopy(pA.path),deepcopy(pathB.path[:i]),deepcopy(C_A)])
#             c_a.append(self.pm.nods[pA.path[-1]].label.n)
#             pA = path.subset(pA)
#         return res

#     def __call__(self):
#         #print(self)
#         from copy import deepcopy
#         res = []
#         for pA in self.pathses[0]:
#             if len(self.pathses)==2:
#                 for pB in self.pathses[1]:
#                     a = self.combine(pA,pB,self.scene)
#                     for A in a:
#                         if len(A[0]) > 1 and len(A[1]) > 1:
#                             res.append(A)
#             else:
#                 C = [o.label("mrg") for o in self.scene.OBJES]#[self.pm.merging[o.class_name()] for o in self.scene.OBJES]
#                 i = 0
#                 for c in pA.path:
#                     if self.pm.nods[c].label.n in C:
#                         C.remove(self.pm.nods[c].label.n)
#                         i+=1
#                     else:
#                         break
#                 if len(C) <= self.tolerate[0] and i > 1:
#                     res.append([deepcopy(pA.path[:i]),deepcopy(C)])
#         return res