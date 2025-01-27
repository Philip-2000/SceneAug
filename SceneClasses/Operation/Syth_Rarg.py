class path:
    def __init__(self,pt,pm):
        self.path = pt
        self.pm = pm
    
    def __str__(self):
        return " -> ".join([self.pm.nods[i].type + " " + str(i) for i in self.path])

    def __len__(self):
        return len(self.path)

    @classmethod
    def append(cls,p,nid):
        return cls([i for i in p.path]+[nid],p.pm)
    
    @classmethod
    def clone(cls,p):
        return cls([i for i in p.path], p.pm)
    
    @classmethod
    def subset(cls,p):
        return cls([i for i in p.path][:-1], p.pm)

class paths:
    def __init__(self,node,pm,lst):
        import random
        self.paths,self.pm = [],pm
        self.recursive_construct(node,path([], pm),pm,lst)
        random.shuffle(self.paths)

    def __str__(self):
        return "paths:\n" + "\n".join([str(p) for p in self.paths])

    def recursive_construct(self,node,p,pm,lst):
        A = [ed for ed in node.edges if ed.endNode.type in lst]
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

class pathses:
    def __init__(self,pm,scene):
        self.scene, self.pm = scene, pm
        self.tolerate = [3,4]
        A = [pm.merging[o.class_name()] for o in scene.OBJES if pm.merging[o.class_name()] in [ed.endNode.type for ed in pm.nods[0].edges]]
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
            raise Exception("No core object in the scene " + str(scene.scene_uid) + " " + str([o.class_name() for o in scene.OBJES]))
        #print([self.pm.merging[o.class_name()] for o in self.scene.OBJES])
        self.B = [pm.merging[o.class_name()] for o in scene.OBJES]
        #print(self.B)
        self.pathses = [paths(ed.endNode, self.pm, self.B) for ed in pm.nods[0].edges if ed.endNode.type in A]
        #print(self)
        
    def __str__(self):
        return "\n".join([str(p) for p in self.pathses])

    def combine(self, pathA, pathB, scene):
        from copy import deepcopy
        res = []
        pA = path.clone(pathA)
        while len(pA) > 0:
            pa = path.clone(pA)
            C = [self.pm.merging[o.class_name()] for o in scene.OBJES]
            while len(pa) > 0 and self.pm.nods[pa.path[-1]].type in C:
                C.remove(self.pm.nods[pa.path[-1]].type)
                pa = path.subset(pa)
            if len(pa) == 0:
                break
            pA = path.subset(pA)
        c_a = deepcopy(C)
        while len(pA) > 0:
            C_A = deepcopy(c_a)
            i=0
            for c in pathB.path:
                if self.pm.nods[c].type in C_A:
                    C_A.remove(self.pm.nods[c].type)
                    i+=1
                else:
                    break
            if len(C_A) <= self.tolerate[1]:
                res.append([deepcopy(pA.path),deepcopy(pathB.path[:i]),deepcopy(C_A)])
            c_a.append(self.pm.nods[pA.path[-1]].type)
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
                C = [self.pm.merging[o.class_name()] for o in self.scene.OBJES]
                i = 0
                for c in pA.path:
                    if self.pm.nods[c].type in C:
                        C.remove(self.pm.nods[c].type)
                        i+=1
                    else:
                        break
                if len(C) <= self.tolerate[0] and i > 1:
                    res.append([deepcopy(pA.path[:i]),deepcopy(C)])
        return res


from .Syth import syth
class rarg(syth):
    def __init__(self,pm,scene,nm="test",v=0):
        super(rarg,self).__init__(pm,scene,self.__class__.__name__,nm,v)

    def __clean_find(self,node,level):
        if level == -1:
            node.chosen_level = 0
            for i in node.edges:
                self.__clean_find(i.endNode,level)
        if node.chosen_level >= 2**level:
            node.chosen_level -= 2**level
        for i in node.edges:
            if i.endNode.chosen_level >= 2**level:
                i.endNode.chosen_level -= 2**level
                self.__clean_find(i.endNode,level)

    def rarg_init(self, res): #self.rarg_init(res[0][:-1]) return centers, orients
        import random, numpy as np
        xs,zs = [w.p[0] for w in self.scene.WALLS],[w.p[2] for w in self.scene.WALLS]
        assert abs(np.max(xs)+np.min(xs))<0.01 and abs(np.max(zs)+np.min(zs))<0.01
        if len(res) == 1:
            p = res[0]
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c = (wmax.center()*(wmax.length+1.0) + wmin.center()*(wmin.length+1.0))/(wmax.length + wmin.length + 2.0)#np.array([np.max(xs), 0.0, (wmax.p[2]+wmax.q[2])/2.0])
            o = (np.array([.0]) if np.random.rand() > 0.5 else np.array([np.pi])) if np.random.rand() > 0.5 else (np.array([np.pi/2]) if np.random.rand() > 0.5 else np.array([-np.pi/2])) 
            c = c - (0.3+np.random.rand()*0.7)*np.array([np.sin(o[0]), 0, np.cos(o[0])])
            c[1] = 0
            return [c], [o]
        elif len(res) == 2:
            p1,p2 = res
            hint = [2,3,1,4,5] #I'm done with it ok? this hint is only for 'losy', I would never coding like this shit if I don't have to finish papers
            walls = sorted(self.scene.WALLS,key=lambda x:(x.p[0]+x.q[0])) if np.max(xs) > np.max(zs) else sorted(self.scene.WALLS,key=lambda x:(x.p[2]+x.q[2]))
            wmax,wmin = walls[-1],walls[0]
            c1,c2 = wmax.center() * ((wmax.length+1.0)/(wmax.length + wmin.length + 2.0)), wmin.center()*((wmin.length+1.0)/(wmax.length + wmin.length + 2.0))
            
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

    def uncond(self, use=True, move=True, draw=False):
        res = pathses(self.pm,self.scene)() #print(len(res), sum([len(r[-1]) for r in res])/len(res), len(res[0])-1)
        from ..Basic.Obje import obje,object_types
        from ..Operation.Plan import plas, pla
        from numpy.linalg import norm as norm
        import random# , numpy as np
        random.shuffle(res)
        res = sorted(res, key=lambda r: len(r[-1]))

        for o in self.scene.OBJES:
            o.nid, o.gid = -1, 0
        centers, orients = self.rarg_init(res[0][:-1]) #[np.array([.0,.0,.0]), np.array([4,.0,.0]), np.array([-4,.0,.0])], [np.array([0]), np.array([0]), np.array([0])]

        if use:
            self.scene.plan = plas(self.scene, self.pm)
            try:
                self.scene.plan.plas = [pla([],[]) for _ in res[0][:-1]]
            except:
                print(len(res), len(res[0]), self.scene.fild)

            for j,r in enumerate(res[0][:-1]):
                N = self.pm.nods[0]
                for i in r:
                    assert i in [ed.endNode.idx for ed in N.edges]
                    M = self.pm.nods[i]
                    K = M.mid
                    b = self.pm[K].bunches[M.idx]

                    k = [ o for o in self.scene.OBJES if o.nid==K ][0] if K > 0 else obje.empty()
                    m = k+obje.fromFlat(b.exp,j=object_types.index(M.type))
                    o = sorted([ o for o in self.scene.OBJES if self.pm.merging[o.class_name()] == M.type and o.nid==-1 ], key=lambda o: norm(o.size - m.size))[0]
                    o.nid, o.gid = M.idx, j+1
                    o.translation = (m.translation if K > 0 else centers.pop(0)) if move else o.translation
                    o.orientation = (m.orientation if K > 0 else orients.pop(0)) if move else o.orientation
                    self.scene.plan.plas[j] = pla(self.scene.plan[j].nids + [(o.idx, o.nid)], self.scene.plan[j].fits + [0])
                    N = M
            
            for o in self.scene.OBJES:
                o.v = (o.nid != -1)
            
            self.scene.plan.update_fit()
            if draw: self.scene.draw(suffix="_rarg")
        return res[0]


    def unconds(self,draw = True):
        raise Exception("Not used")
        import numpy as np
        from ..Semantic.Link import objLink
        from ..Basic.Obje import obje,object_types,objes
        from ..Basic.Scne import scne
        indexs = [f.class_index for f in self.scene.OBJES]
        choose_edges = None
        most_long_edge = []
        most_len = 0
        level = 0
        N = self.pm.nods[0]
        #print([object_types.index(ed.endNode.type) for ed in N.edges])
        #print(indexs)
        begin_len = len(indexs)
        most_level = len([i for i in indexs if i in [4,8,29,7,9]])                              #yuanlin thinks these are the most important semantic types
        choose_edges = [[] for i in range(0,most_level)]                                        #why don't you use class:plan, you might didn't notice that
        chosen_index = [[] for i in range(0,most_level)]
        while(True):
            from ..Basic.Obje import obje,object_types,obje
            N = self.pm.nods[0]
            if most_level > 1 and level >= most_level:                                          # ??? 
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[-1]
                indexs += chosen_index[-2]
                chosen_index[-1] = []
                chosen_index[-2] = []
                choose_edges[-1] = []
                choose_edges[-2] = []
                level -= 2
                continue
            if most_level == 1 and level >= most_level:                                         # ???
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[-1]
                chosen_index[-1] = []
                choose_edges[-1] = []
                level -= 1
                continue
            if level ==  -1:
                break
            while N.edges and indexs:
                if_continue = False
                for ed in N.edges:
                    if ed.endNode.chosen_level >= 2**level: continue
                    number = object_types.index(ed.endNode.type)
                    if number in indexs:
                        chosen_index[level].append(number)
                        indexs.remove(number)
                        choose_edges[level].append(ed)
                        N = ed.endNode
                        if_continue = True
                        break
                    else:
                        ed.endNode.chosen_level += 2**level
                if if_continue: continue
                if N.chosen_level< 2**level: N.chosen_level += 2**level
                break
            if not N.edges:
                if N.chosen_level< 2**level: N.chosen_level += 2**level
                level += 1
                continue
            if not indexs:
                most_long_edge = choose_edges.copy()
                break
            if self.pm.nods[0].chosen_level < 2**level:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[level]
                chosen_index[level] = []
                choose_edges[level] = []
                continue
            else:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                if level == 0:
                    break
                indexs += chosen_index[level-1]
                chosen_index[level-1] = []
                choose_edges[level-1] = []
                self.__clean_find(self.pm.nods[0],level)
                level -= 1
                continue  
        self.__clean_find(self.pm.nods[0],-1)
        choose_edges = most_long_edge
        #print("most_level",most_level)
        #print("edge_len",most_len)
        set_idx = []
        for i in range(0,len(choose_edges)):                                                                    #utilize the found plan
            offset = [[0,0,0],[4,0,0],[-4,0,0],[0,4,0],[0,-4,0]]
            idxes = []
            for ed in choose_edges[i]:
                N,m = ed.endNode,ed.startNode
                while not (N.idx in m.bunches):
                    m = m.source.startNode
                r = m.bunches[N.idx].sample()
                a = [o for o in self.scene.OBJES if o.nid == m.idx and not o.idx in set_idx] if m.idx > 0 else [obje(np.array(offset[i]),np.array([1,1,1]),np.array([0]))]
                o = a[0] + obje.fromFlat(r,j=object_types.index(N.type))
                o.nid = N.idx
                for obj in self.scene.OBJES.OBJES:
                    if obj.class_index == o.class_index:
                        idxes.append(obj.idx)
                        obj.translation = o.translation
                        obj.size = o.size
                        obj.orientation = o.orientation
                        obj.v, obj.modelId, obj.gid, obj.nid,obj.linkIndex,obj.destIndex = o.v, o.modelId, o.gid, o.nid,o.linkIndex,o.destIndex
                        if m.idx > 0:
                            self.scene.LINKS.append(objLink(a[0].idx,obj.idx,len(self.scene.LINKS),self.scene))
                        break
            set_idx += idxes
        if draw:
            self.scene.draw()
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene