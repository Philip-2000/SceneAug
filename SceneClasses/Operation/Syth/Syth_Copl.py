class sltn():
    def __init__(self, pm, scene):
        self.scene = scene
        self.pm = pm
        self.OBJES = []
        self.nid_tags = []
        self.conflict = 0.0

    #方案的话需要被采样才出来的。
    def __gt__(self,on):
        if on[0] not in [ o.idx for o in self.OBJES ]:
            return False
        p = [ o for o in self.OBJES if o.idx == on[0] ][0]
        
        n_ances = self.pm[on[1]].ancs
        for o in self.OBJES:
            if o.nid in n_ances and o.v:
                if o.nid in self.pm[p.nid].ancs:
                    return True
        return False
    
    def __str__(self):
        return '\n'.join([str(o) for o in self.OBJES])
    
    @property
    def ids(self):
        return [o.idx for o in self.OBJES if o.idx != -1]

    def __add__(self,ob_c):
        S = sltn(self.pm,self.scene)
        from ...Basic import obje
        S.OBJES = [obje.copy(o) for o in self.OBJES] + [obje.copy(ob_c[0])]
        S.conflict = self.conflict + ob_c[1]
        return S

    @property
    def hold(self):
        return len([o for o in self.OBJES if o.v])
    
    def conflicts(self,SLTN):
        if not hasattr(self,"occupied"):
            from shapely import unary_union
            self.occupied = unary_union([o.shape() for o in self.OBJES])
        inter = 0.0
        for o in SLTN.OBJES:
            s = o.shape()
            if not o.v:
                t = self.occupied.intersection(s).area
                #print(t)
                inter += t#self.occupied.intersection(s).area
            t = (s.area - self.scene.WALLS.shape().intersection(s).area)
            #print(t)
            inter += t#self.occupied.intersection(s).area
            #inter += (s.area - self.scene.WALLS.shape().intersection(s).area)
        return inter
    
    def conflicting(self,SLTN):
        inter = 0.0
        for o in SLTN.OBJES:
            print(o)
            s = o.shape()
            if not o.v:
                t = self.occupied.intersection(s).area
                print(t)
                inter += t#self.occupied.intersection(s).area
            t = (s.area - self.scene.WALLS.shape().intersection(s).area)
            print(t)
            inter += t#self.occupied.intersection(s).area
            #inter += (s.area - self.scene.WALLS.shape().intersection(s).area)
        return inter

    def cover(self,SLTN=None):
        if SLTN is None:
            m = set([o.idx for o in self.scene.OBJES]) - set([o.idx for o in self.OBJES if o.v])
            return len(m) < 2 and self.conflict < 1.0
        assert self.scene.scene_uid == SLTN.scene.scene_uid
        m = set([o.idx for o in self.scene.OBJES]) - (set([o.idx for o in SLTN.OBJES if o.v]) | set([o.idx for o in self.OBJES if o.v]))
        return len(m) < 2 and len(set([o.idx for o in SLTN.OBJES if o.v]) & set([o.idx for o in self.OBJES if o.v]))==0 and self.conflicts(SLTN) < .4 and self.conflict + SLTN.conflict < 1.2

    def utilize(self):
        #print(self,"\n")
        from ...Basic import obje
        for i,o in enumerate(self.OBJES):
            #o.nid = self.nid_tags[i]
            if o.v == False:
                self.scene.OBJES.addObject(obje.copy(o))
                self.scene.OBJES[-1].v = True
            else:
                self.scene.OBJES[o.idx].nid = o.nid

    def __eq__(self,SLTN):
        if len(self.OBJES) != len(SLTN.OBJES): return False
        for i,o in enumerate(self.OBJES):
            p = SLTN.OBJES[i]
            if o.nid != p.nid: return False
            if not(o.idx is None and p.idx is None) and o.idx != p.idx: return False
        return True
"""
class sltns():
    def __init__(self,scene,pm):
        self.pm = pm
        self.scene = scene
        self.searchingQueue = [o.idx for o in self.scene.OBJES]
        self.SLTNS = []
        self.type_2_nids = self.pm.type_2_nids
        #from shapely import union
        #self.occupied = union([o.shape() for o in self.scene.OBJES])

    def exist(self,i,nid):
        for sltn in self.SLTNS:
            if sltn > [i,nid]:
                return True
        return False
            
    def search_exist(self,oj,SLTN):
        from numpy.linalg import norm
        os = [(o,norm(o.flat()-oj.flat())) for o in self.scene.OBJES if self.pm.merging[o.class_name] == self.pm[oj.nid].type and o.idx not in SLTN.ids]
        os = sorted(os,key=lambda x:x[1])
        if len(os) == 0 or os[0][1] > 1.0: return None
        return os[0]

    def up(self,nid,i):
        #(1) search to the ancestors
        from ...Basic import obje
        SLTN = sltn(self.pm,self.scene)
        for m in self.pm[nid].ancs:
            #enumerate through all its ancestors
            oj = self.pm.rel_object(self.scene[i],m)
            oj.v = False
            o = self.search_exist(oj,SLTN)
            if o is not None and o.idx in self.searchingQueue:
                self.searchingQueue.remove(o.idx)
                self.searchingQueue.insert(o.idx,0)
                self.searchingQueue.append(i)
                return False

            SLTN = SLTN + (oj,c)
            c = self.conflict(oj)
            if SLTN.conflict > 1.0:
                return None
        ob = obje.fromFlat(self.scene[i].flat())
        ob.idx, ob.nid = i, nid
        SLTN = SLTN + (ob,0.0)
        return SLTN

    def down(self,nid,SLTN):
        from ...Basic import obje
        for son_ed in self.pm[nid].edges:
            son_n = son_ed.endNode.idx
            oj = self.pm.rel_object(SLTN.OBJES[-1],son_n)
            oj.v = False
            o = self.search_exist(oj,SLTN)
            if o is not None:
                ob = obje.fromFlat(o.flat())
                ob.idx, ob.nid = o.idx, son_n
                self.down(son_n,SLTN + (ob, 0.0))
            else:
                c = self.conflict(oj)
                if c + SLTN.conflict > 1.0:
                    continue            
                self.down(son_n,SLTN + (oj,c))
        
        self.SLTNS.append(SLTN)

    def search_i(self,i):
        nids = self.type_2_nids[self.scene.OBJES[i].label("mrg")]
        for nid in nids:
            if self.exist(i,nid): continue
            
            SLTN = self.up(nid,i)
            if SLTN is False: return  #ancestors's corresponding object occurs in the scene, let's go to that one and search, instead of searching on this one
            if SLTN is None: continue #conflict when we search on its ancestors, so we don't think object i should take on nid
            self.down(nid,SLTN)

    def __call__(self):
        while len(self.searchingQueue) > 0:
            idx = self.searchingQueue.pop()
            self.search_i(idx)

        pass
"""

from .Syth import syth
class copl(syth):
    def __init__(self,pm,scene,nm="test",v=0):
        super(copl,self).__init__(pm,scene,self.__class__.__name__,nm,v)
        self.ban()
        #raise NotImplementedError

    def ban(self):
        info = {
            "5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088": [0,3,5,6]
        }
        LST = [o for o in self.scene.OBJES if o.idx not in info[self.scene.scene_uid]]
        for i,o in enumerate(LST): o.idx = i
        self.scene.OBJES.OBJES = LST
        #print(self.scene)
        import os
        self.scene.imgDir = "./pattern/syth/"+self.pm.version+"/copl/"+self.scene.scene_uid
        os.makedirs(self.scene.imgDir,exist_ok=True)
        self.scene.scene_uid = "0"
        self.scene.draw(),self.scene.save()

    def uncond(self, use=True, draw=True):#可以类似于rearrange
        from ..Patn import paths
        self.paths = paths(self.pm)
        solutions = self.paths.recognize(self.scene)
        ORIDS = len([o.idx for o in self.scene.OBJES])
        #print("ORIDS",ORIDS)
        cnt = 1
        for solution in solutions[-10:]:
            for o in self.scene.OBJES: o.nid = -1
            self.scene.OBJES.OBJES = self.scene.OBJES.OBJES[:ORIDS]
            #print(self.scene.OBJES,"\n")
            #print(len(solution)) #,solution[0].nid_tags,solution[1].nid_tags,solution[0].nid_tags
            #print(solution[0],"\n",solution[1],"\n") if len(solution) > 1 else print(solution[0],"\n")
            #for o in solution[0].OBJES: print(o.nid)
            for sl in solution: sl.utilize()
            #solution[0].conflicting(solution[1])
            print(sum([sl.conflict for sl in solution]))
            if draw:
                self.scene.scene_uid = str(cnt)
                self.scene.draw(),self.scene.save()
            cnt += 1
            #break
        return self.scene

    def textcond(self):
        #语言指导路径挑选
        return self.scene

    def roomcond(self):
        #场景优化微调
        return self.scene

    def txrmcond(self):
        #语义指导路径挑选+场景优化微调
        return self.scene
