import random, os, numpy as np

class edge():
    def __init__(self,n,nn,c,cc):
        self.startNode = n
        self.endNode = nn
        self.confidence = c
        self.times = 0
        self.choose_times = 0
        self.confidenceIn = cc
        nn.source = self

class node():
    def __init__(self,label,suffix,idx=-1):
        from ...Basic import labl
        self.label = labl("mrg",label)
        self.suffix = suffix
        self.source = None
        self.idx = idx
        self.h = 1
        self.edges = []
        self.bunches = {}
        self.ancs = []
        self.tree_types = set([self.label.n])

    def __str__(self):
        return self.label.n + " " + str(self.idx)
        
    def __getitem__(self,i):
        if type(i) == int and i>0:
            return [self.edges[a].endNode for a in range(len(self)) if self.edges[a].endNode.idx == i][0]
        elif i == "random":
            return random.choice([self.edges[a].endNode for a in range(len(self))])
        elif i == "confidence+":
            R = random.uniform(0, sum([self.edges[a].confidence for a in range(len(self))]) - 1e-5)
            for a in range(len(self)):
                R -= self.edges[a].confidence
                if R <= 0: return self.edges[a].endNode
        elif i == "confidence":
            R = random.uniform(0, 1)
            for a in range(len(self)):
                R -= self.edges[a].confidence
                if R <= 0: return self.edges[a].endNode
            return None
        elif type(i) == str and i.isdigit():
            return random.choice([self.edges[a].endNode for a in range(len(self)) if self.edges[a].endNode.h >= int(i)])
        else:
            raise NotImplementedError("node.__getitem__, "+str(i)+" unrecognized usage")

    def __call__(self,i):
        return self.bunches[i]
    
    def __len__(self):
        return len(self.edges)
    
    @property
    def mid(self):
        m = self.source.startNode
        while not(self.idx in m.bunches):
            m = m.source.startNode
        return m.idx
    
    def height(self):
        if self.h == 1 and len(self.edges):
            self.h = 1 + max([e.endNode.height() for e in self.edges])
        return self.h

class merging():
    def __init__(self,d):
        self.d=d
    def __getitem__(self,a):
        from ...Basic import object_types
        try:
            a = object_types[int(a)]
            return object_types.index(self.d[a] if (a in self.d) else a)
        except:
            return self.d[a] if (a in self.d) else a
    def reversed(self,a):
        return [a]+[k for k in self.d if self.d[k]==a]  

class patternManager():
    def __init__(self,vers,verb=0,new=False):
        self.version = vers
        self.workDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),"pattern")#"./pattern/"
        self.fieldDir = os.path.join(self.workDir,"fields/")
        self.imgDir = os.path.join(self.workDir,"imgs/")
        self.treesDir = os.path.join(self.workDir,"trees/") #self.cnt = len(os.listdir(self.sceneDir))
        self.rootNames=["Dining Table","Coffee Table","King-size Bed","Desk","Dressing Table"]#,"Single bed","Kids Bed"]#
        self.merging = merging(
            {"Single bed":"King-size Bed", 
             "Kids Bed":"King-size Bed",
             "Loveseat Sofa":"Three-seat / Multi-seat Sofa",
             "Lounge Chair / Cafe Chair / Office Chair":"Dining Chair", 
             "Classic Chinese Chair":"Dining Chair", 
             "Dressing Chair":"Dining Chair", 
             "armchair":"Dining Chair",
             "Corner/Side Table": "Nightstand",
             "Ceiling Lamp":"Pendant Lamp"})
        self.nods = [node("","",0)]#[nod(node("","",0))]
        self.verb = verb
        self.sDs = None#scneDs(self.sceneDir, grp=False,wl=False,keepEmptyWL=True,cen=True,rmm=False)
        self.maxDepth = 10
        self.Q = []
        self.objectViewBd = 6
        self.scaled=True
        if not new:
            import json,yaml
            assert os.path.exists(os.path.join(self.treesDir,self.version+".js"))
            self.loadTre(json.loads(open(os.path.join(self.treesDir,self.version+".js")).read()[8:-1]))#self.loadTre(json.load(open(os.path.join(self.treesDir,self.version+".js"))))
            self.nods[0].height()
            config = yaml.load(open(os.path.join(self.treesDir,self.version+".yaml")), Loader=yaml.FullLoader)
            self.rootNames = config["rootNames"]
            self.scaled = config["scale"] if "scale" in config else True
        else:
            assert not os.path.exists(os.path.join(self.treesDir,self.version+".js"))

    #region: in/outputs----------#
        #region: inputs----------#
    def loadTre(self,dct,id=0):
        from .Bnch import bnch
        if id == 0:
            self.nods += [None for _ in dct[1:]] #nod(None)
        suf = "" if id == 0 else self.nods[id].suffix+"+"
        for b in dct[id]["buncs"]:
            self.nods[id].bunches[int(b)] = bnch(None,np.array(dct[id]["buncs"][b][0]),np.array(dct[id]["buncs"][b][1]))
        for f in dct[id]["edges"]:
            e = f[0]
            self.nods[e] = node(dct[e]["type"],suf+dct[id]["type"],e)
            self.nods[id].edges.append(edge(self.nods[id],self.nods[e],c=f[1],cc=f[2]))
            self.loadTre(dct,e)
            if id>0:
                suf += "--"+dct[e]["type"]
        self.set_ancestors()
        #self.set_tree_types()
        #endregion: inputs-------#

        #region: construction----#
    def createNode(self,fat,s,c=0.0,cc=0.0):
        self.nods.append(node(s,fat.suffix if len(fat.suffix)>0 else s.replace('/','.'),len(self.nods)))#nod(nn))
        fat.edges.append(edge(fat,self.nods[-1],c,cc))
        return len(self.nods)-1

    def freqInit(self,n):#筛选出里面有的物体，组成bunch 然后进行分析，每一次向下个分析都有相应的可信度
        from .Bnch import bnch
        from ...Basic import obje, labl
        for s in self.rootNames:
            for ss in labl("mrg",s).reverse("ful"):#self.merging.reversed(s):
                if not os.path.exists(os.path.join(self.fieldDir,ss+".txt")):
                    open(os.path.join(self.fieldDir,ss+".txt"),"w").write('\n'.join([ str(sc) for sc in range(len(self.sDs)) if ss in [o.class_name for o in self.sDs[sc].OBJES] ]))
                for f in open(os.path.join(self.fieldDir,ss+".txt")).readlines():
                    for o in [_ for _ in self.sDs[int(f)].OBJES if _.label("mrg") == s]:#self.merging[_.class_name()] == s]:#F+=1
                        if (len(self.nods)) in n.bunches: #i=self.merging[o.class_index]
                            n.bunches[len(self.nods)].add(obje(np.array([0,0,0]),o.size,np.array([0]),n=o.label("mrg"),idx=o.idx,scne=o.scne),True)
                        else: #i=self.merging[o.class_index]
                            n.bunches[len(self.nods)] = bnch(obje(np.array([0,0,0]),o.size,np.array([0]),n=o.label("mrg"),idx=o.idx,scne=o.scne))
            n.bunches[len(self.nods)].enable(len(self.nods))
            self.createNode(n,s,len(n.bunches[len(self.nods)])/len(self.sDs),len(n.bunches[len(self.nods)])/len(self.sDs))

        self.Q = [e for e in n.edges]
        while len(self.Q):
            e = self.Q.pop(0)
            self.freq(e.endNode)#,[e.endNode.idx]

    def freq(self,n,ex=True,lev=0): #,path
        from .Bnch import bnches,giveup
        from ...Basic import object_types,noOriType
        path,m = [],n
        while m.idx != 0:
            path,m = (path if (m.label.n in noOriType) else [m.idx] + path), m.source.startNode
        field= [self.sDs[int(f)] for f in open(self.fieldDir+n.suffix+".txt").readlines() if n.idx in self.sDs[int(f)].OBJES.nids()]
        while 1:
            sheet = {id:{o:bnches() for o in object_types} for id in path}
            idset = set([e.endNode.idx for e in n.edges])#下一层树的节点的种类
            cnt = 0
            for scene in field:#for f in lstt: assert n.idx in scene.OBJES.nids() #scene = self.sDs[int(f)]
                blackLists = {id:{o:[] for o in object_types} for id in path}
                if (not ex) or len(scene.OBJES.nids() & idset) == 0:
                    cnt += 1
                    for o in scene.OBJES:
                        if o.nid in sheet: #o.nid == n.idx:
                            res = scene.OBJES.objectView(o.idx,self.objectViewBd,self.scaled) #assert len(res) <= self.objectViewBd
                            for r in res:#print(r.class_name) #self.merging[r.class_name] 
                                blackLists[o.nid][r.label("mrg")].append(sheet[o.nid][r.label("mrg")].accept(r,1,blackLists[o.nid][r.label("mrg")]))
            [[sheet[k][s].refresh() for s in sheet[k]] for k in sheet]
    
            if self.verb > 1:
                print('\n'.join([k+str(len(sheet[n.idx][k]))+' '+str(sheet[n.idx][k].all()) for k in sheet[n.idx] if sheet[n.idx][k].all() > 500]))
            
            if self.verb > 2:
                print('\t'.join([str(len(b)) for b in sheet[n.idx]["Three-seat / Multi-seat Sofa"].bunches[:10]]))
                print('\t'.join(["%.3f"%(b.exp[0]) for b in sheet[n.idx]["Three-seat / Multi-seat Sofa"].bunches[:10]]))
                print('\t'.join(["%.3f"%(b.exp[-1]) for b in sheet[n.idx]["Three-seat / Multi-seat Sofa"].bunches[:10]]))
                return
            
            #B = sorted([sheet[k].mx() for k in sheet],key=lambda x:(0 if x is None else -len(x)))[0]
            Bs = {i:sorted([sheet[i][k].mx() for k in sheet[i]],key=lambda x:(0 if x is None else -len(x)))[0] for i in path}
            B = sorted([(Bs[i],i) for i in path],key=lambda x:(0 if x[0] is None else -len(x[0])))[0]
            
            if giveup(B[0],len(field),cnt):
                break
            #self.merging[B[0].obs[0].class_name]
            self.createNode(n,B[0].obs[0].label("mrg"),len(B[0])/len(field),len(B[0])/cnt)
            B[0].enable(len(self.nods)-1)#nn.idx)
            self.nods[B[1]].bunches[len(self.nods)-1] = B[0]

            if self.verb > 1:
                print(B[0].obs[0].label("mrg") + " " + str(len(B[0])) + "\n")

        if self.verb > 0:
            print('\t'.join(['\t']*lev+[e.endNode.label.n for e in n.edges]+["fuck"]))
        if lev < self.maxDepth:
            for e in n.edges:
                self.freq(e.endNode,ex,lev+1)#,(path if (e.endNode.label.n in noOriType) else path+[e.endNode.idx])
        self.Q = self.Q + [e for e in n.edges] if self.maxDepth == -1 else self.Q

    def construct(self,maxDepth,scaled,sDs):
        self.maxDepth,self.scaled,self.sDs = maxDepth,scaled,sDs
        self.freqInit(self.nods[0])
        self.storeTree()
        self.draw()
        #endregion: construction-#

        #region: presentation----#
    def __str__(self):
        assert self.version == "tmp" #only for temporary patterns, for debugging
        resStr = "rootNames:"+str(self.rootNames)+"\n"
        for e in self.nods[0].edges:
            lev = 0
            while 1:
                n = e.endNode
                m = self.nods[n.idx].source.startNode
                while not(n.idx in m.bunches):
                    m = m.source.startNode
                nodeStr = ".".join(["\t"]*lev) + "%s nid=%d c=%.2f cc=%.2f from (%s mid=%d): %s"%(n.label.n,n.idx,e.confidence,e.confidenceIn,m.label.n,m.idx,str(m.bunches[n.idx]))
                resStr += nodeStr + "\n"
                lev += 1
                assert len(n.edges)<=1
                if len(n.edges)==0:
                    break
                e = n.edges[0]
        return resStr

    def draw(self,all=False,lim=5):
        import json
        from ...Basic import obje,object_types
        if not os.path.exists(self.imgDir+self.version+"/"):
            os.makedirs(self.imgDir+self.version+"/")
        info = {}#open(self.imgDir+self.version+"/info.json")

        for n in self.nods[1:]:
            nn,sr,path = n,n.source.startNode,[n]
            while sr.idx > 0:
                while not(nn.idx in sr.bunches.keys()) and sr.source.startNode.idx>0:    
                    sr = sr.source.startNode
                nn = sr
                sr = nn.source.startNode
                path.append(nn)

            B = sr.bunches[nn.idx]
            A = obje.empty()#obje.fromFlat(B.exp,j=object_types.index(nn.type))
            for i in range(len(path)-1,0,-1): #j=object_types.index(path[i-1].type)
                A = (A + obje.fromFlat(B.exp,n=path[i-1].label("ful")))
                B = path[i].bunches[path[i-1].idx] #object_types.index(nn.type)
            B.draw(A,self.imgDir+self.version,str(n.idx),nn.label("ful"),self.scaled,all,lim,path)
            info[path[0].idx] = path[1].idx if len(path)>1 else 0

        open(self.imgDir+self.version+"/info.js","w").write("var info="+json.dumps(info)+";")

    def storeTree(self):
        #if len(name) > 0:
        import yaml,json,inspect
        from .Bnch import giveup,DEN,SIGMA2
        lst = [{"type": N.label.n,"buncs":{i:[N.bunches[i].exp.tolist(),N.bunches[i].dev.tolist(),len(N.bunches[i])] for i in N.bunches},
                "edges":[(e.endNode.idx,e.confidence,e.confidenceIn) for e in N.edges]} for N in self.nods]
        open(os.path.join(self.treesDir,self.version+".js"),"w").write("var dat="+json.dumps(lst)+";")#open(os.path.join(self.treesDir,self.version+".json"),"w").write(json.dumps(lst))
        yaml.dump({"merging":self.merging.d,"rootNames":self.rootNames,"maxDepth":self.maxDepth,"scaled":self.scaled,"DEN":DEN,"SIGMA2":SIGMA2,"giveup":('\n'.join(inspect.getsource(giveup).split('\n')[1:-1]))},open(os.path.join(self.treesDir,self.version+".yaml"),"w"))
        #endregion: presentation-#

    #endregion: in/outputs-------#

    #region: utilize-------------#
    def __getitem__(self,i):
        return self.nods[i]

    def random_path(self,path,N_min,N_max):
        if N_max == 0: return path
        n = self[path[-1]]
        candidates = [n.edges[a] for a in range(len(n)) if n.edges[a].endNode.h >= N_min-1]
        R = random.random()*(sum([e.confidence for e in candidates])-1e-5) if N_min > 0 else random.random()*sum([e.confidence for e in candidates])*1.3
        for e in candidates:
            R -= e.confidence
            if R <= 0: return self.random_path(path + [e.endNode.idx],N_min-1,N_max-1)
        return path

    def random_paths(self,N_min=2,N_max=32,A=[-1,-1]):
        A = np.random.choice([i for i in range(1,1+len(self[0].edges))],size=len(A),replace=False) if A[0]<0 else A
        if len(A) == 1: return [self.random_path([A[0]],N_min,N_max)]
        res_0 = self.random_path([A[0]],max(2,N_min-self[A[1]].h), N_max-4)
        res_1 = self.random_path([A[1]],max(4,N_min-len(res_0)), N_max-len(res_0))
        return [res_0,res_1]
    
    def exp_object(self,i,s,d=.0,t=None,ori=None,add=True):
        from ...Basic import obje
        if self[i].mid > 0: son = s[s(self[i].mid).idx] + obje.fromFlat(self[self[i].mid].bunches[i].sample(d),n=self[i].label("ful"))
        else:               son = obje.fromFlat(np.concatenate([t, self[0].bunches[i].exp[3:6], ori], axis=0), n=self[i].label("ful"))
        son.nid = i
        if add: s.addObject(son)
        return son

    def leaf_scene(self,leaf,scene=None,center=[0,0,0],ori=[0]):
        from ...Basic import scne
        scene = scne.empty() if scene is None else scene
        ancs = sorted(self[leaf].ancs)
        goids = []
        for i in ancs:
            self.exp_object(i,scene,0.0,center,ori)#return scene
            goids.append(scene[-1].idx)
        from ...Semantic import grup
        scene.GRUPS.append(grup(goids))
        return scene

    def random_scene(self,N_min=2,N_max=32,A=[-1],centers=[np.array([1.0,.0,.0]),np.array([-1.0,.0,.0])],oris=[np.array([-np.pi/2]),np.array([np.pi/2])],d=.0):
        from ...Basic import scne
        from ...Semantic import grup
        s, paths = scne.empty(), self.random_paths(N_min=N_min,N_max=N_max,A=A)
        for j, path in enumerate(paths):
            goids = []
            for i in path:
                self.exp_object(i,s,d,centers[j],oris[j])
                goids.append(s[-1].idx)
            s.GRUPS.append(grup(goids))
        return s
    
    def rel_bunch(self,o,nid,d=.0):
        from ...Basic import obje,object_types
        assert o.nid == self[nid].mid or nid == self[o.nid].mid
        if o.nid == self[nid].mid:
            p = o + obje.fromFlat(self[o.nid].bunches[nid].sample(d),n=self[nid].label("ful"))
        else:
            p = o + (obje.fromFlat(self[nid].bunches[o.nid].sample(d),n=o.class_name)-obje.empty(n=self[nid].label("ful")))
        p.nid, p.v = nid, False
        return p

    def rel_object(self,o,nid):
        assert o.nid in self[nid].ancs or nid in self[o.nid].ancs
        o_mid_chain = [o.nid]
        while self[o_mid_chain[0]].mid != 0: o_mid_chain.insert(0,self[o_mid_chain[0]].mid)
        n_mid_chain = [nid]
        while self[n_mid_chain[0]].mid != 0: n_mid_chain.insert(0,self[n_mid_chain[0]].mid)
        
        #print(o_mid_chain,n_mid_chain)
        while len(o_mid_chain) and len(n_mid_chain) and o_mid_chain[0] == n_mid_chain[0]:
            a = o_mid_chain[0]
            o_mid_chain.pop(0), n_mid_chain.pop(0)
        o_mid_chain.insert(0,a)
        n_mid_chain.insert(0,a)
        #print(list(reversed(o_mid_chain[:-1])),n_mid_chain)
        #print(self[215].mid)
        #print(self[6].mid)
        p = o
        for i in reversed(o_mid_chain[:-1]):
            p = self.rel_bunch(p,i)
        #print(p.nid)
        for i in n_mid_chain[1:]:
            p = self.rel_bunch(p,i)
        return p
    #endregion: utilize----------#

    #region: properties----------#
    def set_ancestors(self, nid=0, anc=[]):
        self[nid].ancs = [a for a in anc]
        for n in self[nid].edges:
            self.set_ancestors(n.endNode.idx, anc+[n.endNode.idx])
    
    # def set_tree_types(self, nid=0):
    #     for e in self[nid].edges:

    #         self[nid].tree_types = self[nid].tree_types | 
    #     if len(self[nid].edges) > 0:
    #         self[nid].tree_types = set([self[nid].type] + [t for e in self[nid].edges for t in self.set_tree_types(e.endNode.idx)])

    
    @property
    def type_2_nids(self):
        from ...Basic import object_types
        return { t: [n.idx for n in self.nods if n.label.n == self.merging[t]] for t in object_types }
    
    #endregion: properties-------#
