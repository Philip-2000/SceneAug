import os, numpy as np

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
    def __init__(self,type,suffix,idx=-1):
        self.type = type
        self.suffix = suffix
        self.source = None
        self.idx = idx
        self.chosen_level = 0
        self.edges = []
        self.bunches = {}

class merging():
    def __init__(self,d):
        self.d=d
    def __getitem__(self,a):
        from ..Basic.Obje import object_types
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
        self.workDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"pattern")#"./pattern/"
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
            config = yaml.load(open(os.path.join(self.treesDir,self.version+".yaml")), Loader=yaml.FullLoader)
            self.rootNames = config["rootNames"]
            # self.maxDepth = config["maxDepth"] if "maxDepth" in config else 10
            self.scaled = config["scale"] if "scale" in config else True
            # global DEN
            # global SIGMA2
            # DEN,SIGMA2 = config["DEN"], config["SIGMA2"]
        else:
            assert not os.path.exists(os.path.join(self.treesDir,self.version+".js"))
    
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

                nodeStr = ".".join(["\t"]*lev) + "%s nid=%d c=%.2f cc=%.2f from (%s mid=%d): %s"%(n.type,n.idx,e.confidence,e.confidenceIn,m.type,m.idx,str(m.bunches[n.idx]))

                resStr += nodeStr + "\n"
                
                lev += 1
                assert len(n.edges)<=1
                if len(n.edges)==0:
                    break
                e = n.edges[0]
        return resStr

    def mid(self,nid):
        m = self.nods[nid].source.startNode
        while not(nid in m.bunches):
            m = m.source.startNode
        return m.idx

    def createNode(self,fat,s,c=0.0,cc=0.0):
        self.nods.append(node(s,fat.suffix if len(fat.suffix)>0 else s.replace('/','.'),len(self.nods)))#nod(nn))
        fat.edges.append(edge(fat,self.nods[-1],c,cc))
        return len(self.nods)-1

    def freqInit(self,n):#筛选出里面有的物体，组成bunch 然后进行分析，每一次向下个分析都有相应的可信度
        from .Bnch import bnch
        from ..Basic.Obje import obje
        for s in self.rootNames:
            for ss in self.merging.reversed(s):
                if not os.path.exists(os.path.join(self.fieldDir,ss+".txt")):
                    open(os.path.join(self.fieldDir,ss+".txt"),"w").write('\n'.join([ str(sc) for sc in range(len(self.sDs)) if ss in [o.class_name() for o in self.sDs[sc].OBJES] ]))
                for f in open(os.path.join(self.fieldDir,ss+".txt")).readlines():
                    for o in [_ for _ in self.sDs[int(f)].OBJES if self.merging[_.class_name()] == s]:#F+=1
                        if (len(self.nods)) in n.bunches:
                            n.bunches[len(self.nods)].add(obje(np.array([0,0,0]),o.size,np.array([0]),i=self.merging[o.class_index],idx=o.idx,scne=o.scne),True)
                        else:
                            n.bunches[len(self.nods)] = bnch(obje(np.array([0,0,0]),o.size,np.array([0]),i=self.merging[o.class_index],idx=o.idx,scne=o.scne))
            n.bunches[len(self.nods)].enable(len(self.nods))
            self.createNode(n,s,len(n.bunches[len(self.nods)])/len(self.sDs),len(n.bunches[len(self.nods)])/len(self.sDs))

        self.Q = [e for e in n.edges]
        while len(self.Q):
            e = self.Q.pop(0)
            self.freq(e.endNode)#,[e.endNode.idx]

    def freq(self,n,ex=True,lev=0): #,path
        from .Bnch import bnches,giveup
        from ..Basic.Obje import object_types,noOriType
        path,m = [],n
        while m.idx != 0:
            path,m = (path if (m.type in noOriType) else [m.idx] + path), m.source.startNode
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
                            for r in res:#print(r.class_name())
                                blackLists[o.nid][self.merging[r.class_name()]].append(sheet[o.nid][self.merging[r.class_name()]].accept(r,1,blackLists[o.nid][self.merging[r.class_name()]]))
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
            
            self.createNode(n,self.merging[B[0].obs[0].class_name()],len(B[0])/len(field),len(B[0])/cnt)
            B[0].enable(len(self.nods)-1)#nn.idx)
            self.nods[B[1]].bunches[len(self.nods)-1] = B[0]

            if self.verb > 1:
                print(self.merging[B[0].obs[0].class_name()] + " " + str(len(B[0])) + "\n")

        if self.verb > 0:
            print('\t'.join(['\t']*lev+[e.endNode.type for e in n.edges]+["fuck"]))
        if lev < self.maxDepth:
            for e in n.edges:
                self.freq(e.endNode,ex,lev+1)#,(path if (e.endNode.type in noOriType) else path+[e.endNode.idx])
        self.Q = self.Q + [e for e in n.edges] if self.maxDepth == -1 else self.Q

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

    def storeTree(self):
        #if len(name) > 0:
        import yaml,json,inspect
        from .Bnch import giveup,DEN,SIGMA2
        lst = [{"type": N.type,"buncs":{i:[N.bunches[i].exp.tolist(),N.bunches[i].dev.tolist(),len(N.bunches[i])] for i in N.bunches},
                "edges":[(e.endNode.idx,e.confidence,e.confidenceIn) for e in N.edges]} for N in self.nods]
        open(os.path.join(self.treesDir,self.version+".js"),"w").write("var dat="+json.dumps(lst)+";")#open(os.path.join(self.treesDir,self.version+".json"),"w").write(json.dumps(lst))
        yaml.dump({"merging":self.merging.d,"rootNames":self.rootNames,"maxDepth":self.maxDepth,"scaled":self.scaled,"DEN":DEN,"SIGMA2":SIGMA2,"giveup":('\n'.join(inspect.getsource(giveup).split('\n')[1:-1]))},open(os.path.join(self.treesDir,self.version+".yaml"),"w"))

    def construct(self,maxDepth,scaled,sDs):
        self.maxDepth,self.scaled,self.sDs = maxDepth,scaled,sDs
        self.freqInit(self.nods[0])
        self.storeTree()
        self.draw()

    def draw(self,all=False,lim=5):
        import json
        from ..Basic.Obje import obje,object_types
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
            A = obje()#obje.fromFlat(B.exp,j=object_types.index(nn.type))
            for i in range(len(path)-1,0,-1):
                A = (A + obje.fromFlat(B.exp,j=object_types.index(path[i-1].type)))
                B = path[i].bunches[path[i-1].idx]
            B.draw(A,self.imgDir+self.version,str(n.idx),object_types.index(nn.type),self.scaled,all,lim,path)
            info[path[0].idx] = path[1].idx if len(path)>1 else 0

        open(self.imgDir+self.version+"/info.js","w").write("var info="+json.dumps(info)+";")
