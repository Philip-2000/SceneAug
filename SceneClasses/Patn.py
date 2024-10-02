import os
import numpy as np
from .Obje import *
from .Scne import *
import json
import yaml
from .Bnch import *
from matplotlib import pyplot as plt
import inspect

class edge():
    def __init__(self,n,nn,c,cc):
        self.startNode = n
        self.endNode = nn
        self.confidence = c
        self.confidenceIn = cc
        nn.source = self

class node():
    def __init__(self,type,suffix,idx=-1):
        self.type = type
        self.suffix = suffix
        self.source = None
        self.idx = idx
        self.edges = []
        self.bunches = {}

class merging():
    def __init__(self,d):
        self.d=d
    def __getitem__(self,a):
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
        self.workDir = "./pattern/"
        self.fieldDir = os.path.join(self.workDir,"fields/")
        self.imgDir = os.path.join(self.workDir,"imgs/")
        self.treesDir = os.path.join(self.workDir,"trees/") #self.cnt = len(os.listdir(self.sceneDir))
        self.rootNames=["Dining Table","Coffee Table","King-size Bed","Desk","Dressing Table"]#,"Single bed","Kids Bed"]#
        self.merging = merging(
            {"Single bed":"King-size Bed", "Kids Bed":"King-size Bed",
             "Loveseat Sofa":"Three-seat / Multi-seat Sofa",
             "Lounge Chair / Cafe Chair / Office Chair":"Dining Chair", "Classic Chinese Chair":"Dining Chair", "Dressing Chair":"Dining Chair", "armchair":"Dining Chair",
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
    
    def createNode(self,fat,s,c=0.0,cc=0.0):
        self.nods.append(node(s,fat.suffix if len(fat.suffix)>0 else s.replace('/','.'),len(self.nods)))#nod(nn))
        fat.edges.append(edge(fat,self.nods[-1],c,cc))

    def freqInit(self,n):
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
        path,m = [],n
        while m.idx != 0:
            path,m = (path if (m.type in noOriType) else [m.idx] + path), m.source.startNode
        field= [self.sDs[int(f)] for f in open(self.fieldDir+n.suffix+".txt").readlines() if n.idx in self.sDs[int(f)].nids()]
        while 1:
            sheet = {id:{o:bnches() for o in object_types} for id in path}
            idset = set([e.endNode.idx for e in n.edges])
            cnt = 0
            for scene in field:#for f in lstt: assert n.idx in scene.nids() #scene = self.sDs[int(f)]
                blackLists = {id:{o:[] for o in object_types} for id in path}
                if (not ex) or len(scene.nids() & idset) == 0:
                    cnt += 1
                    for o in scene.OBJES:
                        if o.nid in sheet: #o.nid == n.idx:
                            res = scene.objectView(o.idx,self.objectViewBd,self.scaled) #assert len(res) <= self.objectViewBd
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
        lst = [{"type": N.type,"buncs":{i:[N.bunches[i].exp.tolist(),N.bunches[i].dev.tolist(),len(N.bunches[i])] for i in N.bunches},
                "edges":[(e.endNode.idx,e.confidence,e.confidenceIn) for e in N.edges]} for N in self.nods]
        open(os.path.join(self.treesDir,self.version+".js"),"w").write("var dat="+json.dumps(lst)+";")#open(os.path.join(self.treesDir,self.version+".json"),"w").write(json.dumps(lst))
        yaml.dump({"merging":self.merging.d,"rootNames":self.rootNames,"maxDepth":self.maxDepth,"scaled":self.scaled,"DEN":DEN,"SIGMA2":SIGMA2,"giveup":('\n'.join(inspect.getsource(giveup).split('\n')[1:-1]))},open(os.path.join(self.treesDir,self.version+".yaml"),"w"))

    def construct(self,maxDepth,scaled,sDs):
        self.maxDepth,self.scaled,self.sDs = maxDepth,scaled,sDs
        self.freqInit(self.nods[0])
        self.storeTree()
        self.draw(self.version)

    def treeConstruction(self,load="",name="",draw=True):#print(self.treesDir+load+".json")
        raise NotImplementedError
        self.version = name if load == "" else load
        if load != "":
            self.loadTre(json.loads(open(os.path.join(self.treesDir,self.version+".js")).read()[8:-1]))#self.loadTre(json.load(open(os.path.join(self.treesDir,self.version+".js"))))
            config = yaml.load(open(os.path.join(self.treesDir,self.version+".yaml")), Loader=yaml.FullLoader)
            self.rootNames = config["rootNames"]
            self.maxDepth = config["maxDepth"] if "maxDepth" in config else self.maxDepth
            global DEN
            global SIGMA2
            DEN,SIGMA2 = config["DEN"], config["SIGMA2"]
        else:
            self.freqInit(self.nods[0])
            self.storeTree()
        #if draw:
            self.draw(load if len(load) > 0 else name)

    def draw(self,name,all=False,lim=5):
        if not os.path.exists(self.imgDir+name+"/"):
            os.makedirs(self.imgDir+name+"/")
        info = {}#open(self.imgDir+name+"/info.json")

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
                A = A.rely(obje.fromFlat(B.exp,j=object_types.index(path[i-1].type)),self.scaled) #A.rely(obje(B.exp[:3],B.exp[3:6],B.exp[-1:]),self.scaled)
                B = path[i].bunches[path[i-1].idx]
            B.draw(A,self.imgDir+name,str(n.idx),object_types.index(nn.type),self.scaled,all,lim,path)
            info[path[0].idx] = path[1].idx if len(path)>1 else 0

        open(self.imgDir+name+"/info.js","w").write("var info="+json.dumps(info)+";")

    def tempDebugger(self,theScene,cObjectList,o,spc,imgName):
        tmpScene = scne.empty()
        wa = copy(theScene.WALLS)
        cObjectLis = copy(cObjectList)
        sp = copy(theScene.SPCES)
        sp.SPCES.append(spc)
        tmpScene.registerWalls(wa)
        tmpScene.registerSpces(sp)
        for oo in cObjectLis:
            pp = spc.transformOutward([oo])
            tmpScene.addObject(pp[0])
        if o is not None:
            p = spc.transformOutward([o])
            tmpScene.addObject(copy(p[0]))
        #tmpScene.imgDir="./spce_generate/"
        tmpScene.draw(imageTitle="./spce_generate/"+imgName,d=True,lim=5)
        
    def generate(self,nm="generate",theScene=None,useWalls=False,rots=[],useText=False,debug=False):
        #assert (useWalls or useText)
        if useWalls:
            assert theScene.WALLS is not None
            #f = open("./spce_generate/log.txt",'w')#sys.stdout#
            if len(rots)==0:
                rots = ["King-size Bed","Dressing Table"] if np.random.rand()<-0.5 else ["Coffee Table","Dining Table"]

            Sps = spces(wals=theScene.WALLS,drawFolder="./spce_generate/"+theScene.WALLS.name+"/")
            theScene.registerSpces(Sps)
            for r in rots:#[:1]
                if rots.index(r)>-1:
                    spc = Sps.extractingSpce(hint=[5,5])
                else:
                    PRO = prob(np.array([-0.2,0,-0.04]))
                    PRO.res=[(3.1785,True),(2.6265,True),(3.1785,True),(1.2910,True)]
                    PRO.areaFunctionDetection(theScene.WALLS,Sps.delta)
                    spc = spce.fromPro(PRO,scne=theScene,delta=Sps.delta)

                cObjectList = []
                rf = self.nods[0].bunches[self.rootNames.index(r)+1].exp
                N = self.nods[self.rootNames.index(r)+1]
                ro= obje.fromFlat(rf,j=object_types.index(N.type))
                ro.translation = np.copy(ro.size)
                ro.samplesBound()
                ro.nid = N.idx
                cObjectList.append(ro)

                self.tempDebugger(theScene,cObjectList,None,spc,"%d os=%d"%(rots.index(r),len(cObjectList)))

                adding=True
                while adding:
                    cs,m = 0,None
                    if len(N.edges)==0:
                        break
                    for ed in N.edges:
                        cs += ed.confidence
                        if ed in N.edges and N.edges.index(ed)==0:#np.random.rand() < ed.confidenceIn:
                            N,m = ed.endNode,ed.startNode
                            BD = np.array([o.samplesBound()[1] for o in cObjectList]).max(axis=0)
                            area = 2*spc.relA-BD #np.array([BD[1]-2*spc.a,[0,0,0]]).max(axis=0)
                            # print("---------------------start------------------"+str(len(cObjectList)))
                            # print("spc.relA",spc.relA,"BD",BD,"area",area)
                            Ntype = N.type if N.type.find('/') == -1 else N.type[:N.type.find('/')]

                            while not (N.idx in m.bunches):
                                m = m.source.startNode
                            ro = m.bunches[N.idx].exp
                            a = [o for o in cObjectList if o.nid == m.idx]
                            o = a[0].rely(obje.fromFlat(ro,j=object_types.index(N.type)),self.scaled)
                            o.nid = N.idx

                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s before viola"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))

                            delta = 0.04
                            bd = o.samplesBound()
                            viola = [np.array([bd[0],[0,0,0]]).min(axis=0),np.array([bd[1]-2*spc.relA,[0,0,0]]).max(axis=0)]
                            # print(viola) viola[0] is zero or negative, viola[1] is zero or positive
                            if min(viola[0][0],viola[0][2]) < -delta*2 and max(viola[1][0],viola[1][2]) > delta*2:
                                # print("viola[0].min() < -delta*5 and viola[1].max() > delta*5 no space")
                                adding=False
                                break
                            if min(viola[0][0],viola[0][2]) > -delta*2 and max(viola[1][0],viola[1][2]) < delta*2:
                                # print("no viola")
                                self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s with no voila"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))
                                cObjectList.append(o)
                                
                                continue

                            moving = (min(viola[0][0],viola[0][2]) < -delta*2)
                            #vio = viola[0] if min(viola[0][0],viola[0][2]) < -delta*2 else viola[1]
                            if moving:
                                # print("moving "+str(len(cObjectList)))
                                if min(area[0]+viola[0][0],area[2]+viola[0][2]) <-delta*0:
                                    # print("too large viola, please find another node")
                                    continue
                                else:
                                    vio0 = [viola[0][0],viola[0][1],viola[0][2]]
                                    viola[0] = [max(viola[0][0],-area[0]),viola[0][1],max(viola[0][2],-area[2])]
                                    vio = [(viola[0][0]+vio0[0])/2.0,viola[0][1],(viola[0][2]+vio0[2])/2.0]
                            else:
                                if max(viola[1][0],viola[1][2]) > delta*0:
                                    # print("too large viola, please find another node")
                                    continue
                                else:
                                    vio = viola[1]/2.0
                            
                            o.translation -= vio

                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s after fixing vio"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))
                            if not moving:
                                collide = False
                                for oo in cObjectList:
                                    if oo.class_name().find("Lamp")==-1:
                                        a,b=oo.distance(o)
                                        collide = b
                                        if collide:
                                            break
                                if collide:
                                    # print("collision occurs while fixing viola")
                                    continue

                            if moving:
                                for oo in cObjectList:
                                    oo.translation -= vio
                            
                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s after moving everything"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))

                            cObjectList.append(o)
                        else:
                            break

                for oo in cObjectList:
                    theScene.addObject(oo)
                
                BD = np.array([o.samplesBound()[1] for o in cObjectList]).max(axis=0)
                spc.recycle(BD,Sps.WALLS)
                Sps.SPCES.append(spc)
                Sps.draw()
                spc.scne=theScene

                self.tempDebugger(theScene,cObjectList,None,spc,"%d os=%d after recycling"%(rots.index(r),len(cObjectList)))
                            
                a = Sps.eliminatingSpace(spc)
                if not a:
                    break
    
            return

        if useText is not None:
            return

        
        scene = scne.empty(nm) if theScene is None else theScene
        scene.imgDir = "./pattern/gens/"
        N = self.nods[0]
        while len(N.edges)>0:
            cs = 0
            for ed in N.edges:
                cs += ed.confidence
                if np.random.rand() < ed.confidenceIn:
                    N,m = ed.endNode,ed.startNode
                    while not (N.idx in m.bunches):
                        m = m.source.startNode
                    r = m.bunches[N.idx].sample()
                    a = [o for o in scene.OBJES if o.nid == m.idx] if m.idx > 0 else [obje(np.array([0,0,0]),np.array([1,1,1]),np.array([0]))]
                    o = a[0].rely(obje.fromFlat(r,j=object_types.index(N.type)),self.scaled)
                    scene.addObject(o)
                    o.nid = N.idx
                    if m.idx > 0:
                        scene.LINKS.append(objLink(a[0].idx,o.idx,len(scene.LINKS),scene))
                    cs = 0
                    break
            
                if np.random.rand() < cs:
                    break
        scene.draw(d=True)

    def completion(self,scne):
        #we will not extract space in the room walls then,
        #It's so weird, while the nodes are scattering in the tree
        scne.draw(d=True)

    def rearrangment(self,scne):
        #search in the tree.
        #assign nodes for these objects.
        scne.draw(d=True)