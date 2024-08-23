import os
import numpy as np
from Obje import *
from Scne import *
import json
from Bnch import *
from matplotlib import pyplot as plt

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
        
class patternManager():
    def __init__(self,verb=0,maxDepth=1,s=False,loadDataset=True):
        self.version = ""
        self.sceneDir = "../novel3DFront/"
        self.workDir = "./pattern/"
        self.fieldDir = self.workDir+"fields/"
        self.imgDir = self.workDir+"imgs/"
        self.treesDir = self.workDir+"trees/" #self.cnt = len(os.listdir(self.sceneDir))
        self.rootNames = ["Dining Table","King-size Bed","Coffee Table","Single bed"]#
        self.nods = [node("","",0)]#[nod(node("","",0))]
        self.verb = verb
        if loadDataset:
            self.sDs = scneDs(self.sceneDir)
            if verb == 0:
                print("scene dataset loaded")
        self.maxDepth = maxDepth
        self.objectViewBd = 6
        self.scaled=s
    
    def createNode(self,fat,s,c=0.0,cc=0.0):
        self.nods.append(node(s,fat.suffix if len(fat.suffix)>0 else s.replace('/','.'),len(self.nods)))#nod(nn))
        fat.edges.append(edge(fat,self.nods[-1],c,cc))

    def freqInit(self,n):
        for s in self.rootNames:
            for f in open(self.fieldDir+s+".txt").readlines():
                for o in [_ for _ in self.sDs[int(f)].OBJES if _.class_name() == s]:#F+=1
                    if (len(self.nods)) in n.bunches:
                        n.bunches[len(self.nods)].add(obje(np.array([0,0,0]),o.size,np.array([0]),i=o.class_index,idx=o.idx,scne=o.scne),True)
                    else:
                        n.bunches[len(self.nods)] = bnch(obje(np.array([0,0,0]),o.size,np.array([0]),i=o.class_index,idx=o.idx,scne=o.scne))
            n.bunches[len(self.nods)].enable(len(self.nods))
            self.createNode(n,s,len(n.bunches[len(self.nods)])/len(self.sDs),len(n.bunches[len(self.nods)])/len(self.sDs))
            
        for e in n.edges:
            self.freq(e.endNode,[e.endNode.idx])#frequentRecursive(n.endNode,form)#

    def freq(self,n,path,ex=True,lev=0):
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
                                blackLists[o.nid][r.class_name()].append(sheet[o.nid][r.class_name()].accept(r,1,blackLists[o.nid][r.class_name()]))
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
            
            self.createNode(n,B[0].obs[0].class_name(),len(B[0])/len(field),len(B[0])/cnt)
            B[0].enable(len(self.nods)-1)#nn.idx)
            self.nods[B[1]].bunches[len(self.nods)-1] = B[0]

            if self.verb > 1:
                print(B[0].obs[0].class_name() + " " + str(len(B[0])) + "\n")

        if self.verb > 0:
            print('\t'.join(['\t']*lev+[e.endNode.type for e in n.edges]+["fuck"]))
        if lev < self.maxDepth:
            for e in n.edges:
                self.freq(e.endNode,(path if (e.endNode.type in noOriType) else path+[e.endNode.idx]),ex,lev+1)

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

    def storeTree(self,name):
        if len(name) > 0:
            lst = [{"type": N.type,"buncs":{i:[N.bunches[i].exp.tolist(),N.bunches[i].dev.tolist(),len(N.bunches[i])] for i in N.bunches},
                    "edges":[(e.endNode.idx,e.confidence,e.confidenceIn) for e in N.edges]} for N in self.nods]
            open(self.treesDir+name+".js","w").write("var dat="+json.dumps(lst)+";")
            open(self.treesDir+name+".json","w").write(json.dumps(lst))
        
    def treeConstruction(self,load="",name="",draw=True):#print(self.treesDir+load+".json")
        if load != "":
            self.loadTre(json.load(open(self.treesDir+load+".json")))
        else:
            self.freqInit(self.nods[0])
            self.storeTree(name)
        if draw:
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

    def generate(self,nm="generate"):
        scene = scne.empty(nm)
        scene.imgDir = "./gens/"
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

##################
import sys
import argparse
def parse(argv):
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-v','--verbose', default=0)
    parser.add_argument('-d','--maxDepth', default=2)
    parser.add_argument('-n','--name', default="")
    parser.add_argument('-l','--load', default="vise")
    parser.add_argument('-s','--scaled', default=True, action="store_true")
    parser.add_argument('-u','--uid', default="0a9f23f6-f0a6-4cbb-8db5-48be2996d10a_LivingDiningRoom-507")
    parser.add_argument('-g','--gen', default="")
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__": #load="testings",
    args=parse(sys.argv[1:])
    T = patternManager(verb=int(args.verbose),maxDepth=int(args.maxDepth),s=args.scaled,loadDataset=(len(args.load)==0))
    T.treeConstruction(load=args.load,name=args.name,draw=len(args.name)>0 or (len(args.uid)==0 and len(args.gen)==0))#
    if len(args.gen)>0:
        for i in range(16):
            T.generate(str(i))#args.gen)
    elif len(args.uid)>0:
        scne(fullLoadScene(args.uid),grp=False,cen=True,wl=True).tra(T)