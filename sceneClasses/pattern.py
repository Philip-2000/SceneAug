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
        self.idx = idx
        self.edges = []
        self.bunches = {}
        
class patternManager():
    def __init__(self,verb=0,maxDepth=1):
        self.version = ""
        self.sceneDir = "../novel3DFront/"
        self.workDir = "./pattern/"
        self.fieldDir = self.workDir+"fields/"
        self.treesDir = self.workDir+"trees/" #self.cnt = len(os.listdir(self.sceneDir))
        self.rootNames = ["King-size Bed"]#"Coffee Table","Dining Table",
        self.nods = [node("","",0)]#[nod(node("","",0))]
        self.verb = verb
        self.sDs = scneDs(self.sceneDir)
        if verb == 0:
            print("scene dataset loaded")
        self.maxDepth = maxDepth
        self.objectViewBd = 5
    
    def createNode(self,fat,s,c=0.0,cc=0.0):
        self.nods.append(node(s,fat.suffix if len(fat.suffix)>0 else s.replace('/','.'),len(self.nods)))#nod(nn))
        fat.edges.append(edge(fat,self.nods[-1],c,cc))

    def freq(self,n,ex=True,lev=0):
        lstt = open(self.fieldDir+n.suffix+".txt").readlines()
        field= [self.sDs[int(f)] for f in lstt if n.idx in self.sDs[int(f)].nids()]
        path,m = [n.idx],n
        while m.source is None:
            m = m.source.startNode
            path.append(m.idx)
        while 1:
            sheet = {id:{o:bnches() for o in object_types} for id in path}
            idset = set([e.endNode.idx for e in n.edges])
            cnt = 0
            for scene in field:#for f in lstt: assert n.idx in scene.nids() #scene = self.sDs[int(f)]
                if (not ex) or len(scene.nids() & idset) == 0:
                    cnt += 1
                    for o in scene.OBJES:
                        if o.nid in sheet: #o.nid == n.idx:
                            res = scene.objectView(o.idx,self.objectViewBd) #assert len(res) <= self.objectViewBd
                            for r in res:#print(r.class_name())
                                sheet[o.nid][r.class_name()].accept(r,1)
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
                self.freq(e.endNode,ex,lev+1)

    def loadTre(self,dct,id=0):
        if id == 0:
            self.nods += [None for _ in dct] #nod(None)
        suf = "" if id == 0 else self.nods[id].suffix+"+"
        for b in dct[id]["buncs"]:
            self.nods[id].bunches[b] = bnch(None,np.array(dct[id]["buncs"][b][0]),np.array(dct[id]["buncs"][b][1]))
        for f in dct[id]["edges"]:
            e = f[0]
            self.nods[e] = node(dct[e]["type"],suf+dct[id]["type"])
            self.nods[id].edges.append(edge(self.nods[id],self.nods[e],c=f[1],cc=f[2]))
            self.loadTre(dct,e)
            if id>0:
                suf += "--"+dct[e]["type"]

    def storeTree(self,name):
        if len(name) > 0:
            lst = [{"type": N.type,"buncs":{i:[N.bunches[i].exp.tolist(),N.bunches[i].dev.tolist(),len(N.bunches[i])] for i in N.bunches},
                    "edges":[(e.endNode.idx,e.confidence,e.confidenceIn) for e in N.edges]} for N in self.nods]
            open(self.treesDir+name+".json","w").write(json.dumps(lst))
        
    def treeConstruction(self,load="",name=""):
        if load != "":
            return self.loadTre(json.load(open(self.treesDir+load+".json")))
        for s in self.rootNames:
            self.createNode(self.nods[0],s)
            for f in open(self.fieldDir+self.nods[-1].suffix+".txt").readlines():
                for o in [_ for _ in self.sDs[int(f)].OBJES if _.class_name() == self.nods[-1].type]:#F+=1
                    o.nid = self.nods[-1].idx
        for n in self.nods[0].edges:
            self.freq(n.endNode)#frequentRecursive(n.endNode,form)#
        self.storeTree(name)

    def draw(self):
        #画什么？
        #它是什么？
        #每个node都需要画。
        #以根为参考系
        #以源的期望为参照物，

        #把obs里的东西全画上
        #把线连上？哪里有线啊？怎么连呢？初始树状结构，是有的，但是就是说这个，bnch，这个关系，其实可以通过额，
        #一方面是图片中有原始信息，另一方面是额，这个，比如说，鼠标挪上去后，源节点加粗或者凸显之类的，可以

        pass

##################
import sys
import argparse
def parse(argv):
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-v','--verbose', default=0)
    parser.add_argument('-d','--maxDepth', default=2)
    parser.add_argument('-n','--name', default="")
    parser.add_argument('-l','--load', default="")
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__": #load="testings",
    args=parse(sys.argv[1:])
    #patternManager().singleExistStorage(object_types[:-2])#frequent(["L-shaped Sofa"],store=False)
    patternManager(verb=int(args.verbose),maxDepth=int(args.maxDepth)).treeConstruction(load=args.load,name=args.name)#