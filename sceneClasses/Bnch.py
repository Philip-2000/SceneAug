import numpy as np
from Obje import *
#from Scne import *

DEN,SIGMA2,MDE,OPTRATE = [[0.9**2,0.5**2,0.9**2,0.9**2,0.5**2,0.9**2,0.5**2]]*5,2.0**2,True,0.5
def giveup(B,A,C):#c=len(B)/A, cc=len(B)/C
    return (B is None) or (len(B) < 10) #or (len(B)/C < 0.3) or (len(B)/A < 0.1)
def singleMatch(l,c,cc,od,cs):
    #匹配上了，是一件好事，加分100，但是如果空间分布差异比较大，就扣掉一些分，也就是100-l；
    #如果是在前面给出的，那么算是比较常见的，cs越小越靠前，但如果比较靠后，那么即使比较适配我们也认为你有一定问题？
    return 100-l

class bnch():
    def __init__(self,obj,exp=None,dev=None):
        self.exp = exp if obj is None else obj.flat()
        self.dev = dev if obj is None else np.array(DEN[0])
        self.obs = []  if obj is None else [obj]
    
    def __len__(self):
        return len(self.obs)

    def sample(self):
        return self.exp + np.random.randn(self.exp.shape[-1])*(self.dev**(0.5))/5#np.array([5,1,5,5,1,5,5])

    def loss(self,obj):
        return ((obj.flat()-self.exp)**2/self.dev).sum()

    def test(self,obj):
        return ((obj.flat()-self.exp)**2).sum()

    def optimize(self,obj):
        return obje.fromFlat((obj.flat()-self.exp)*OPTRATE+self.exp,j=obj.class_index)

    def add(self,obj,f=False):
        if self.accept(obj) or f:
            self.exp = (self.exp * len(self.obs) + obj.flat()) / (len(self.obs)+1)
            self.obs.append(obj)
            self.dev = np.average(np.array([(o.flat()-self.exp)**2 for o in self.obs]+DEN),axis=0) if MDE else self.dev
            return True
        return False

    def accept(self,obj):#,add=True):
        return np.min(self.dev*SIGMA2 - (obj.flat()-self.exp)**2) > 0
    
    def refresh(self):
        self.exp = np.average(np.array([o.flat() for o in self.obs]),axis=0) if len(self.obs) > 0 else self.exp
        self.dev = np.average(np.array([(o.flat()-self.exp)**2 for o in self.obs]+DEN),axis=0) if MDE and (len(self.obs) > 0) else self.dev
        return len(self.obs) > 0

    def enable(self,nid):
        for o in self.obs:#if o.scne.OBJES[o.idx].nid != -1: print(o.scne.scene_uid+" "+str(o.idx)+" "+o.scne.OBJES[o.idx].class_name()) #assert 1 == 0
            o.scne.OBJES[o.idx].nid = nid

    def draw(self,basic,dir,idx,J,scaled,all,lim,path,offset=[4.0,0.0]):
        plt.axis('equal')
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)

        if len(path)>1:
            basic.draw(d=True,color="black",cr="blue")
        if all and len(self.obs)>0:
            for a in self.obs:
                basic.rely(a).draw(color="red",alpha=1.0/len(self.obs))
        else:
            me = basic.rely(obje.fromFlat(self.exp,j=J),scaled)
            if len(path)>1:
                plt.Rectangle((me.translation[0],-me.translation[2]),width=self.dev[0],height=self.dev[2],color="yellow")
                plt.plot([me.translation[0], me.translation[0]+0.5*np.math.sin(me.orientation+self.dev[-1])], [-me.translation[2],-me.translation[2]-0.5*np.math.cos(me.orientation+self.dev[-1])], color="lime")
                plt.plot([me.translation[0], me.translation[0]+0.5*np.math.sin(me.orientation-self.dev[-1])], [-me.translation[2],-me.translation[2]-0.5*np.math.cos(me.orientation-self.dev[-1])], color="lime")
                fat = path[0].source.startNode
                while fat.idx != path[1].idx:
                    if fat.idx in path[1].bunches:
                        basic.rely(obje.fromFlat(path[1].bunches[fat.idx].exp,j=object_types.index(fat.type)),scaled).draw(color="gray",d=True,cr="gray")
                    fat = fat.source.startNode

            me.draw(d=True,color="red",cr="green")
            me.translation[0] = offset[0]
            me.size = me.size + self.dev[3:6]
            me.draw(d=False,color="pink")
            me.size = me.size - self.dev[3:6]
            me.draw(d=False,color="red")
            me.size = me.size - self.dev[3:6]
            me.draw(d=False,color="pink")

        plt.savefig(dir+"/"+idx+".png")
        plt.clf()

class bnches():
    def __init__(self):
        self.bunches = []

    def __len__(self):
        return len(self.bunches)

    def mx(self):
        return None if len(self.bunches)==0 else sorted(self.bunches,key=lambda x:-len(x))[0]

    def all(self):
        return sum([len(b) for b in self.bunches])
    
    def refresh(self):
        obs = []
        for b in self.bunches:
            obs += b.obs
            b.obs = [] 
        for o in obs:
            for b in self.bunches:
                if b.add(o):
                    break
        for b in self.bunches:
            if not b.refresh():
                self.bunches.remove(b)

    def accept(self,obj,create=True,blackList=[]):
        for t in sorted([(b,self.bunches[b].test(obj)) for b in range(len(self.bunches)) if (not b in blackList)], key=lambda x:x[1]):#tests:#
            if self.bunches[t[0]].add(obj):
                return t[0]
        if create:
            self.bunches.append(bnch(obj))
            return len(self.bunches)-1
        return -1
