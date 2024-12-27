import numpy as np

DEN,SIGMA2,MDE = [[1.2**2,1.2**2,1.2**2,1.2**2,1.2**2,1.2**2,0.5**2]]*5,2.0**2,True
def giveup(B,A,C):#c=len(B)/A, cc=len(B)/C  #or (len(B)/C < 0.3) or (len(B)/A < 0.1)
    return (B is None) or (len(B) < 30) 
def singleMatch(l,c,cc,od,cs):
    return 15-l*1

class bnch():
    def __init__(self,obj,exp=None,dev=None):
        self.exp = exp if obj is None else obj.flat()
        self.dev = dev if obj is None else np.array(DEN[0])
        self.obs = []  if obj is None else [obj]
    
    def __len__(self):
        return len(self.obs)
    
    def __str__(self):
        return "("+",".join(["%.3f±%.3f"%(self.exp[i],self.dev[i],) for i in range(len(self.exp))])+")"

    def sample(self,d=-1):
        return self.exp + np.random.randn(self.exp.shape[-1])*((self.dev if d<0 else d)**(0.5))/5#np.array([5,1,5,5,1,5,5])

    def diff(self,obj): #self.diff(obj)
        from ..Basic.Obje import angleNorm
        a = obj.flat()-self.exp
        return np.concatenate([a[:-1],[angleNorm(a[-1])]])

    def loss(self,obj,hint=1):#Nothing for /1       None for /self.dev       np.array for weight
        return ((self.diff(obj))**2).sum() if type(hint)==int else (((self.diff(obj))**2/self.dev).sum() if hint is None else (hint*(self.diff(obj))**2/self.dev).sum())#/self.dev 

    def optimize(self,obj,iRate):
        from ..Basic.Obje import obje
        return obje.fromFlat((self.diff(obj))*iRate+self.exp,j=obj.class_index)

    def add(self,obj,f=False):
        from ..Basic.Obje import angleNorm
        if self.accept(obj) or f:
            self.exp = (self.exp * (len(self.obs) + 1) + self.diff(obj)) / (len(self.obs)+1)
            self.exp[-1] = angleNorm(self.exp[-1])
            self.obs.append(obj)
            self.dev = np.average(np.array([(self.diff(obj))**2 for o in self.obs]+DEN),axis=0) if MDE else self.dev
            return True
        return False

    def accept(self,obj):#,add=True):
        return np.min(self.dev*SIGMA2 - (self.diff(obj))**2) > 0
    
    def refresh(self):
        if len(self.obs)==0:
            return False
        self.exp = np.zeros_like(self.obs[0].flat())
        
        for oi in range(len(self.obs)):
            self.exp = (self.exp * (oi + 1) + self.diff(self.obs[oi])) / (oi+1)

        #self.exp = np.average(np.array([o.flat() for o in self.obs]),axis=0) if len(self.obs) > 0 else self.exp
        self.dev = np.average(np.array([(self.diff(o))**2 for o in self.obs]+DEN),axis=0) if MDE and (len(self.obs) > 0) else self.dev
        return len(self.obs) > 0

    def enable(self,nid):
        for o in self.obs:#if o.scne[o.idx].nid != -1: print(o.scne.scene_uid+" "+str(o.idx)+" "+o.scne[o.idx].class_name()) #assert 1 == 0
            o.scne[o.idx].nid = nid

    def draw(self,basic,dir,idx,J,scaled,all,lim,path,offset=[4.0,0.0]):
        from matplotlib import pyplot as plt
        from ..Basic.Obje import obje, object_types
        plt.axis('equal')
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)

        if len(path)>1:
            basic.draw(d=True,color="black",cr="blue",text=False)
        if all and len(self.obs)>0:
            for a in self.obs:
                (basic + a).draw(color="red",alpha=1.0/len(self.obs),text=False)
        else:
            me = basic + (obje.fromFlat(self.exp,j=J))
            if len(path)>1:
                plt.Rectangle((me.translation[0],-me.translation[2]),width=self.dev[0],height=self.dev[2],color="yellow")
                plt.plot([me.translation[0], me.translation[0]+0.5*np.math.sin(me.orientation+self.dev[-1])], [-me.translation[2],-me.translation[2]-0.5*np.math.cos(me.orientation+self.dev[-1])], color="lime")
                plt.plot([me.translation[0], me.translation[0]+0.5*np.math.sin(me.orientation-self.dev[-1])], [-me.translation[2],-me.translation[2]-0.5*np.math.cos(me.orientation-self.dev[-1])], color="lime")
                fat = path[0].source.startNode
                while fat.idx != path[1].idx:
                    if fat.idx in path[1].bunches:
                        (basic + obje.fromFlat(path[1].bunches[fat.idx].exp,j=object_types.index(fat.type))).draw(color="gray",d=True,cr="gray",text=False)
                    fat = fat.source.startNode

            me.draw(d=True,color="red",cr="green",text=False)
            me.translation[0] = offset[0]
            me.size = me.size + self.dev[3:6]
            me.draw(d=False,color="pink",text=False)
            me.size = me.size - self.dev[3:6]
            me.draw(d=False,color="red",text=False)
            me.size = me.size - self.dev[3:6]
            me.draw(d=False,color="pink",text=False)

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
            #sorted([(b,self.bunches[b].loss(o)) for b in range(len(self.bunches))], key=lambda x:x[1])
            for b in self.bunches:
                if b.add(o):
                    break
        for b in self.bunches:
            if not b.refresh():
                self.bunches.remove(b)

    def accept(self,obj,create=True,blackList=[]):
        for t in sorted([(b,self.bunches[b].loss(obj)) for b in range(len(self.bunches)) if (not b in blackList)], key=lambda x:x[1]):#tests:#
            if self.bunches[t[0]].add(obj):
                return t[0]
        if create:
            self.bunches.append(bnch(obj))
            return len(self.bunches)-1
        return -1

class bnch_node():
    def __init__(self, oid, nid, fid, scene):
        self.oid = oid
        self.nid = nid
        self.fid = fid
        self.child_id = []
        self.o = scene[oid]
        self.norm = scene[oid].adjust.norm()
        #print(self.norm)
        self.I,self.J = nid,oid

    
    def Str(self, tree, lev):
        return ".".join(["\t"]*lev)+"(%d %d %s)"%(self.nid, self.oid, tree.scene[self.oid].class_name()) + ("\n" if len(self.child_id) else "") + "\n".join([tree[i].Str(tree,lev+1) for i in self.child_id])

    def upward(self, tree):
        if len(self.child_id):
            self.I,self.J = self.nid,self.oid
            for cid in self.child_id:
                if tree[cid].norm > self.norm:
                    self.norm = tree[cid].norm
                    self.I,self.J = cid, tree[cid].oid
            if self.I != self.nid:
                from ..Basic.Obje import obje
                #print("assert")
                self.o = tree[self.I].o+(obje.fromFlat(tree.pm[self.nid].bunches[self.I].exp,j=tree.scene[tree[self.I].oid].class_index)-obje.empty(j=tree.scene[self.oid].class_index))
                #print(self.o)

    def downward(self, tree, ir):
        from ..Basic.Obje import obje
        exp_son = tree.scene[tree[self.fid].oid] + obje.fromFlat(tree.pm[self.fid].bunches[self.nid].exp, j=tree.scene[self.oid].class_index) if self.fid else self.o
        tree.scene[self.oid].adjust.toward(exp_son, ir)
        [tree[cid].downward(tree, ir) for cid in self.child_id]

class bnch_tree():
    def __init__(self, pla, pm, scene):
        self.scene, self.root, self.pm = scene, pla.nids[0][1], pm
        self.nodes = { self.root:bnch_node(pla.nids[0][0], self.root, 0, scene) }
        for i, (oid, nid) in enumerate(pla.nids[1:]):
            if pla.nids[i][1] == nid:
                scene[oid].v = False
            else:
                self.nodes[nid] = bnch_node(oid, nid, pm[nid].mid, scene)
                self.nodes[pm[nid].mid].child_id.append(nid)#print(self)

    def __str__(self):
        return self.nodes[self.root].Str(self, 0)
            
    def __getitem__(self, nid):
        return self.nodes[nid]
    
    def __call__(self, oid):
        return [self.nodes[nid] for nid in self.nodes if self.nodes[nid].oid == oid][0]
    
    def __len__(self):
        return len(self.nodes)
    
    def __iter__(self):
        return iter(sorted(self.nodes.items(), key= lambda x:x[0]))
    
    def optimize(self, ir):
        for nid, nod in sorted(self.nodes.items(), key= lambda x:-x[0]):
            nod.upward(self)
        #print(self)
        #print(self.pm[self[self.root].nid])
        #print(self[self.root].o)
        self[self.root].o.align(self.pm[self[self.root].nid])
        self[self.root].downward(self, ir)
        return {n.oid:n.J for n in self.nodes.values()}
