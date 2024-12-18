import numpy as np
from numpy.linalg import norm
from copy import copy
grupC=["black","red","gray","purple","yellow","red","gray","purple","yellow"]
object_types = ["Pendant Lamp", "Ceiling Lamp", "Bookcase / jewelry Armoire", \
"Round End Table", "Dining Table", "Sideboard / Side Cabinet / Console table", "Corner/Side Table", "Desk", "Coffee Table", "Dressing Table", \
"Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Wine Cabinet", \
"Lounge Chair / Cafe Chair / Office Chair", "Classic Chinese Chair", "Dressing Chair", "Dining Chair", "armchair", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", \
"Three-seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "Lazy Sofa", "Chaise Longue Sofa", "Wardrobe", "TV Stand", "Nightstand", \
"King-size Bed", "Kids Bed", "Bunk Bed", "Single bed", "Bed Frame", "window", "door"]
noOriType = ["Pendant Lamp", "Ceiling Lamp", "Round End Table", "Corner/Side Table", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", "Nightstand"]
SCL = True
def angleNorm(ori): #ori = ori % (2*np.math.pi)  #return ori-2*np.math.pi if 2*np.math.pi-ori < ori else ori
    return min(ori%(2*np.math.pi), ori%(2*np.math.pi)-2*np.math.pi, key=lambda x:abs(x))

def reversedObj(o,j): #reversedObj(bx2d.fromFlat(m.bunches[nid].exp),self.scene[fid].class_index)
    return o-obje.empty(j)
class bx2d(): #put those geometrical stuff into this base class
    def __init__(self,t=np.array([.0,.0,.0]),s=np.array([1.,1.,1.]),o=np.array([.0]),b=None): #no semantical information here
        self.translation = t if not b else b.translation
        self.size        = s if not b else b.size
        self.orientation = o if not b else b.orientation
        from ..Operation.Adjs import adj
        self.adjust = adj(o=self,call=False)
        
    #region: in/outputs----------#

        #region: inputs----------#
    @classmethod
    def fromFlat(cls,v):
        return cls(v[:3],v[3:6],v[-1:])
          
    @classmethod
    def fromBoxJson(cls,bj):
        bbox = [bj["bbox"]["min"],bj["bbox"]["max"]]
        b = cls.fromFlat([(bbox[0][0]+bbox[1][0])/2.0,(bbox[0][1]+bbox[1][1])/2.0,(bbox[0][2]+bbox[1][2])/2.0,1,1,1,bj["orient"]])
        b.size = np.abs(b.matrix(-1) @ np.array([(bbox[1][0]-bbox[0][0])/2.0,(bbox[1][1]-bbox[0][1])/2.0,(bbox[1][2]-bbox[0][2])/2.0]))
        return b
        #endregion: inputs-------#

        #region: presentation----#
    def __str__(self):
        return "tr:[%.3f,%.3f,%.3f], sz:[%.3f,%.3f,%.3f], oi:[%.3f],[%.3f,0.000,%.3f]"%(self.translation[0],self.translation[1],self.translation[2],self.size[0],self.size[1],self.size[2],self.orientation[0],np.sin(self.orientation[0]),np.cos(self.orientation[0]))

    def draw(self,g=False,d=False,color="",alpha=1.0,cr="",text=False,marker="."):
        from matplotlib import pyplot as plt
        corners = self.corners2()
        if g:
            plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker=marker, color=grupC[self.gid])
        else:
            if len(color):
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker=marker, color=color, alpha=alpha)
            else:
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker=marker)
        if d:
            if len(cr):
                plt.plot([self.translation[0], self.translation[0]+0.5*self.direction()[0]], [-self.translation[2],-self.translation[2]-0.5*self.direction()[2]], marker="x", color=cr)
            else:
                plt.plot([self.translation[0], self.translation[0]+0.5*self.direction()[0]], [-self.translation[2],-self.translation[2]-0.5*self.direction()[2]], marker="x")
        if text:
            plt.text(self.translation[0]-0.15, -self.translation[2]-0.15, text[:min(len(text),10)],fontdict={"fontsize":8})

    def drao(self,way,colors):
        if way in ["pnt","pns"] :
            self.draw(d=False,color=(0.6,0.6,0.6),text=False)
            self.samples.draw(way,colors)
        else:
            if way=="pat":
                self.draw(d=True,cr=(0.0,0.0,0.0),text=True)
                return
            self.draw(d=True,color=colors["res"],cr=(0.0,0.0,0.0),text=False)
            if way == "syn":
                from matplotlib import pyplot as plt
                a = bx2d(b=self)
                a.size -= self.adjust["S"]
                a.draw(color=colors["s"])
                a.orientation -= self.adjust["R"]
                plt.plot( [self.translation[0],self.translation[0]+a.direction()[0]*0.3], [-self.translation[2],-self.translation[2]-a.direction()[2]*0.3], marker=".", color=colors["r"])
                plt.plot( [self.translation[0],self.translation[0]-self.adjust["T"][0]], [-self.translation[2],-self.translation[2]+self.adjust["T"][2]], marker=".", color=colors["t"])
                
    def toBoxJson(self):
        bb = self.bbox3()
        bj = {"bbox":{"min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],"max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]},
            "translate":[float(self.translation[0]),float(self.translation[1]),float(self.translation[2])], "scale":[1,1,1],
            "rotate":[0,float(self.orientation[0]),0], "rotateOrder": "XYZ", "orient":float(self.orientation[0])}
        return bj
    
    def bpt(self):
        return np.concatenate([self.translation,self.size,self.orientation])
        #endregion: presentation-#
    
    #endregion: in/outputs-------#

    #region: properties----------#
    def direction(self):
        return np.array([np.math.sin(self.orientation),0,np.math.cos(self.orientation)])

    def matrix(self,u=1): #u=-1: transform others into my co-ordinates; u=1 or unset:transform mine into the world's co-ordinate
        return np.array([[np.math.cos(self.orientation),0,np.math.sin(self.orientation*u)],[0,1,0],[-np.math.sin(self.orientation*u),0,np.math.cos(self.orientation)]])

    def corners3(self):
        seeds = np.array([[1,0,1],[1,0,-1],[-1,0,-1],[-1,0,1]])
        return  self.translation + (self.matrix() @ (seeds * self.size.reshape(1,-1)).T).T

    def corners2(self):
        c = self.corners3()
        return np.concatenate([c[:,:1],c[:,-1:]],axis=-1)

    def shape(self):
        from shapely.geometry import Polygon
        return Polygon(self.corners2()).convex_hull

    def bbox3(self):
        cs = self.corners2()
        return np.array([[cs[:,0].min(),self.translation[1]-self.size[1],cs[:,1].min()],[cs[:,0].max(),self.translation[1]+self.size[1],cs[:,1].max()]])
    
    @classmethod
    def mat(cls,ori,size):
        return np.array([[np.math.cos(ori),0,np.math.sin(ori)],[0,1,0],[-np.math.sin(ori),0,np.math.cos(ori)]]) * size[None,:] # Sep.1st: from [:,None] to [None,:] test! test! test!

    def sampls(self, s = np.array([[1,0,0],[1,0,1],[0,0,1],[-1,0,1],[-1,0,0],[-1,0,-1],[0,0,-1],[1,0,-1]]).reshape((-1,1,3))):
        return self.translation+(self.matrix().reshape((1,3,3))*s*self.size.reshape((1,1,3))).sum(axis=-1)

    def samplesBound(self):
        return [self.sampls().min(axis=0), self.sampls().max(axis=0)]
    #endregion: properties-------#
    
    #region: operations----------#

        #region: movements-------#
    def setTransformation(self,t,o):
        self.translation,self.orientation = t,o

    def distance(self, b):#.reshape((1,3))
        vs = (b.samples()-self.translation).reshape((-1,1,3))
        ds = (self.matrix(-1).reshape((1,3,3))*vs).sum(axis=-1)
        ns = ds / self.size.reshape((1,3))
        return (ds**2).sum(-1)**0.5, np.abs(ns).min() < 1
        #endregion: movements----#

        #region: relatives-------#
    def __add__(self,b):
        try:
            b = copy(b)
            b.translation = self.translation + self.matrix()@(b.translation * (self.size if SCL else np.array([1,1,1])) )
            b.size = b.size * np.linalg.norm(bx2d.mat(-b.orientation,self.size), axis=1) if SCL else b.size
            b.orientation = angleNorm(b.orientation + self.orientation)
            return b
        except:
            return self.translation+(self.matrix()@(b*self.size))#.sum(axis=-1)

    def __sub__(self,b):
        try:
            b = copy(b)
            b.orientation = angleNorm(b.orientation - self.orientation)
            b.size = b.size / np.linalg.norm(bx2d.mat(b.orientation,self.size), axis=1) if SCL else b.size
            b.translation = self.matrix(-1)@(b.translation-self.translation) / (self.size if SCL else np.array([1,1,1]))
            return b
        except:
            return (self.matrix(-1)@(b-self.translation))/self.size
        #endregion: relatives----#

        #region: samples---------#
    def optimizePhy(self,config,debug=False,ut=-1):
        from ..Semantic.Samp import samps
        self.samples = samps(self,config["ss"],debug)
        return self.samples(config,ut)
        #endregion: samples------#

    #endregion: operations-------#
    
class obje(bx2d):
    def __init__(self,t=np.array([0,0,0]),s=np.array([1,1,1]),o=np.array([0]),c=None,i=None,n=None,idx=None,modelId=None,gid=0,scne=None,v=True,b=None):
        super(obje,self).__init__(t,s,o) if b is None else super(obje,self).__init__(b=b)
        self.class_index = object_types.index(n) if n is not None else (i if i is not None else np.argmax(c))
        self.idx, self.v,  self.modelId,  self.scne     =idx,v, modelId,scne#pointer(scne)
        self.gid, self.nid,self.linkIndex,self.destIndex=gid,-1,[],[]

    #region: in/outputs----------#

        #region: inputs----------#
    @classmethod
    def empty(cls,j=0,v=True):
        return cls(i=j,v=v)

    @classmethod
    def fromFlat(cls,v,j):
        return cls(b=bx2d.fromFlat(v),i=j)
 
    def fromTensor(self,tensor,fmt):
        import torch
        i = 0
        for f in format:
            if f[0]=="t":
                self.translation[0] = float(tensor[i])
                if f[1]==3:
                    self.translation[1] = float(tensor[i+1])
                    self.translation[2] = float(tensor[i+2])
                else:
                    self.translation[2] = float(tensor[i+1])
            elif f[0]=="s":
                self.size[0] = float(tensor[i])
                if f[1]==3:
                    self.size[1] = float(tensor[i+1])
                    self.size[2] = float(tensor[i+2])
                else:
                    self.size[2] = float(tensor[i+1])
            elif f[0]=="o":
                self.orientation[0] = tensor[i] if f[1]==1 else torch.atan(tensor[i+1],tensor[i])
            elif f[0]=="c":
                pass
            i += f[1]
        pass

    @classmethod
    def fromObjectJson(cls,oj,idx,scne=None):
        return cls(b=bx2d.fromBoxJson(oj),n=oj["coarseSemantic"],idx=idx,modelId=oj["modelId"],scne=scne)
        #endregion: inputs-------#

        #region: presentation----#
    def __str__(self):
        return "%d %s\t"%(self.idx, self.class_name()[:10]) + (super(obje,self).__str__()) + ("" if self.nid<0 else "\tgid=%d,nid=%d"%(self.gid,self.nid))

    def toObjectJson(self, rid=0):
        return {**(super(obje,self).toBoxJson()), "id":self.scne.scene_uid if self.scne.scene_uid else ""+"_"+str(self.idx), "type":"Object", "modelId":self.modelId, "coarseSemantic":self.class_name(), "roomId":rid, "inDatabase":False}
    
    def draw(self,g=False,d=False,color="",alpha=1.0,cr="",text=True):
        return super(obje,self).draw(g,d,color,alpha,cr,("%d "%(self.nid) if self.nid>-1 else "")+self.class_name() if text else "")
 
    def renderable(self,objects_dataset,color_palette,no_texture=True,depth=0):
        from simple_3dviz import Mesh
        from simple_3dviz.renderables.textured_mesh import TexturedMesh
        furniture = objects_dataset.get_closest_furniture_to_box(self.class_name(), self.size)
        try:
            raw_mesh = Mesh.from_file(furniture.raw_model_path, color=color_palette[self.class_index, :]) if no_texture else TexturedMesh.from_file(furniture.raw_model_path)
        except:
            print(furniture.raw_model_path)
            assert 1==0
        raw_mesh.scale(furniture.scale)
        self.translation[1] = self.translation[1] if ("Lamp" in self.class_name()) else self.size[1] - depth
        raw_mesh.affine_transform(t=-(raw_mesh.bbox[0] + raw_mesh.bbox[1])/2)
        raw_mesh.affine_transform(R=self.matrix(-1), t=self.translation)
        return raw_mesh
        
    def flat(self,inter=True):
        return np.concatenate([self.translation,self.size,[0] if (inter and (self.class_name() in noOriType)) else self.orientation])#])#

    def toTensor(self,format):
        import torch
        res = torch.Tensor([])
        for f in format:
            if f[0]=="t":
                cur = torch.Tensor(self.translation) if f[1]==3 else torch.Tensor([self.translation[0],self.translation[2]]) 
            elif f[0]=="s":
                cur = torch.Tensor(self.size) if f[1]==3 else torch.Tensor([self.size[0],self.size[2]]) 
            elif f[0]=="o":
                cur = torch.Tensor(self.orientation) if f[1]==1 else torch.Tensor([torch.cos(self.orientation[0]),torch.sin(self.orientation[0])]) 
            elif f[0]=="c":
                cur = torch.cat([torch.Tensor(self.class_label())] + [[0]*f[1]] )
                if not self.v:
                    cur = torch.zeros_like(cur)
                    cur[-1] = 1
            res = torch.cat([res,cur])
        return res

    def bpt(self):
        return np.concatenate([self.class_label(),super(obje,self).bpt()]).reshape((1,-1))
        #endregion: presentation-#

    #endregion: in/outputs-------#

    #region: properties----------#    
    def project(self,wid):
        w = self.scne.WALLS[wid]
        corners = self.corners2()
        a=(corners-np.array([[w.p[0],w.p[2]]])) @ np.array([(w.q[0]-w.p[0])/w.length,(w.q[2]-w.p[2])/w.length])
        return np.min(a)/w.length,np.max(a)/w.length

    def class_name(self):
        return object_types[self.class_index]
    
    def class_label(self):
        return np.eye((len(object_types)))[self.class_index]
    #endregion: properties-------# 

    #region: operations----------#

        #region: movement--------#
    # def adjust(self,movement):
    #     self.translation+=movement
    #     for i in self.linkIndex:
    #         self.scne.LINKS[i].adjust(movement)
        
    #     if len(self.destIndex) > 1:
    #         for i in self.destIndex:
    #             self.scne.LINKS[i].update(self.translation)
        #endregion: movement-----#

        #region: optField--------#
    def optimizePhy(self,config,timer,debug=False,ut=-1):
        from ..Semantic.Samp import samps
        self.samples = samps(self,config["s4"],debug)
        return self.samples(config,timer,ut)
    
    def violate(self):
        return self.samples.violate()

    def optField(self, sp, c):
        newSelf = bx2d(t=self.translation + self.matrix()@(np.array([0,0,c[0]])*self.size), s=np.array([c[2],1,c[1]])*self.size, o=self.orientation)
        try: #for object samples
            X0Z, A0C = newSelf-sp.transl, newSelf.matrix(-1)@(-sp.radial) / newSelf.size
            X0Z[1],A0C[1] = 0,0
            if norm(X0Z) > 1.0:
                return np.array([.0,.0,.0])
            #[F,0,H]= { √(A²+C²-(AZ-CX)²)-AX-CZ }/{ A²+C² }  [A,0,C]
            F0H = (np.sqrt(norm(A0C)**2 - (np.cross(A0C,X0Z)[1])**2) - A0C@X0Z)/(norm(A0C)**2) * A0C
            return (newSelf + F0H) - newSelf.translation#transform this field back to the world
        except:
            X0Z = newSelf-sp.transl
            X0Z[1] = 0
            n = norm(X0Z)
            v = (newSelf + (X0Z if norm(X0Z)<0.000001 else X0Z*(max(1-n,0.0)/n))) - newSelf.translation
            return v, (norm(v)**2)/2.0
        #endregion: optField-----#

    #endregion: operations-------#

class objes():
    def __init__(self,scene,ce,scne=None):
        tr,si,oi,cl = scene["translations"],scene["sizes"],scene["angles"],scene["class_labels"]
        self.OBJES=[obje(tr[i]+ce,si[i],oi[i],cl[i],idx=i,scne=scne) for i in range(len(tr))]
        self.scne=scne
    
    #region: in/outputs----------#

        #region: inputs----------#
    @classmethod
    def empty(cls,ce=np.array([0,0,0]),scne=None):
        return cls({"translations":[],"sizes":[],"angles":[],"class_labels":[],"scene_uid":""},ce,scne=scne)
    
    def fromTensor(self,tensor,fmt):
        import torch
        assert (tensor[:len(self),-1]).sum() == 0 and (torch.ones_like(tensor[len(self):,-1])-tensor[len(self):,-1]).sum() == 0
        [o.fromTensor(tensor[i],fmt) for i,o in enumerate(self)]

    @classmethod
    def fromSceneJson(cls,rsj,scene):
        objects = cls.empty(scne=scene)
        [objects.addObject(obje.fromObjectJson(rsj["objList"][oi],oi,scene)) for oi in range(len(rsj["objList"]))]
        #for oj in []:#rsj["blockList"]:
        #    objects.addObject(obje.fromObjectJson(oj))
        return objects
        #endregion: inputs-------#

        #region: presentation----#
    def __str__(self):
        return '\n'.join([str(o) for o in self.OBJES])
 
    def draw(self,grp,drawUngroups,d,classText):
        [self[i].draw(grp,d,text=classText) if (not grp) or drawUngroups or (self[i].gid) else None for i in range(len(self.OBJES))]

    def drao(self,way,colors):
        [o.drao(way,colors) for o in self]
      
    def renderables(self,scene_render,objects_dataset,no_texture,depth):
        import seaborn
        [scene_render.add(o.renderable(objects_dataset, np.array(seaborn.color_palette('hls', len(object_types)-2)), no_texture,depth)) for o in [_ for _ in self.OBJES if _.v] ]
      
    def exportAsSampleParams(self,c):
        c["translations"] = np.array([o.translation for o in self.OBJES if (o.gid >= 1 or (not self.grp))])
        c["sizes"] = np.array([o.size for o in self.OBJES])
        c["angles"] = np.array([[np.cos(o.orientation),np.sin(o.orientation)] if c["angles"].shape[-1] == 2 else o.orientation for o in self.OBJES])
        return c
    
    def bpt(self):
        return np.concatenate([o.bpt() for o in self.OBJES],axis=0)
    
    def toTensor(self,fmt,length):
        import torch
        return torch.cat([o.toTensor(fmt) for o in self]+[obje.empty(v=False).tensor(format)]*(length-len(self)),axis=0).reshape((1,length,-1))

    def toSceneJson(self,rsj):
        for o in [_ for _ in self.OBJES if _.v]:
            rsj["objList"].append(o.toObjectJson())
            if o.class_name().lower() in ["window", "door"]:
                raise NotImplementedError
                rsj["blockList"].append(o.toObjectJson())
        return rsj
        #endregion: presentation-#

    #endregion: in/outputs-------#

    #region: properties----------#
    def __len__(self):
        return len(self.OBJES)
    
    def __getitem__(self,cl):
        return self.OBJES[cl] if type(cl) == int else [o for o in self.OBJES if o.class_name() == cl]

    def nids(self):
        return set([o.nid for o in self.OBJES])

    def __iter__(self):
        return iter(self.OBJES)
    #endregion: properties-------#

    #region: operations----------#

        #region: basic-----------#

    def addObject(self,objec):
        objec.idx,objec.scne = len(self.OBJES), self.scne
        self.OBJES.append(objec)
        return objec.idx
    
    def objectView(self,id,bd=100000,scl=False,maxDis=100000):
        newOBJES = [(self[id] - o) for o in self.OBJES if (o.idx != id and o.nid == -1)] # and not(o.class_name() in noPatternType)
        return sorted(newOBJES,key=lambda x:(x.translation**2).sum())[:min(len(newOBJES),bd)]
    
        #endregion: basic--------#
    
        #region: optFields-------#
    def optFields(self,sp,o,config):
        if o: #for o's samples
            return np.array([oo.optField(sp,config[oo.class_name()]) for oo in [_ for _ in self.OBJES if (_.idx != o.idx)]] ).sum(axis=0)
        else: #for field
            A = np.array([oo.optField(sp,config[oo.class_name()])[0] for oo in self.OBJES])
            return  A.sum(axis=0), np.array([(norm(a)**2)/2.0 for a in A] ).sum(axis=0)
        
    def optimizePhy(self,config,timer,debug=False,ut=-1):
        #print(ut)
        for o in self.OBJES:
            o.optimizePhy(config,timer,debug,ut)
        from ..Operation.Adjs import adjs
        #from ..Experiment.ExOp import EXOP_BASE_DIR
        #self.scne.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d+opt.png"%(self.scne.scene_uid[:10],1))#return #
        return adjs(self)#[o.optimizePhy(config,timer,debug,ut) for o in self.OBJES]
    
    def violates(self):
        return sum([o.violate() for o in self])/float(len(self))
        #endregion: optFields----#

    #endregion: operations-------#
