import numpy as np
from shapely.geometry import Polygon
grupC=["black","red","gray","purple","yellow","red","gray","purple","yellow"]
object_types = ["Pendant Lamp", "Ceiling Lamp", "Bookcase / jewelry Armoire", \
"Round End Table", "Dining Table", "Sideboard / Side Cabinet / Console table", "Corner/Side Table", "Desk", "Coffee Table", "Dressing Table", \
"Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Wine Cabinet", \
"Lounge Chair / Cafe Chair / Office Chair", "Classic Chinese Chair", "Dressing Chair", "Dining Chair", "armchair", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", \
"Three-seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "Lazy Sofa", "Chaise Longue Sofa", "Wardrobe", "TV Stand", "Nightstand", \
"King-size Bed", "Kids Bed", "Bunk Bed", "Single bed", "Bed Frame", "window", "door"]

noOriType = ["Pendant Lamp", "Ceiling Lamp", "Round End Table", "Corner/Side Table", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", "Nightstand"]
        
def angleNorm(ori):
    ori = ori % (2*np.math.pi)
    return ori-2*np.math.pi if 2*np.math.pi-ori < ori else ori


from matplotlib import pyplot as plt
#OBJES=[]
class obje():
    def __init__(self,t=np.array([0,0,0]),s=np.array([1,1,1]),o=np.array([0]),c=None,i=None,n=None,idx=None,modelId=None,gid=0,scne=None,v=True):
        self.translation = t
        self.size = s
        self.orientation = o if len(o)==1 or o.shape == (1,) else np.math.atan2(o[1],o[0])
        self.idx=idx
        self.v=v
        if i is None and n is None:
            self.class_label = c #print(self.class_label,len(object_types))
            self.class_index = np.argmax(c) #print(self.class_index)
        elif c is None and n is None:
            self.class_index = i
            self.class_label = np.zeros((len(object_types)))
            self.class_label[i] = 1
        else:
            self.class_index = object_types.index(n)
            self.class_label = np.zeros((len(object_types)))
            self.class_label[self.class_index] = 1
        self.modelId = modelId
        
        self.linkIndex=[]
        self.destIndex=[]
        self.gid=gid
        self.nid=-1
        self.scne=scne#pointer(scne)

    @classmethod
    def fromFlat(cls,v,j):
        return cls(v[:3],v[3:6],v[-1:],i=j)

    def direction(self):
        return np.array([np.math.sin(self.orientation),0,np.math.cos(self.orientation)])

    def matrix(self,u=1): #u=-1: transform others into my co-ordinates; u=1 or unset:transform mine into the world's co-ordinate
        return np.array([[np.math.cos(self.orientation),0,np.math.sin(self.orientation*u)],[0,1,0],[-np.math.sin(self.orientation*u),0,np.math.cos(self.orientation)]])

    def corners2(self):
        CenterZ = np.repeat([self.translation[2]],4)#,t[2]+ce[2],t[2]+ce[2],t[2]+ce[2],t[2]+ce[2]]
        CenterX = np.repeat([self.translation[0]],4)#[t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0]]
        CornerOriginalZ = np.array([self.size[2], self.size[2],-self.size[2],-self.size[2]])
        CornerOriginalX = np.array([self.size[0],-self.size[0],-self.size[0], self.size[0]])

        z2z,z2x = np.repeat(np.cos(self.orientation),4),np.repeat(np.sin(self.orientation),4)#[np.cos(a[0]), np.cos(a[0]), np.cos(a[0]), np.cos(a[0]), np.cos(a[0])]
        x2z,x2x =-z2x,z2z#[-np.sin(a[0]),-np.sin(a[0]),-np.sin(a[0]),-np.sin(a[0]),-np.sin(a[0])]
        
        realZ = CenterZ + CornerOriginalZ*z2z + CornerOriginalX*x2z
        realX = CenterX + CornerOriginalZ*z2x + CornerOriginalX*x2x
        return np.array([[realX[i],realZ[i]] for i in range(4)])
    
    def shape(self):
        return Polygon(self.corners2()).convex_hull

    def draw(self,g=False,d=False,color="",alpha=1.0,cr="",text=False):
        corners = self.corners2()
        if g:
            plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*", color=grupC[self.gid])
        else:
            if len(color):
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*", color=color, alpha=alpha)
            else:
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*")
        if d:
            if len(cr):
                plt.plot([self.translation[0], self.translation[0]+0.5*self.direction()[0]], [-self.translation[2],-self.translation[2]-0.5*self.direction()[2]], marker="x", color=cr)
            else:
                plt.plot([self.translation[0], self.translation[0]+0.5*self.direction()[0]], [-self.translation[2],-self.translation[2]-0.5*self.direction()[2]], marker="x")
        if text:
            plt.text(self.translation[0]-0.15, -self.translation[2]-0.15, ("%d "%(self.nid) if self.nid>-1 else "")+self.class_name()[:min(len(self.class_name()),10)],fontdict={"fontsize":8})


    def bbox3(self):
        cs = self.corners2()
        return np.array([[cs[:,0].min(),self.translation[1]-self.size[1],cs[:,1].min()],[cs[:,0].max(),self.translation[1]+self.size[1],cs[:,1].max()]])

    def toObjectJson(self, rid=0):
        if self.modelId is None:
            uid = self.scne.scene_uid if self.scne.scene_uid else ""
            bb = self.bbox3()
            oj = {"id":uid+"_"+str(self.idx), "type":"Object", "modelId":self.modelId,
                "bbox":{"min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],"max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]},
                "translate":[float(self.translation[0]),float(self.translation[1]),float(self.translation[2])],
                "scale":[1,1,1], "rotate":[0,float(self.orientation[0]),0], "rotateOrder": "XYZ",
                "orient":float(self.orientation[0]), "coarseSemantic":self.class_name(), "roomId":rid, "inDatabase":False}
        else: 
            raise NotImplementedError
        return oj

    def fromBbox(self,bbox):
        self.translation = np.array([(bbox[0][0]+bbox[1][0])/2.0,(bbox[0][1]+bbox[1][1])/2.0,(bbox[0][2]+bbox[1][2])/2.0])
        corner = np.array([(bbox[1][0]-bbox[0][0])/2.0,0,(bbox[1][2]-bbox[0][2])/2.0])
        cnr = self.matrix(-1) @ corner
        self.size = np.array([abs(cnr[0]),(bbox[1][1]-bbox[0][1])/2.0,abs(cnr[2])])
        
    @classmethod
    def fromObjectJson(cls,oj,idx):
        o = cls.fromFlat([0,0,0,0,0,0,0],0)
        o.idx = idx
        #print(oj)
        #print('\n') 
        o.orientation = np.array([oj["orient"]]) # is it different?
        o.fromBbox(np.array([oj["bbox"]["min"],oj["bbox"]["max"]]))

        o.class_index = object_types.index(oj["coarseSemantic"])
        o.class_label = np.zeros((len(object_types)))
        o.class_label[o.class_index] = 1
        o.modelId = oj["modelId"]
        
        return o

    def project(self,wid):
        w = self.scne.WALLS[wid]
        corners = self.corners2()
        a=(corners-np.array([[w.p[0],w.p[2]]])) @ np.array([(w.q[0]-w.p[0])/w.length,(w.q[2]-w.p[2])/w.length])
        
        return np.min(a)/w.length,np.max(a)/w.length

    def class_name(self):
        return object_types[self.class_index]
        
    def adjust(self,movement):
        self.translation+=movement
        for i in self.linkIndex:
            self.scne.LINKS[i].adjust(movement)
        
        if len(self.destIndex) > 1:
            for i in self.destIndex:
                self.scne.LINKS[i].update(self.translation)

    def setTransformation(self,t,o):
        self.translation,self.orientation = t,o

    @classmethod
    def mat(cls,ori,size):
        return np.array([[np.math.cos(ori),0,np.math.sin(ori)],[0,1,0],[-np.math.sin(ori),0,np.math.cos(ori)]]) * size[None,:] # Sep.1st: from [:,None] to [None,:] test! test! test!

    def samples(self):#.reshape((1,3))
        s = np.array([[1,0,0],[1,0,1],[0,0,1],[-1,0,1],[-1,0,0],[-1,0,-1],[0,0,-1],[1,0,-1]]).reshape((-1,1,3))
        return self.translation+(self.matrix().reshape((1,3,3))*s*self.size.reshape((1,1,3))).sum(axis=-1)

    def samplesBound(self):
        sam = self.samples()
        #print(sam)
        return [sam.min(axis=0), sam.max(axis=0)]

    def distance(self, o):#.reshape((1,3))
        vs = (o.samples()-self.translation).reshape((-1,1,3))
        ds = (self.matrix(-1).reshape((1,3,3))*vs).sum(axis=-1)
        ns = ds / self.size.reshape((1,3))
        return (ds**2).sum(-1)**0.5, np.abs(ns).min() < 1

    def rely(self, o, scl=True):
        t = self.translation + self.matrix()@(o.translation * (self.size if scl else np.array([1,1,1])) )
        ori = angleNorm(o.orientation + self.orientation)#- (0.0 if (o.orientation - self.orientation) % (2*np.math.pi) < np.math.pi else 2*np.math.pi)
        s = o.size * np.linalg.norm(obje.mat(-o.orientation,self.size), axis=1) if scl else o.size
        return obje(t,s,ori,i=o.class_index,idx=o.idx,scne=o.scne)

    def rela(self, o, scl=True):
        t = self.matrix(-1)@(o.translation-self.translation) / (self.size if scl else np.array([1,1,1]))
        ori = angleNorm(o.orientation - self.orientation)#- (0.0 if (o.orientation - self.orientation) % (2*np.math.pi) < np.math.pi else 2*np.math.pi)
        s = o.size / np.linalg.norm(obje.mat(ori,self.size), axis=1) if scl else o.size
        return obje(t,s,ori,i=o.class_index,idx=o.idx,scne=o.scne)

    def flat(self,inter=True):
        return np.concatenate([self.translation,self.size,[0] if (inter and (self.class_name() in noOriType)) else self.orientation])#])#

    def bpt(self):
        return np.concatenate([self.class_label,self.translation,self.size,self.orientation]).reshape((1,-1))

class objes():
    def __init__(self,scene,ce,windoor,scne=None):
        
        tr,si,oi,cl = scene["translations"],scene["sizes"],scene["angles"],scene["class_labels"]
        self.OBJES=[obje(tr[i]+ce,si[i],oi[i],np.concatenate([cl[i],[0,0]])if windoor else cl[i],idx=i,scne=scne) for i in range(len(tr))]
        self.scne=scne

    def __len__(self):
        return len(self.OBJES)

    def __str__(self):
        return '\n'.join([str(o) for o in self.OBJES])
    
    def draw(self):
        pass

    def addObject(self,objec):
        objec.idx = len(self.OBJES)
        objec.scne = self.scne
        self.OBJES.append(objec)

    def nids(self):
        return set([o.nid for o in self.OBJES])

    def searchNid(self, nid, sig=True):
        s = [o for o in self.OBJES if o.nid == nid]
        return (s[0] if sig else s) if len(s)>0 else None

    def __iter__(self):
        return iter(self.OBJES)

# sc = {"translations":np.array([[0,0,0]]),"sizes":np.array([[0,0,0]]),"angles":np.array([[0]]),"class_labels":np.array([[1,0,0]])}
# a = objes(sc,np.array([0,0,0]),False)
# for A in a:
#     print(A.class_name())