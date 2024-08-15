import numpy as np
grupC=["black","red","gray"]
object_types = ["Pendant Lamp", "Ceiling Lamp", "Bookcase / jewelry Armoire", \
"Round End Table", "Dining Table", "Sideboard / Side Cabinet / Console table", "Corner/Side Table", "Desk", "Coffee Table", "Dressing Table", \
"Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Wine Cabinet", \
"Lounge Chair / Cafe Chair / Office Chair", "Classic Chinese Chair", "Dressing Chair", "Dining Chair", "armchair", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", \
"Three-seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "Lazy Sofa", "Chaise Longue Sofa", "Wardrobe", "TV Stand", "Nightstand", \
"King-size Bed", "Kids Bed", "Bunk Bed", "Single bed", "Bed Frame", "window", "door"]

noOriType = ["Pendant Lamp", "Ceiling Lamp", "Round End Table", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", "Nightstand"]
        

from matplotlib import pyplot as plt
#OBJES=[]
class obje():
    def __init__(self,t,s,o,c=None,i=None,idx=None,gid=0,scne=None,v=True):
        self.translation = t
        self.size = s
        self.orientation = o if o.shape == (1,) else np.math.atan2(o[1],o[0])
        self.idx=idx
        self.v=v
        if i is None:
            self.class_label = c #print(self.class_label,len(object_types))
            self.class_index = np.argmax(c) #print(self.class_index)
        elif c is None:
            self.class_index = i
            self.class_label = np.zeros((len(object_types)))
            self.class_label[i] = 1
        
        self.linkIndex=[]
        self.destIndex=[]
        self.gid=gid
        self.nid=-1
        self.scne=scne#pointer(scne)

    def direction(self):
        return np.array([np.math.sin(self.orientation),0,np.math.cos(self.orientation)])

    def matrix(self,u=1):
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

    def draw(self,g=False,d=False,color="",alpha=1.0):
        corners = self.corners2()
        if g:
            plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*", color=grupC[self.gid])
        else:
            if len(color):
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*", color=color, alpha=alpha)
            else:
                plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*")
        if d:
            plt.plot([self.translation[0], self.translation[0]+0.5*self.direction()[0]], [-self.translation[2],-self.translation[2]-0.5*self.direction()[2]], marker="x")

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

    def rely(self, o, scl=False):
        t = self.matrix()@(o.translation + self.translation) / (self.size if scl else np.array([1,1,1]))
        ori = o.orientation + self.orientation
        while ori <-np.math.pi:
            ori += 2*np.math.pi
        while ori > np.math.pi:
            ori -= 2*np.math.pi
        s = o.size * (self.size if scl else np.array([1,1,1]))
        return obje(t,s,ori,i=o.class_index,idx=o.idx,scne=o.scne)

    def rela(self, o, scl=False):
        t = self.matrix(-1)@(o.translation - self.translation) / (self.size if scl else np.array([1,1,1]))
        ori = o.orientation - self.orientation
        while ori <-np.math.pi:
            ori += 2*np.math.pi
        while ori > np.math.pi:
            ori -= 2*np.math.pi
        s = o.size / (self.size if scl else np.array([1,1,1]))
        return obje(t,s,ori,i=o.class_index,idx=o.idx,scne=o.scne)

    def flat(self,inter=True):
        return np.concatenate([self.translation,self.size,[0] if (inter and (self.class_name() in noOriType)) else self.orientation])#])#

    def bpt(self):
        return np.concatenate([self.class_label,self.translation,self.size,self.orientation]).reshape((1,-1))