import numpy as np
#from . import WALLS
#from . import LINKS
from . import grupC
object_types = ["Pendant Lamp", "Ceiling Lamp", "Bookcase / jewelry Armoire", \
"Round End Table", "Dining Table", "Sideboard / Side Cabinet / Console table", "Corner/Side Table", "Desk", "Coffee Table", "Dressing Table", \
"Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Wine Cabinet", \
"Lounge Chair / Cafe Chair / Office Chair", "Classic Chinese Chair", "Dressing Chair", "Dining Chair", "armchair", "Barstool", "Footstool / Sofastool / Bed End Stool / Stool", \
"Three-seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "Lazy Sofa", "Chaise Longue Sofa", "Wardrobe", "TV Stand", "Nightstand", \
"King-size Bed", "Kids Bed", "Bunk Bed", "Single bed", "Bed Frame", "window", "door"]

from matplotlib import pyplot as plt
#OBJES=[]
class obje():
    def __init__(self,t,s,o,c=None,i=None,idx=None,gid=-1,scne=None):
        self.translation = t
        self.size = s
        self.orientation = o
        self.idx=idx
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
        self.scne=scne#pointer(scne)

    def direction(self):
        return np.array([np.math.sin(self.orientation),0,np.math.cos(self.orientation)])

    def matrix(self):
        return np.array([[np.math.cos(self.orientation),0,np.math.sin(self.orientation)],[0,1,0],[-np.math.sin(self.orientation),0,np.math.cos(self.orientation)]])

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

    def draw(self,g=False):
        corners = self.corners2()
        if g:
            plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*", color="black" if self.gid == -1 else grupC[self.gid])
        else:
            plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-self.class_index>2 else "*")

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