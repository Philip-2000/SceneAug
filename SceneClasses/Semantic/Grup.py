grupA=["white","lightblue","lightgreen","pink","lime","lightblue","lightgreen","pink","lime"]
import numpy as np

def fTheta(theta):
    while theta > np.math.pi:
        theta -= 2*np.math.pi
    while theta < -np.math.pi:
        theta += 2*np.math.pi
    return theta

def matrix(ori):
    return np.array([[np.math.cos(ori),0,np.math.sin(ori)],[0,1,0],[-np.math.sin(ori),0,np.math.cos(ori)]])
    
#WALLS=[]
class grup():
    def __init__(self, objIdList, imgMeta, idx=0, scne=None):
        self.objIdList = [i for i in objIdList]
        self.scne=scne
        for i in objIdList:#assert (self.scne.OBJES[i].gid == idx)
            self.scne.OBJES[i].gid = idx
        assert len(objIdList) > 0
        cs = np.array([self.scne.OBJES[i].corners2() for i in objIdList]).reshape((-1,2))
        Xs,Zs = cs[:,0],cs[:,1]
        self.translation = np.array([(np.min(Xs)+np.max(Xs))/2.0,0,(np.min(Zs)+np.max(Zs))/2.0])
        self.orientation = 0.0
        self.size = np.array([np.max(Xs),1.0,np.max(Zs)])-self.translation
        self.scale = np.array([1,1,1])
        self.idx = idx
        self.isz=imgMeta["sz"]>>1
        self.irt=imgMeta["rt"]

    def update(self):
        self.objIdList = []
        for o in self.scne.OBJES:
            if o.gid == self.idx:
                self.objIdList.append(o.idx)
        if len(self.objIdList) == 0:
            return
        cs = np.array([self.scne.OBJES[i].corners2() for i in self.objIdList]).reshape((-1,2))
        Xs,Zs = cs[:,0],cs[:,1]
        self.translation = np.array([(np.min(Xs)+np.max(Xs))/2.0,0,(np.min(Zs)+np.max(Zs))/2.0])
        self.orientation = 0.0
        self.size = np.array([np.max(Xs),1.0,np.max(Zs)])-self.translation
        self.scale = np.array([1,1,1])
        pass

    def bbox2(self):
        cs = np.array([self.scne.OBJES[i].corners2() for i in self.objIdList]).reshape((-1,2))
        return [np.min(cs,axis=0), np.max(cs,axis=0)]
    
    def shape(self):
        from shapely.geometry import Polygon
        bbox = self.bbox2()
        return Polygon(np.array([[bbox[0][0],bbox[0][1]],[bbox[0][0],bbox[1][1]],[bbox[1][0],bbox[1][1]],[bbox[1][0],bbox[0][1]]])).convex_hull

    def adjust(self, t, s, o):
        rTrans,t[1],s[1] = {}, 0.0, 1.0
        for i in self.objIdList:
            rTrans[i] = [matrix(-self.orientation)@(self.scne.OBJES[i].translation-self.translation)/self.scale,fTheta(self.scne.OBJES[i].orientation-self.orientation)]
        self.translation,self.scale, self.orientation=t,s,o
        for i in self.objIdList:
            self.scne.OBJES[i].setTransformation(matrix(o)@(rTrans[i][0]*s)+t,fTheta(rTrans[i][1]+o))
        cs = np.array([self.scne.OBJES[i].corners2() for i in self.objIdList]).reshape((-1,2))
        self.size = np.array([np.max(cs[:,0]),1.0,np.max(cs[:,1])])-self.translation
        self.update()

    def draw(self):
        from matplotlib import pyplot as plt
        scl = [1.0,0.7,0.4,0.1]
        c,a = self.translation, matrix(self.orientation)@(self.scale*self.size),
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color=grupA[self.idx])

    #img ->real: [(i-self.imgMeta["sz"])/self.imgMeta["rt"],0.0,(j-self.imgMeta["sz"])/self.imgMeta["rt"]]
    #real-> img: [int(i*self.imgMeta["rt"]+self.imgMeta["sz"]),int(j*self.imgMeta["rt"]+self.imgMeta["sz"])]        
    def imgSpaceCe(self):
        return (int(self.translation[0]*self.irt+self.isz),int(self.translation[2]*self.irt+self.isz))

    def imgSpaceBbox(self):
        return [int((self.translation[0]-self.size[0])*self.irt+self.isz),int((self.translation[2]-self.size[2])*self.irt+self.isz),int((self.translation[0]+self.size[0])*self.irt+self.isz),int((self.translation[2]+self.size[2])*self.irt+self.isz)]

    def recommendedWalls(self):
        #we are going 
        pass