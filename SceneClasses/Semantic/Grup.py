import numpy as np

def fTheta(theta):
    while theta > np.math.pi: theta -= 2*np.math.pi
    while theta < -np.math.pi: theta += 2*np.math.pi
    return theta

def matrix(ori):
    return np.array([[np.math.cos(ori),0,np.math.sin(ori)],[0,1,0],[-np.math.sin(ori),0,np.math.cos(ori)]])
    
grupA=["white","lightblue","lightgreen","pink","lime","lightblue","lightgreen","pink","lime"]
class grup():
    def __init__(self, objIdList, imgMeta=None, idx=0, scne=None):
        self.objIdList = [i for i in objIdList]
        self.scne=scne
        self.idx=idx
        if scne: self.update()
        # if scne: 
        #     for i in objIdList: self.scne[i].gid = idx
        # assert len(objIdList) > 0
        # cs = np.array([self.scne[i].corners2() for i in objIdList]).reshape((-1,2))
        # Xs,Zs = cs[:,0],cs[:,1]
        # self.translation = np.array([(np.min(Xs)+np.max(Xs))/2.0,0,(np.min(Zs)+np.max(Zs))/2.0])
        # self.orientation = 0.0
        # self.size = np.array([np.max(Xs),1.0,np.max(Zs)])-self.translation
        # self.scale = np.array([1,1,1])
        #self.isz=imgMeta["sz"]>>1
        #self.irt=imgMeta["rt"]

    def update(self):
        self.objIdList = []
        for o in self.scne.OBJES:
            if o.gid == self.idx:
                self.objIdList.append(o.idx)
        if len(self.objIdList) == 0:
            return
        cs = np.array([self.scne[i].corners2() for i in self.objIdList]).reshape((-1,2))
        Xs,Zs = cs[:,0],cs[:,1]
        self.translation = np.array([(np.min(Xs)+np.max(Xs))/2.0,0,(np.min(Zs)+np.max(Zs))/2.0])
        self.orientation = 0.0
        self.size = np.array([np.max(Xs),1.0,np.max(Zs)])-self.translation
        self.scale = np.array([1,1,1])

    def bbox2(self):
        cs = np.array([self.scne[i].corners2() for i in self.objIdList]).reshape((-1,2))
        return [np.min(cs,axis=0), np.max(cs,axis=0)]
    
    def shape(self):
        from shapely.geometry import Polygon
        bbox = self.bbox2()
        return Polygon(np.array([[bbox[0][0],bbox[0][1]],[bbox[0][0],bbox[1][1]],[bbox[1][0],bbox[1][1]],[bbox[1][0],bbox[0][1]]])).convex_hull

    def adjust(self, t, s, o):
        rTrans,t[1],s[1] = {}, 0.0, 1.0
        for i in self.objIdList:
            rTrans[i] = [matrix(-self.orientation)@(self.scne[i].translation-self.translation)/self.scale,fTheta(self.scne[i].orientation-self.orientation)]
        self.translation,self.scale, self.orientation=t,s,o
        for i in self.objIdList:
            self.scne[i].setTransformation(matrix(o)@(rTrans[i][0]*s)+t,fTheta(rTrans[i][1]+o))
        cs = np.array([self.scne[i].corners2() for i in self.objIdList]).reshape((-1,2))
        self.size = np.array([np.max(cs[:,0]),1.0,np.max(cs[:,1])])-self.translation
        self.update()

    def move(self, t=np.array([0.0,0.0,0.0]), s=np.array([0.0,0.0,0.0]), o=np.array([0.0])):
        self.adjust(self.translation+t, self.scale+s, self.orientation+o)
        # rTrans,t[1],s[1] = {}, 0.0, 1.0
        # for i in self.objIdList:
        #     rTrans[i] = [matrix(-self.orientation)@(self.scne[i].translation-self.translation)/self.scale,fTheta(self.scne[i].orientation-self.orientation)]
        # self.translation,self.scale, self.orientation=t,s,o
        # for i in self.objIdList:
        #     self.scne[i].setTransformation(matrix(o)@(rTrans[i][0]*s)+t,fTheta(rTrans[i][1]+o))
        # cs = np.array([self.scne[i].corners2() for i in self.objIdList]).reshape((-1,2))
        # self.size = np.array([np.max(cs[:,0]),1.0,np.max(cs[:,1])])-self.translation
        # self.update()

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

    def toGrupJson(self):
        return {"objIdList":self.objIdList}

class grups():
    def __init__(self, scene, irt=16):
        self.scne = scene
        self.isz, self.irt = scene.roomMask.shape[-1], irt
        self.GRUPS = []

    def __iter__(self):
        return iter(self.GRUPS)
    
    def __getitem__(self, idx):
        return self.GRUPS[idx]
    
    def __len__(self):
        return len(self.GRUPS)
    
    def append(self, group):
        group.idx = len(self.GRUPS)
        group.scne = self.scne
        group.update()
        group.isz, group.irt = self.isz, self.irt
        self.GRUPS.append(group)

    def draw(self):
        for g in self.GRUPS:
            g.draw()
    
    def clear(self):
        self.GRUPS = []

    def draftRoomMask(self):
        from PIL import Image, ImageDraw
        L = self.scne.roomMask.shape[-1]
        img = Image.new("L",(L,L))  
        img1 = ImageDraw.Draw(img)  
        for g in self.GRUPS: img1.rectangle(g.imgSpaceBbox(), fill ="white",outline="gray",width=2)
        img1.line([((L>>1,L>>1) if len(self.GRUPS)==1 else self.GRUPS[1].imgSpaceCe()),self.GRUPS[0].imgSpaceCe()],fill ="white",width=15)
        return img
    
    def toGrupsJson(self,rsj):
        rsj["grups"] = [g.objIdList for g in self.GRUPS]
        return rsj