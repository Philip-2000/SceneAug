from .Obje import *
from .Link import *
from .Wall import *
from .Grup import *
from .Spce import *
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from PIL import Image, ImageDraw

def two23(a):
    return np.array([a[0],0,a[1]])

#WALLS=[]
class scne():
    def __init__(self, scene, grp=False, windoor=False, wl=False, cen=False, rmm=True, irt=16, imgDir="./"):
        self.LINKS=[]
        self.SPCES=[]
        self.copy = copy(scene)
        self.scene_uid = str(scene["scene_uid"]) if "scene_uid" in scene else ""
        tr,si,oi,cl = scene["translations"],scene["sizes"],scene["angles"],scene["class_labels"]
        #firstly, store those objects and walls into the WALLS and OBJES
        ce = scene["floor_plan_centroid"] if cen else np.array([0,0,0])
        c_e= np.array([0,0,0]) if (not cen) or (not (wl or windoor)) else scene["floor_plan_centroid"]
        self.grp = grp
        self.imgDir = imgDir

        self.OBJES=[obje(tr[i]+ce,si[i],oi[i],np.concatenate([cl[i],[0,0]])if windoor else cl[i],idx=i,scne=self) for i in range(len(tr))]

        self.roomMask = scene["room_layout"][0]
        self.rmm=rmm
        self.GRUPS=[]
        if grp:
            grops = np.ones(tr.shape[0]) if scene["grops"] is None else scene["grops"]
            self.GRUPS = [grup([o.idx for o in self.OBJES if grops[o.idx]==j+1],{"sz":self.roomMask.shape[-1],"rt":irt},j+1,scne=self) for j in range(int(max(grops)))]
        
        if windoor:
            widos = scene["widos"]
            for k in range(len(widos)):
                oii = np.math.atan2(widos[k][-1],widos[k][-2])
                sii = np.array([max(widos[k][3],widos[k][5]),widos[k][4],min(widos[k][3],widos[k][5])]) #?
                tri = widos[k][:3]
                c = len(object_types)-1 if tri[1]-sii[1] < 0.1 else len(object_types)-2
                self.OBJES.append(obje(tri-c_e,sii,oii,i=c,idx=len(self.OBJES),scne=self))

        #obje(t,s,o,c,i)
        #wall(p,q,n,w1,w2)
        self.WALLS=[]
        if wl:
            walls = scene["walls"]
            self.WALLS = [wall(two23(walls[j][:2])-c_e,two23(walls[(j+1)%len(walls)][:2])-c_e,np.array([walls[j][3],0,walls[j][2]]),(j-1)%len(walls),(j+1)%len(walls),j,scne=self) for j in range(len(walls))]

    def draw(self,imageTitle="",lim=-1,drawWall=True,drawUngroups=False,drawRoomMask=False):
        for i in range(len(self.SPCES)):
            self.SPCES[i].draw()

        if self.grp:
            for i in range(len(self.GRUPS)):
                self.GRUPS[i].draw()

        for i in range(len(self.OBJES)):
            if (not self.grp) or drawUngroups or (self.OBJES[i].gid):
                self.OBJES[i].draw(self.grp)#corners = OBJES[i].corners2()
    
        if drawWall:
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            contour,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]], self.WALLS[J].w2
            while w != 0:
                contour.append([self.WALLS[w].p[0],self.WALLS[w].p[2]])
                w = self.WALLS[w].w2
            contour = np.array(contour)
            plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o")
        plt.axis('equal')

        for li in self.LINKS:
            src,dst = li.arrow()
            plt.plot([dst[0]], [-dst[2]], marker="x")
            plt.plot([src[0], dst[0]], [-src[2], -dst[2]], marker=".")

        if lim > 0:
            plt.xlim(-lim,lim)
            plt.ylim(-lim,lim)
        else:
            plt.axis('off')
        
        plt.savefig(self.imgDir+self.scene_uid+"_Layout.png" if imageTitle=="" else imageTitle)
        plt.clf()

        if drawRoomMask:
            self.drawRoomMask(self.imgDir+self.scene_uid+"_Mask.png" if imageTitle=="" else imageTitle[:-4]+"_Mask.png")

    def formGraph(self):
        #把经典关系标示出来
        for oi in range(len(self.OBJES)):
            o = self.OBJES[oi]
            shortest = 3
            RI = -1
            for ri in range(len(self.OBJES)):
                r = self.OBJES[ri]
                if (oi == ri) or not(o.class_name() in common_links.keys() and r.class_name() in common_links[o.class_name()]):
                    continue #semantic check print(o.class_name()+"   "+r.class_name())
                Tor = r.translation - o.translation
                Tor[1] = 0
                Lor = (Tor**2).sum()**0.5 + 0.0001
                Ior = Tor / Lor
                if Lor < shortest:
                    shortest = Lor
                    RI = ri
            if RI >= 0:
                self.LINKS.append(objLink(RI,oi,len(self.LINKS),self))

        #把贴墙关系都标识出来
        for oi in range(len(self.OBJES)):
            o = self.OBJES[oi]
            if len(o.destIndex)>0:
                continue
            om = o.matrix()
            for wi in range(len(self.WALLS)):
                w = self.WALLS[wi]
                #translate w.p into o's co-ordinate, scaled
                p=(om @ (w.p-o.translation)) / o.size
                #translate w.q into o's co-ordinate, scaled
                q=(om @ (w.q-o.translation)) / o.size
                #get a distance and projection
                n=(om @ w.n)

                pp=np.cross(n,p)[1]#min(np.cross(n,p),np.cross(n,q))
                qq=np.cross(n,q)[1]#max(np.cross(n,p),np.cross(n,q))
                
                if(wi==4 and oi == 5) and False:
                    print(w.p)
                    print(w.q)
                    print(w.n)
                    print(o.class_name())
                    print(o.translation)
                    print(" ")
                    print(p)
                    print(q)
                    print(n)
                    print(pp)
                    print(qq)
                    
                    pass

                if abs(abs(p@n)-1.0)<0.1 and min(pp,qq) < 0.9 and max(pp,qq) > -0.9:
                    self.LINKS.append(walLink(wi,oi,len(self.LINKS),o.translation,self))#a,b = LINKS[-1].arrow()
                
                elif abs((p*o.size-o.size)@n)<0.1 and min(pp,qq) < 0.0 and max(pp,qq) > -0.0:
                    self.LINKS.append(walLink(wi,oi,len(self.LINKS),o.translation,self))#a,b = LINKS[-1].arrow()

        #物体指向它朝向的一个东西
        for oi in range(len(self.OBJES)):
            o = self.OBJES[oi] #print("\n"+o.class_name() + "\n")
            if len(o.destIndex)>0:
                continue
            shortest = 3
            RI = -1
            for ri in range(len(self.OBJES)):
                r = self.OBJES[ri] #print(r.class_name())
                if ri == oi: #len(r.destIndex)==0:
                    continue 
                Tor = r.translation - o.translation
                Tor[1] = 0
                Lor = (Tor**2).sum()**0.5 + 0.0001
                Ior = Tor / Lor
                #find the facing one
                if (Ior @ o.direction()) > 0.5:# or True: #print("     ??? "+r.class_name())
                    if Lor < shortest:
                        shortest = Lor
                        RI = ri
            if RI >= 0:
                self.LINKS.append(objLink(RI,oi,len(self.LINKS),self))#print("    "+OBJES[RI].class_name())

    def adjustScene(self,movements):
        for move in movements:
            #print(w) for w in self.WALLS
            if "rate" in move.keys():
                self.breakWall(move["id"],move["rate"])
            elif "length" in move.keys():
                self.WALLS[move["id"]].mWall(move["length"])

    def breakWall(self,id,rate):
        #load all the links of 
        
        delList = []
        for l in self.WALLS[id].linkIndex:
            r = self.LINKS[l].rate
            if r < rate:
                self.LINKS[l].rate = r / rate
            else:
                self.LINKS[l].src = len(self.WALLS)+1
                self.LINKS[l].rate = (r-rate) / (1-rate)
                delList.append(l)
        for l in delList:
            self.WALLS[id].linkIndex.remove(l)
        #WALLS[id].linkIndex

        cutP = rate*self.WALLS[id].q + (1-rate)*self.WALLS[id].p
        A = len(self.WALLS)
        self.WALLS.append(
            wall(cutP,cutP,np.cross(self.WALLS[id].n,np.array([0,1,0])),id,A+1,A,scne=self)
        )
        self.WALLS.append(
            wall(cutP,self.WALLS[id].q,self.WALLS[id].n,A,self.WALLS[id].w2,A+1,scne=self)
        )
        for l in delList:
            self.WALLS[A+1].linkIndex.append(l)
        self.WALLS[self.WALLS[id].w2].w1= A+1
        self.WALLS[id].q = np.copy(cutP)
        self.WALLS[id].w2= A

    def formGroup(self): # to be continued
        pass

    def adjustGroup(self,sdev=0.2,cdev=2.):
        from numpy.random import rand as R
        from numpy.random import randint as Ri 
        from math import pi as PI 
               
        if len(self.GRUPS) == 1:
            self.GRUPS[0].adjust((R(3,)-0.5)*cdev,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
        else:
            t,l = (R()*2-1)*PI,max([self.GRUPS[0].size[0],self.GRUPS[0].size[2],self.GRUPS[1].size[0],self.GRUPS[1].size[2]]) - R()*0.1
            d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
            self.GRUPS[0].adjust( d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
            self.GRUPS[1].adjust(-d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
        if self.rmm:
            self.draftRoomMask()

    def drawRoomMask(self,maskTitle=""):
        Image.fromarray(self.roomMask.astype(np.uint8)).save(self.imgDir+self.scene_uid + "_Mask.png" if maskTitle=="" else maskTitle)

    def draftRoomMask(self):
        L = self.roomMask.shape[-1]
        img = Image.new("L",(L,L))  
        img1 = ImageDraw.Draw(img)  
        for g in self.GRUPS:
            img1.rectangle(g.imgSpaceBbox(), fill ="white",outline="gray",width=2)
        img1.line([((L>>1,L>>1) if len(self.GRUPS)==1 else self.GRUPS[1].imgSpaceCe()),self.GRUPS[0].imgSpaceCe()],fill ="white",width=15)

        self.roomMask = np.array(img).astype(np.float32)
        
    def exportAsSampleParams(self):
        c = copy(self.copy)
        c["translations"] = np.array([o.translation for o in self.OBJES if (o.gid >= 1 or (not self.grp))])
        c["sizes"] = np.array([o.size for o in self.OBJES])
        c["angles"] = np.array([[np.cos(o.orientation),np.sin(o.orientation)] for o in self.OBJES]) if c["angles"].shape[-1] == 2 else np.array([o.orientation for o in self.OBJES])
        c["room_layout"] = self.roomMask[None,:]
        if len(self.WALLS)>0:
            c["walls"] = []
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            I = self.WALLS[J].w2
            while I != J:
                I = self.WALLS[I].w2
                assert self.WALLS[I].v
                c["walls"].append([self.WALLS[I].p[0],self.WALLS[I].p[2],self.WALLS[I].n[0],self.WALLS[I].n[2]])
            c["walls"] = np.array(c["walls"])
        return c

    def recommendedWalls(self):
        #we are going 
        pass
