#from . import OBJES
from .Obje import *
from .Link import *
from .Wall import *
from .Grup import *
from .Spce import *
import numpy as np
from matplotlib import pyplot as plt
from copy import copy

def two23(a):
    return np.array([a[0],0,a[1]])

#WALLS=[]
class scne():
    def __init__(self, scene, grp=False, windoor=False, wl=False, cen=False):
        self.LINKS=[]
        self.SPCES=[]
        self.copy = copy(scene)

        tr,si,oi,cl = scene["translations"],scene["sizes"],scene["angles"],scene["class_labels"],
        #firstly, store those objects and walls into the WALLS and OBJES
        ce = np.array([0,0,0]) if (not cen) else scene["floor_plan_centroid"]
        c_e= np.array([0,0,0]) if (cen and (wl or windoor)) else scene["floor_plan_centroid"]
        if windoor:
            widos = scene["widos"]
        if wl:
            walls = scene["walls"]
        if grp:
            grops=scene["grops"]

        self.OBJES=[]
        for i in range(len(tr)):
            cli = np.concatenate([cl[i],[0,0]])
            self.OBJES.append(obje(tr[i]+ce,si[i],oi[i],cli,idx=len(self.OBJES),gid=-1 if (not grp) else grops[i],scne=self))

        self.GRUPS=[]
        if grp:
            A = {-1:[],0:[],1:[]}
            for i in range(len(grops)):
                A[grops[i]].append(i)
            self.GRUPS.append(grup(A[0],0,scne=self))
            if len(A[1]):
                self.GRUPS.append(grup(A[1],1,scne=self))
        
        if windoor:
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
            for j in range(len(walls)):
                w1 = (j-1)%len(walls)
                w2 = (j+1)%len(walls)
                self.WALLS.append(wall(two23(walls[j][:2])-c_e,two23(walls[w2][:2])-c_e,np.array([walls[j][3],0,walls[j][2]]),w1,w2,j,scne=self))
        self.roomMask = None

    def draw(self,imageTitle,lim=-1,drawWall=True,objectGroup=False,drawUngroups=False,drawRoomMask=False):
        for i in range(len(self.SPCES)):
            self.SPCES[i].draw()

        for i in range(len(self.GRUPS)):
            self.GRUPS[i].draw()

        for i in range(len(self.OBJES)):
            if objectGroup and (not drawUngroups) and self.OBJES[i].gid == -1:
                continue
            self.OBJES[i].draw(objectGroup)#corners = OBJES[i].corners2()
            #plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-OBJES[i].class_index>2 else "*")

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
        
        plt.savefig(imageTitle)
        plt.clf()

        if drawRoomMask:
            self.drawRoomMask(imageTitle[:-4]+"Mask.png")

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

    def adjustGroup(self):
        #return
        #for g in GRUPS:
        #    g.adjust(np.array([0,0,0]),np.array([1,1,1]),1.5714)
        s0 = np.array([np.random.rand()*0.2+0.9,1.0,np.random.rand()*0.2+0.9])
        o0 = (np.random.randint(4)-2)*np.math.pi/2.0
        c = np.array([np.random.rand()*4.-2.,0.0,np.random.rand()*4.-2.])
        if len(self.GRUPS) == 1:
            self.GRUPS[0].adjust(c,s0,o0)
            self.draftRoomMask()
            return

        s1 = np.array([np.random.rand()*0.2+0.9,1.0,np.random.rand()*0.2+0.9])
        o1 = (np.random.randint(4)-2)*np.math.pi/2.0
        l = max([self.GRUPS[0].size[0],self.GRUPS[0].size[2],self.GRUPS[1].size[0],self.GRUPS[1].size[2]]) - np.random.rand()*0.5
        #l = np.random.rand()*2.0+1.5
        t = np.random.rand()*np.math.pi*2-np.math.pi
        d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
        self.GRUPS[0].adjust(d*l,s0,o0)
        self.GRUPS[1].adjust(-d*l,s1,o1)
        self.draftRoomMask()

    def drawRoomMask(self,maskTitle=""):
        from PIL import Image
        Image.fromarray(self.roomMask).convert("L").save(maskTitle)#.save(maskTitle)

    #from PIL import Image
    def draftRoomMask(self):
        #scale the image space to the real world
        #with a center and a scale
        #
        sz=64
        ce=np.array([0.,0.,0.])
        rt=8
        # if len(GRUPS)==1:
        #     ce = GRUPS[0].translation + np.array([np.random.rand()*2.-1.,0.0,np.random.rand()*2.-1.])
        # else:
        #     ce =(GRUPS[0].translation + GRUPS[1].translation)/2.0
        

        N = np.zeros((sz*2,sz*2))
        if len(self.GRUPS)==1:
            con = ce - self.GRUPS[0].translation
            lineMin,lineMax = 0.4,0.8
        else:
            con = self.GRUPS[1].translation - self.GRUPS[0].translation#print(con)
            lineMin,lineMax = 1.0,2.0
        con_ = con/np.linalg.norm(con)#print(con_)
        nom = np.cross(con_,np.array([0,1,0]))#print(nom)
        cons = con_/np.linalg.norm(con)
        areaMin,areaMax = 0.8,1.2
        for i in range(sz*2):
            for j in range(sz*2):
                t = np.array([(i-sz)/rt,0.0,(j-sz)/rt])+ce
                #check distance toward 
                #print(t)
                norm0t = np.clip( np.max(np.abs((t - self.GRUPS[0].translation)/self.GRUPS[0].size)),areaMin,areaMax)
                c0 = np.math.floor(255*(areaMax-norm0t)/(areaMax-areaMin))
                N[i,j]=c0
                if len(self.GRUPS)>1:
                    norm1t = np.clip( np.max(np.abs((t - self.GRUPS[1].translation)/self.GRUPS[1].size)),areaMin,areaMax)
                    c1 = np.math.floor(255*(areaMax-norm1t)/(areaMax-areaMin))
                    c0 = max(c0,c1)
                    N[i,j]=c0
                
                
                rat = (t-self.GRUPS[0].translation)@cons
                dis = np.abs((t-self.GRUPS[0].translation)@nom)
                #GRUPS[0].translation -> GRUPS[1].translation, GRUPS[0]
                if rat < 0.0 or rat > 1.0:
                    continue
                clipdis = np.clip(dis,lineMin,lineMax)
                c2 = np.math.floor(255*(lineMax-clipdis)/(lineMax-lineMin))
                c0 = max(c0,c2)
                N[i,j]=c0

        
        M = np.zeros((sz*2,sz*2))
        K = 2
        for i in range(sz*2):
            for j in range(sz*2):
                v = 0
                for k in range(max(i-K,0),min(i+K+1,sz*2)):
                    for l in range(max(j-K,0),min(j+K+1,sz*2)):
                        v += N[k,l]
                M[i,j] = int(v/25)
        self.roomMask = M
        return M
         
    def exportAsSampleParams(self):
        c = copy(self.copy)
        c["translations"] = np.array([o.translation for o in self.OBJES])
        #print(c["sizes"].shape)
        c["sizes"] = np.array([o.size for o in self.OBJES])
        #print(c["sizes"].shape)
        c["angles"] = np.array([[np.cos(o.orientation),np.sin(o.orientation)] for o in self.OBJES]) if c["angles"].shape[-1] == 2 else np.array([o.orientation for o in self.OBJES])
        c["room_layout"] = self.roomMask
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

    def exportAsTensor():
        pass

    def recommendedWalls(self):
        #we are going 
        pass