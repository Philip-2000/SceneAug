from .Obje import *
from .Link import *
from .Wall import *
from .Grup import *
from .Spce import *
from .Plan import plans
from .Bnch import singleMatch
import numpy as np
from matplotlib import pyplot as plt
from copy import copy,deepcopy
from PIL import Image, ImageDraw
import os

class scne():
    def __init__(self, scene, grp=False, windoor=False, wl=False, keepEmptyWL=False, cen=False, rmm=True, irt=16, imgDir="./"):#print(wl)print(keepEmptyWL)
        self.LINKS=[]
        self.copy = copy(scene)
        self.scene_uid = str(scene["scene_uid"]) if "scene_uid" in scene else ""
        rs = [r for r in ["Bedroom","KidsRoom", "ElderlyRoom", "LivingDiningRoom", "LivingRoom", "DiningRoom", "Library"] if self.scene_uid.find(r) > -1]
        self.roomType = "Room" if len(rs)==0 else rs[0]
        tr,si,oi,cl = scene["translations"],scene["sizes"],scene["angles"],scene["class_labels"]
        #firstly, store those objects and walls into the WALLS and OBJES
        ce = scene["floor_plan_centroid"] if cen else np.array([0,0,0])
        c_e= np.array([0,0,0]) if (cen) or (not (wl or windoor)) else scene["floor_plan_centroid"]
        self.grp = grp
        self.imgDir = imgDir

        self.OBJES=[obje(tr[i]+ce,si[i],oi[i],np.concatenate([cl[i],[0,0]])if windoor else cl[i],idx=i,scne=self) for i in range(len(tr))]

        self.roomMask = scene["room_layout"][0] if rmm else None
        self.rmm=rmm
        self.GRUPS=[]
        if grp:
            grops = np.ones(tr.shape[0]) if scene["grops"] is None else scene["grops"]
            self.GRUPS = [grup([o.idx for o in self.OBJES if grops[o.idx]==j+1],{"sz":self.roomMask.shape[-1],"rt":irt},j+1,scne=self) for j in range(int(max(grops)))]
        
        if windoor:
            widos = scene["widos"]
            for k in range(len(widos)):
                oii = np.array([np.math.atan2(widos[k][-1],widos[k][-2])])
                sii = np.array([max(widos[k][3],widos[k][5]),widos[k][4],min(widos[k][3],widos[k][5])]) #?
                tri = widos[k][:3]
                c = len(object_types)-1 if tri[1]-sii[1] < 0.1 else len(object_types)-2
                self.OBJES.append(obje(tri-c_e,sii,oii,i=c,idx=len(self.OBJES),scne=self))

        #obje(t,s,o,c,i)
        #wall(p,q,n,w1,w2)
        self.WALLS = walls(scene["walls"] if wl else [], c_e, self, keepEmptyWL=keepEmptyWL)
            
        
        # if wl:
        #     walls = scene["walls"]
        #     self.WALLS = [wall(two23(walls[j][:2])-c_e,two23(walls[(j+1)%len(walls)][:2])-c_e,np.array([walls[j][3],0,walls[j][2]]),(j-1)%len(walls),(j+1)%len(walls),j,scne=self) for j in range(len(walls))]
        self.text = ""
        self.SPCES = spces(self)#[]
        self.plans = []

    @classmethod
    def empty(cls,nm="",keepEmptyWL=False):
        return cls({"translations":[],"sizes":[],"angles":[],"class_labels":[],"scene_uid":nm},rmm=False,keepEmptyWL=keepEmptyWL)

    @classmethod
    def fromDict(cls, dct): #json.load(open(f,"r"))
        a = cls.empty()
        a.__dict__.update(dct)
        return a

    def addObject(self,objec):
        objec.idx = len(self.OBJES)
        objec.scne = self
        self.OBJES.append(objec)

    def registerWalls(self,wls):
        for w in wls.WALLS:
            w.scne=self
        wls.scne=self
        self.WALLS = wls

    def registerSpces(self,sps):
        for s in sps.SPCES:
            s.scne=self
        sps.scne=self
        self.SPCES = sps

    def draw(self,imageTitle="",d=False,lim=-1,drawWall=True,drawUngroups=False,drawRoomMask=False,classText=False):
        plt.figure(figsize=(10, 8))
        self.SPCES.draw(dr=False)
        # for i in range(len(self.SPCES)):
        #     self.SPCES[i].draw()

        if self.grp:
            for i in range(len(self.GRUPS)):
                self.GRUPS[i].draw()
                
        for i in range(len(self.OBJES)):
            if (not self.grp) or drawUngroups or (self.OBJES[i].gid):
                self.OBJES[i].draw(self.grp,d,text=classText)#corners = OBJES[i].corners2()
    
        if drawWall and len(self.WALLS):
            self.WALLS.draw()
            # J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            # contour,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]], self.WALLS[J].w2
            # while w != 0:
            #     contour.append([self.WALLS[w].p[0],self.WALLS[w].p[2]])
            #     w = self.WALLS[w].w2
            # contour = np.array(contour)
            # plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o", color="black")
        plt.axis('equal')

        for li in self.LINKS:
            li.draw()

        if lim > 0:
            plt.xlim(-lim,lim)
            plt.ylim(-lim,lim)
        else:
            plt.axis('off')
        
        plt.savefig(os.path.join(self.imgDir,self.scene_uid+"_Layout.png") if imageTitle=="" else imageTitle)
        plt.clf()
        plt.close()

        if drawRoomMask:
            self.drawRoomMask(os.path.join(self.imgDir,self.scene_uid+"_Mask.png") if imageTitle=="" else imageTitle[:-4]+"_Mask.png")

    def render(self):
        pass

    def formGraph(self):
        raise NotImplementedError
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

    def adjustGroup(self,sdev=0.2,cdev=2.): #augmentation
        raise NotImplementedError
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

    def roomMaskFromWalls(self,LST=None):
        L = self.roomMask.shape[-1]
        img = Image.new("L",(L,L)) 
        img1 = ImageDraw.Draw(img)  
        img1.polygon([ (w.p[0]*25.+(L>>1), w.p[2]*25.+(L>>1)) for w in self.WALLS], fill ="white")  
        self.roomMask = np.array(img).astype(np.float32)
        pass

    def areaField():
        pass

    def recommendedWalls(self):
        #we are going 
        pass
    
    def bpt(self):
        return np.concatenate([o.bpt() for o in self.OBJES],axis=0)

    def toSceneJson(self, roomType=None):
        sj = {"origin": self.scene_uid, "id": self.scene_uid,"bbox": {"min": [0,0,0], "max": [0,0,0]},"up": [0,1,0], "front": [0,0,1], "rooms":[]}
        #load the room

        rsj = {"id": self.scene_uid+"_0" if self.scene_uid else "0",
            "modelId": self.scene_uid,"roomTypes": [roomType],"origin": self.scene_uid,"roomId": 0,
            "bbox": {"min":[1000,1000,1000],"max":[-1000,-1000,-1000]},
            "objList":[],"blockList":[],"roomShape":[],"roomNorm":[],"roomOrient":[]
            }
        
        for o in self.OBJES:
            rsj["objList"].append(o.toObjectJson())
            if o.class_name().lower() in ["window", "door"]:
                rsj["blockList"].append(o.toObjectJson())
        
        #for o in self.OBJES:

        rsj["roomShape"],rsj["roomNorm"],rsj["roomOrient"] = self.WALLS.toWallsJson()
        if len([w for w in self.WALLS if w.v]) > 0:
            bb = self.WALLS.bbox()
            rsj["bbox"] = {"min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],"max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]}

        sj["bbox"] = rsj["bbox"]
        sj["rooms"].append(rsj)
        return sj

    @classmethod
    def fromSceneJson(cls,sj):
        scene = cls.empty(keepEmptyWL=True)
        scene.scene_uid = sj["id"]
        rsj = sj["rooms"][0]

        for oi in range(len(rsj["objList"])):
            scene.OBJES.append(obje.fromObjectJson(rsj["objList"][oi],oi))
        
        for oj in rsj["blockList"]:
            if False:
                scene.OBJES.append(obje.fromObjectJson(oj))

        scene.WALLS = walls.fromWallsJson(rsj["roomShape"],rsj["roomNorm"])
        return scene

    @classmethod
    def fromNpzs(cls,boxes=None,contours=None,conts=None,grops=None,load=True,name=None,dir="../novel3DFront/",**kwargs):
        if load:
            dn = dir+name
            boxes = np.load(dn+"/boxes.npz")
            contours = np.load(dn+"/contours.npz")["contour"] if os.path.exists(dn+"/contours.npz") else None
            conts = np.load(dn+"/conts.npz")["cont"] if os.path.exists(dn+"/conts.npz") else None
            grops = np.load(dn+"/group.npz")["group"] if os.path.exists(dn+"/group.npz") else None
        sceneDict = {"room_layout":np.zeros((1,64,64)).astype(np.uint8),"walls":contours,"widos":conts,"grops":grops,
            "translations":boxes["translations"],"sizes":boxes["sizes"],"angles":boxes["angles"],"class_labels":boxes["class_labels"],
            "floor_plan_centroid":boxes["floor_plan_centroid"],"scene_uid":boxes["scene_uid"]}
        kwargs["grp"] = (grops is not None) and ("grp" in kwargs and kwargs["grp"])
        kwargs["wl"] = (grops is not None) and ("wl" in kwargs and kwargs["wl"])
        kwargs["windoor"] = (grops is not None) and ("windoor" in kwargs and kwargs["windoor"])#print(kwargs)

        return cls(sceneDict,**kwargs)

    def objectView(self,id,bd=100000,scl=False,maxDis=100000):
        newOBJES = [self.OBJES[id].rela(o,scl) for o in self.OBJES if (o.idx != id and o.nid == -1)] # and not(o.class_name() in noPatternType)
        return sorted(newOBJES,key=lambda x:(x.translation**2).sum())[:min(len(newOBJES),bd)]
        newOBJES = [self.OBJES[id].rela(o) for o in self.OBJES if (o.idx != id)] # and not(o.class_name() in noPatternType)
        newOBJES = sorted(newOBJES,key=lambda x:(x.translation**2).sum())[:min(len(newOBJES),bd)]
        return [o for o in newOBJES if (o.nid == -1 and (o.translation**2).sum() < maxDis)]

    def nids(self):
        return set([o.nid for o in self.OBJES])

    def searchNid(self, nid, sig=True):
        s = [o for o in self.OBJES if o.nid == nid]
        return (s[0] if sig else s) if len(s)>0 else None

    def outOfBoundaries(self): #intersect area,  object occupied area in total,  room area
        contour = self.WALLS.shape()
        return sum([o.area-contour.intersection(o.shape()).area for o in self.OBJES]), sum([o.area for o in self.OBJES]), contour.area

import tqdm
class scneDs():
    def __init__(self,name="../novel3DFront",lst=[],prepare="uncond",num=8,**kwargs):#print(kwargs)
        from numpy.random import choice
        self.name,self._dataset = name, []
        if prepare=="uncond":
            LST = [choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst
            self.load(LST,name,num,**kwargs)
        elif prepare=="textcond":
            LST = [choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst
            self.prepareTextCond(LST,name,num,**kwargs)
        elif prepare=="roomcond":
            LST = [choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst
            self.prepareRoomCond(LST,name,num,**kwargs)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
    
    def load(self,LST,name="",num=8,**kwargs):
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s loading %s "%(name,LST[i][:20]))
                try:
                    scene = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    scene = scne.fromSceneJson(json.load(os.path.join(dir,LST[i])))
            else:
                pbar.set_description("empty scene %s "%(i))
                scene = scne.empty(str(i))
            self._dataset.append(scene)
    
    def prepareTextCond(self,LST,name="",num=8,**kwargs):
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s texting %s "%(name,LST[i][:20]))
                try:
                    template = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    template = scne.fromSceneJson(json.load(os.path.join(dir,LST[i])))
                scene = scne.empty(template.scene_uid)
                scene.text = template.text
            else:
                pbar.set_description("random texting %s "%(i))
                scene = scne.empty(str(i))
                #FIXME: generate a text for scene, although it hasn't been done yet, 
            self._dataset.append(scene)

    def prepareRoomCond(self,LST,name="",num=8,**kwargs):
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s rooming %s "%(name,LST[i][:20]))
                try:
                    template = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    template = scne.fromSceneJson(json.load(os.path.join(dir,LST[i])))
                scene = scne.empty(template.scene_uid)
                scene.registerWalls = template.WALLS
            else:
                pbar.set_description("random rooming %s "%(i))
                scene = scne.empty(str(i),keepEmptyWL=False)
                scene.WALLS.randomWalls()
            self._dataset.append(scene)

    def synthesis(self,syth,cond,T):
        from .Syth import agmt,gnrt,copl,rarg
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("%s-%s, %s "%(syth,cond,self._dataset[i].scene_uid[:20]))
            V=3 if len(self._dataset)==1 else 0
            if syth == "gnrt":
                S = gnrt(T,self._dataset[i],v=V)
            elif syth == "copl":
                S = copl(T,self._dataset[i],v=V)
            elif syth == "rarg":
                S = rarg(T,self._dataset[i],v=V)
            elif syth == "agmt":
                S = agmt(T,self._dataset[i],v=V)

            if syth == "agmt":
                S.scene.scene_uid += str(i)
                S.augment(cnt=1,draw=True)
            elif cond == "uncond":
                S.uncond(draw=True)
            elif cond == "textcond":
                S.textcond(draw=True)
            elif cond == "roomcond":
                S.roomcond(draw=True)
            #plans(self._dataset[i],T,v=3 if len(self._dataset)==1 else 0).recognize(**kwargs)
        pass

    def recognize(self,T,show=False,**kwargs):
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("recognizing %s "%(self._dataset[i].scene_uid[:20]))
            plans(self._dataset[i],T,v=3 if (len(self._dataset)==1 and not show) else 0).recognize(show=show,**kwargs)

    def optimize(self,T,**kwargs):
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("optimizing %s:"%(self._dataset[i].scene_uid[:20]))
            plans(self._dataset[i],T,v=3 if len(self._dataset)==1 else 0).recognize(opt=True,**kwargs)

    def evaluate(self,T):
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("evaluating %s "%(self._dataset[i].scene_uid[:20]))
            P = plans(self._dataset[i],T,v=0)
            fit,ass,_ = P.recognize(use=True,draw=False)
            pl = P.currentPlas
            #self.currentPlas.fit, sum([len(p) for p in self.currentPlas.plas]), None