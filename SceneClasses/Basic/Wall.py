import numpy as np
from numpy.linalg import norm
EPS = 0.001#WALLSTATUS = True
class wall():
    def __init__(self, p, q, n, w1, w2, idx, v=True, spaceIn=False, sig=-1, scene=None, array=None):
        #global WALLSTATUS
        if n is not None and v and abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) > 0.01: #assert abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) < 0.01
            print("not straight %d, id=%d p:[%.3f,%.3f], q:[%.3f,%.3f], n:[%.3f,%.3f]"%(sig,idx,p[0],p[2],q[0],q[2],n[0],n[2])) #WALLSTATUS = False
            print(array)
        if n is not None and (p[0]-q[0])*n[2]>(p[2]-q[2])*n[0]: #assert (p[0]-q[0])*n[2]<=(p[2]-q[2])*n[0]-
            #print(traceback.format_stack())
            print("not right-hand %d, id=%d p:[%.3f,%.3f], q:[%.3f,%.3f], n:[%.3f,%.3f]"%(sig,idx,p[0],p[2],q[0],q[2],n[0],n[2])) #WALLSTATUS = False
            raise NotImplementedError
        self.linkIndex=[]
        self.idx=idx
        self.p=np.copy(p)
        self.q=np.copy(q)
        #self.n=np.copy(n) if n is not None else 
        self.w1=w1
        self.w2=w2
        self.v=v
        self.spaceIn=spaceIn
        self.scne=scene
        self.array=array#scne.WALLS if scne is not None else array#return WALLSTATUS
        
    #region: in/outputs---------#

        #region: presentation---#
    def __str__(self):
        return (" " if self.v else "              ")+str(self.w1)+"<-"+str(self.idx)+"->"+str(self.w2)+"\t[%.3f,%.3f]\t[%.3f,%.3f]\t[%.3f,%.3f]"%(self.p[0],self.p[2],self.q[0],self.q[2],self.n[0],self.n[2])

    def fromTensor(self,tensor,fmt):
        pass

    def toTensor(self,fmt):
        import torch
        res = torch.Tensor([])
        for f in fmt:
            if f=="tx":
                res = torch.cat([res,[self.p[0]]])
            elif f=="tz":
                res = torch.cat([res,[self.p[2]]])
            elif f=="nz":
                res = torch.cat([res,[self.n[2]]])
            elif f=="nx":
                res = torch.cat([res,[self.n[0]]])
        return res

    def renderable(self, colors=(0.5,0.5,0.5,1), width=0.5, height=0.5, back=0.4):
        from simple_3dviz import Lines
        return Lines( [ self.p + np.array([0,height,0]) - self.n*width*back - self.array[self.w1].n*width*back , self.q+np.array([0,height,0]) - self.n*width*back - self.array[self.w2].n*width*back ], colors=colors, width=width )
        
        #endregion: presentation#
    
    #endregion: in/outputs------#

    #region: properties---------#
    @property
    def n(self):
        assert norm(np.array([self.p[2]-self.q[2],0,self.q[0]-self.p[0]]))>1e-6
        return np.array([self.p[2]-self.q[2],0,self.q[0]-self.p[0]])/norm(np.array([self.p[2]-self.q[2],0,self.q[0]-self.p[0]]))

    @property
    def length(self):
        return (((self.p-self.q)**2).sum())**0.5

    @property
    def center(self):
        return (self.p+self.q)/2

    def distance(self,P):
        return (P-self.p)@(self.n), (P-self.p)@(self.n)*self.n

    def minVec(self,P):
        return self.distance(P)[1] if self.over(P) else min(P-self.p,P-self.q,key=lambda x:norm(x))

    def over(self,P):
        return (P-self.q)@(self.p-self.q) > -EPS and (P-self.p)@(self.q-self.p) > -EPS

    def on(self,P,eps=EPS):
        return self.over(P) and abs(self.distance(P)[0]) < eps

    def rate(self,p,checkOn=False):
        r = ((p-self.p)@(self.q-self.p)) / ((self.q-self.p)@(self.q-self.p))
        assert (not checkOn) or (abs((p-self.p)@self.n)<EPS and 0<=r and r<=1)
        return r

    #endregion: properties------#

    #region: relationship-------#
    def face(self,w):
        return self.n @ w.n < -1+1e-3
    
    def face_in(self,w):
        if not self.face(w): return False
        return self.over(w.p) or self.over(w.q) or w.over(self.p) or w.over(self.q)
    
    def stick(self,x):
        wp,wq,xp,xq = np.cross(np.abs(self.n),self.p)[1],np.cross(np.abs(self.n),self.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
        return self.v and x.v and abs(self.n@x.n)>0.9 and abs(np.abs(self.n)@self.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

    def cross(self,x):
        wxp,wxq,xwp,xwq,wwp,xxp = self.n@x.p, self.n@x.q, x.n@self.p, x.n@self.q, self.n@self.p, x.n@x.q
        return self.v and x.v and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))

    #endregion: relationship----#

    #region: operations---------#

        #region: movement-------#
    def mWall(self,L): 
        n = np.copy(self.n)
        #oldp = np.copy(self.p)
        self.p += n*L
        self.array[self.w1].q += n*L
        #oldq = np.copy(self.q)
        self.q += n*L
        self.array[self.w2].p += n*L
        #self.array[self.w1].adjustWall(oldp,np.copy(self.p),self.idx)
        #self.array[self.w2].adjustWall(oldq,np.copy(self.q),self.idx)
        for i in self.linkIndex:
            self.scne.LINKS[i].adjust(self.n*L)
        
    # def adjustWall(self,oldp,p,hint=-1):
    #     raise NotImplementedError
    #     oldn = self.n
    #     if hint<0:
    #         if ((self.p-oldp)**2).sum()<0.001:
    #             oldq = self.q
    #             self.p=p
    #         elif ((self.q-oldp)**2).sum()<0.001:
    #             oldq = oldp
    #             oldp = self.p
    #             self.q=p
    #         else:
    #             print("adjustWall error")
    #             return
    #     else:
    #         if hint == self.w1:
    #             oldq = self.q
    #             self.p=p
    #         elif hint == self.w2:
    #             oldq = oldp
    #             oldp = self.p
    #             self.q=p
    #         else:
    #             print("false hint")
    #             return
    #     if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
    #         self.n = -self.n
    #     #elf.lengthh()#=(((self.p-self.q)**2).sum())**0.5

    #     for i in self.linkIndex:
    #         self.scne.LINKS[i].modify(oldp, oldq, oldn)
        
    def break_(self,rate,length=None):
        if rate is None: rate = length / self.length
    
        delList = []
        for l in self.linkIndex:
            r = self.scne.LINKS[l].rate
            if r < rate:
                self.scne.LINKS[l].rate = r / rate
            else:
                self.scne.LINKS[l].src = len(self.array)+1
                self.scne.LINKS[l].rate = (r-rate) / (1-rate)
                delList.append(l)
        for l in delList: self.linkIndex.remove(l)

        self.q = np.copy(rate*self.q + (1-rate)*self.p)
        self.array.insertWall(self.idx)
        self.array.insertWall(self.idx)

        for l in delList: self.array[len(self.WALLS)-2].linkIndex.append(l)

    def delete(self,delta=EPS):
        w = self.array[self.w1]
        assert self.length < delta*2 or abs(abs(w.n@self.n)-1.0)<EPS or abs((w.p-self.q)@self.n)<EPS or abs((w.p-self.q)@w.n)<EPS
        self.v = False
        w.q = np.copy(self.q)
        w.w2 = self.w2
        self.array[self.w2].w1 = w.idx
    
    def squeeze(self):
        L1 = self.length/2.0 if (self.q-self.p) @ self.array[self.w1].n > 0 else -self.length/2.0
        L2 = self.length/2.0 if (self.p-self.q) @ self.array[self.w2].n > 0 else -self.length/2.0
        self.array[self.w1].mWall(L1)
        self.array[self.w2].mWall(L2)
        self.array[self.w2].delete() #self.deleteWall(self.WALLS[I].w2,EPS)
        self.delete() #self.deleteWall(I,EPS)
        #endregion: movement----#

        #region: optFields----------#
    def field(self,sp,config): #w_o, w_i, out
        minVec = -self.minVec(sp.transl)
        try: #for object samples
            projectedVec = max(minVec@(-sp.o.direction()),0.0)*(-sp.o.direction())
            w_i = projectedVec * np.clip(1.0-norm(projectedVec)/config["bound"],0.0,1.0)
            return minVec, w_i if sp.TRANSL[2]==-1 else np.array([.0,.0,.0]), (minVec@self.n > 0)
        except: #for field sams
            projectedVec = minVec
            w_i = projectedVec * np.clip(1.0-norm(projectedVec)/config["bound"],0.0,1.0)
            return minVec, w_i, (minVec@self.n > 0)
        # if norm(w_i) > 0.01 and sp.TRANSL[2] == -1:
        #     print(sp.o)
        #     print(sp.o.direction())
        #     print(-sp.o.direction())
        #     print(minVec)
        #     print(projectedVec)
        #     print(w_i)
        #endregion: optFields-------#

    #endregion: operations------#

def two23(a):
    return np.array([a[0],0,a[1]])
class walls(): #Walls[j][2] is z, Walls[j][3] is x
    def __init__(self, Walls=[], c_e=0, scne=None, c=[0,0], a=[2.0,2.0], printLog=False, name="", flex=1.2, drawFolder="", keepEmptyWL = False, cont=[]):
        #print(keepEmptyWL)
        if len(Walls)>0: #Walls[j][2] is z, Walls[j][3] is x
            try:
                self.WALLS = [wall(two23(Walls[j][:2])-c_e,two23(Walls[(j+1)%len(Walls)][:2])-c_e,np.array([Walls[j][3],0,Walls[j][2]]),(j-1)%len(Walls),(j+1)%len(Walls),j,scene=scne,array=self) for j in range(len(Walls))]
            except:
                self.WALLS = [wall(two23(Walls[j][:2])-c_e,two23(Walls[(j+1)%len(Walls)][:2])-c_e,None,(j-1)%len(Walls),(j+1)%len(Walls),j,scene=scne,array=self) for j in range(len(Walls))]
        else:#if a[0]<0 or a[1]<0:print(a)
            self.WALLS = [wall(two23([c[0]+a[0]-2*a[0]*int((i+1)%4>1),c[1]+a[1]-2*a[1]*int(i%4>1)]),two23([c[0]+a[0]-2*a[0]*int(i%4<2),c[1]+a[1]-2*a[1]*int((i+1)%4>1)]),two23([(2.0-i)*(i%2),(-1.0+i)*(1-i%2)]),(i-1)%4,(i+1)%4,i,array=self) for i in range(4)] if not keepEmptyWL else []
        self.LOGS = []
        self.printLog = printLog
        self.scne = scne
        self.name = name
        self.flex = flex
        self.drawFolder = drawFolder
        from .Wndr import wndrs
        self.windoors = wndrs(self,cont,c_e)
        self.other={}

    #region: magics--------------#
    def __getitem__(self, idx):
        return self.WALLS[idx]

    def __len__(self):
        return len(self.WALLS)

    def __iter__(self):
        return iter(self.WALLS)
    #endregion: magics-----------#
    
    #region：in/outputs----------#
        #region: inputs----------#
    def __str__(self):
        return '\n'.join([str(w) for w in self.WALLS]) + "\n" + str(self.windoors)

    @classmethod
    def fromLog(cls,f,name="",drawFolder=""):
        raise NotImplementedError
        from .Logg import distribute
        import json
        a = cls(name=name,drawFolder=drawFolder)
        a.LOGS = [distribute(a,l) for l in open(f,"r").readlines()]
        a.centerize()
        [print(l) for l in a.LOGS if a.printLog]
        
        from .Obje import obje, object_types
        wf = f.replace(".txt",".json")
        try:
            #assert False
            dd = json.load(open(wf,"r"))
            for i in dd:
                d = dd[i]
                windoor = obje.fromFlat(np.array(d),j=len(object_types)-1 if d[1]-d[4]<0.01 else len(object_types)-2)
                a.windoors[i] = windoor
        except:
            #assert False
            ws = []
            while (np.random.rand()<0.7 and len(ws)<3) or len(ws)<2:
                vws = [w for w in a.WALLS if w.v and (w.idx not in ws) and w.length>1.5]
                w=vws[np.random.randint(len(vws))]
                
                thick2 = 0.12
                rt = np.random.rand()*(0.5)+0.25
                mins = min(rt*w.length, (1-rt)*w.length)
                if mins<0.8:
                    continue
                
                doc = w.p*(1-rt)+w.q*rt - (thick2)*w.n
                width2 = np.random.rand()*(mins-0.8)+0.4
                
                if np.random.rand()<0.8:
                    windoor = obje(doc+np.array([0,1.0,0]),np.array([width2,1.0,thick2]),np.array([np.math.atan2(w.n[0],w.n[2])]),i=len(object_types)-1)
                else:
                    windoor = obje(doc+np.array([0,1.2,0]),np.array([width2,0.5,thick2]),np.array([np.math.atan2(w.n[0],w.n[2])]),i=len(object_types)-2)
                a.windoors[str(w.idx)] = windoor
                ws.append(w.idx)

            wfr= open(wf,"w")
            wfr.write(json.dumps({i:a.windoors[i].flat(False).tolist() for i in a.windoors}))
        a.processWithWindoor()
        return a
    
    def fromTensor(self,tensor,fmt):
        [w.fromTensor(tensor[i],fmt) for i,w in enumerate(self)]
        self.windoors.fromTensor()

    @classmethod
    def fromWallsJson(cls,sha,nor,scne,jsn=[]): #nor[:][0] is x, nor[:][1] is z
        return walls(np.array([[sha[i][0],sha[i][1],nor[i][1],nor[i][0]] for i in range(len(sha))]),scne=scne,cont=jsn) #Walls[j][2] should be z, Walls[j][3] should be x
        #endregion: inputs-------#

        #region: presentation----#
    def exportAsSampleParams(self):
        c = []
        if len(self.WALLS)>0:
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            I = self.WALLS[J].w2
            while I != J:
                I = self.WALLS[I].w2
                assert self.WALLS[I].v
                c.append([self.WALLS[I].p[0],self.WALLS[I].p[2],self.WALLS[I].n[0],self.WALLS[I].n[2]])
        return np.array(c)
    
    def toTensor(self,fmt,length):
        import torch
        return torch.cat([w.toTensor(fmt) for w in self.WALLS]+[self.WALLS[-1].tensor(format)]*(length-len(self)),axis=0).reshape((1,length,-1))

    def toWallsJson(self,rsj={}):#nor[:][0] is x, nor[:][1] is z
        sha,nor,ori = [],[],[]
        if len([w.idx for w in self.WALLS if w.v]):
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            sha,nor,ori,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]],[[self.WALLS[J].n[0],self.WALLS[J].n[2]]],[np.math.atan2(self.WALLS[J].n[0],self.WALLS[J].n[2])], self.WALLS[J].w2
            while w != J:
                sha.append([float(self.WALLS[w].p[0]),float(self.WALLS[w].p[2])])
                nor.append([float(self.WALLS[w].n[0]),float(self.WALLS[w].n[2])])
                ori.append(float(np.math.atan2(self.WALLS[J].n[0],self.WALLS[J].n[2])))
                w = self.WALLS[w].w2
            bb = self.bbox()
            rsj["bbox"] = {"min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],"max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]}
        rsj["roomShape"],rsj["roomNorm"],rsj["roomOrient"] = sha, nor, ori
        rsj["blockList"] = self.windoors.toBlocksJson()
        return rsj #nor[:][0] is x, nor[:][1] is z
    
    def draw(self,end=False,suffix=".png",color="black"):#print(self)
        from matplotlib import pyplot as plt
        if len([w.idx for w in self.WALLS if w.v]):
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            contour,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]], self.WALLS[J].w2
            while w != J:
                contour.append([self.WALLS[w].p[0],self.WALLS[w].p[2]])
                w = self.WALLS[w].w2
            contour = np.array(contour)
            plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o", color=color)
            if end and self.drawFolder:
                plt.axis('equal')
                L = max(self.LH())
                plt.xlim(-self.flex*L,self.flex*L)
                plt.ylim(-self.flex*L,self.flex*L)
                plt.savefig(self.drawFolder+self.name+suffix)
                plt.clf()
        self.windoors.draw()

    def writeLog(self):
        with open(self.drawFolder+self.name+".txt","w") as f:
            [f.write(str(l)) for l in self.LOGS]
        
    def output(self):
        self.draw(True)
        self.writeLog()

    def draftRoomMask(self,sz=64,rt=25.):
        from PIL import Image,ImageDraw
        img = Image.new("L",(sz,sz)) 
        img1 = ImageDraw.Draw(img)  
        img1.polygon([ (w.p[0]*rt+(sz>>1), w.p[2]*rt+(sz>>1)) for w in self.WALLS], fill ="white")  
        self.scne.roomMask = np.array(img).astype(np.float32)#img.save("./roommask.png")
        return img.load() #self.scne.roomMask,

    def renderable_floor(self,depth=0, sz=192, rt=25.):
        from simple_3dviz import Lines
        pixels,points = self.draftRoomMask(sz*2+1,rt),[]
        for i in range(-sz,sz+1):
            out = True #
            for j in range(-sz,sz+1):
                if (pixels[i+sz,j+sz] < 0.9) ^ out:
                    points.append([i/rt, -depth, j/rt])#print("%.3f, %.3f"%(i/rt,j/rt))
                    out = not out
            assert out
        return Lines(points,colors=(1,1,1,1),width=5/rt)
    
    def renderable(self):
        return [self.renderable_floor()] + [w.renderable() for w in self.WALLS]
        #endregion: presentation-#
    #endregion: in/outputs-------#

    #region: properties----------#
    @property
    def maxx(self):
        return max([w.p[0] for w in self.WALLS if w.v])
    
    @property
    def minx(self):
        return min([w.p[0] for w in self.WALLS if w.v])
    
    @property
    def x(self):
        return (self.maxx+self.minx)/2.0

    @property
    def maxz(self):
        return max([w.p[2] for w in self.WALLS if w.v])
    
    @property
    def minz(self):
        return min([w.p[2] for w in self.WALLS if w.v])

    @property
    def z(self):
        return (self.maxz+self.minz)/2.0
    
    @property
    def lenx(self):
        return self.maxx-self.minx
    
    @property
    def lenz(self):
        return self.maxz-self.minz
    
    @property
    def c(self):
        return np.array([self.minx+self.lenx/2,0,self.minz+self.lenz/2])
    
    @property
    def a(self):
        return [self.lenx,0.0,self.lenz]

    @property
    def LH(self):
        return [max([abs(w.q[0]) for w in self.WALLS if w.v]),max([abs(w.q[2]) for w in self.WALLS if w.v])]

    @property
    def bbox(self):
        return np.array([[min([w.p[0] for w in self.WALLS if w.v]),0,min([w.p[2] for w in self.WALLS if w.v])],
                       [max([w.p[0] for w in self.WALLS if w.v]),3,max([w.p[2] for w in self.WALLS if w.v])]])

    def shape(self):
        from shapely.geometry import Polygon
        return Polygon(self.toWallsJson()["roomShape"])
    
    def npArray(self):
        rsj = self.toWallsJson()
        sha,nor = rsj["roomShape"], rsj["roomNorm"]
        return np.concatenate([np.array(sha), np.array(nor)[:,1:], np.array(nor)[:,:1]],axis=0)

    def cross(self,other=None):
        otherWALLS = self.WALLS if other is None else other.WALLS
        for i in range(len(self.WALLS)):
            for j in range(i if other is None else len(otherWALLS)):
                if self.WALLS[i].v and otherWALLS[j].v and self.WALLS[i].cross(otherWALLS[j]):#if self.crossWall(self.WALLS[i], otherWALLS[j]):
                    return True
        return False
    #endregion: properties-------#

    #region: operations----------#
    def inward_door(self):
        wls = walls([],scne=self.scne,flex=self.flex,drawFolder=self.drawFolder,keepEmptyWL=True)
        for w in self.WALLS:
            wls.WALLS.append(wall(w.p,w.q,w.n,w.w1,w.w2,w.idx,scene=self.scne,array=wls))
            wls.WALLS[-1].v = w.v
        for d in self.windoors:
            f = d.inward(wls)
        return wls

    def max_in(self,id,bound=1):
        for w in self.WALLS:
            if w.v and w.idx != id and w.face_in(self.WALLS[id]):
                dis = self.WALLS[id].distance(w.p)[0] #print("dis",dis)
                if dis > 0 and dis < bound:
                    bound = dis
        return bound

    def searchWall(self,p,back=False,f=False):
        d,W=1000,None
        for w in self.WALLS:
            if w.v and w.over(p):
                if f: print(w.idx,w.distance(p)[0])
                if abs(w.distance(p)[0])<d:
                    W=w
                    d=abs(w.distance(p)[0])
        if back:
            if self.WALLS[W.w2].over(p) and abs(self.WALLS[W.w2].distance(p)[0])<=d:
                return self.WALLS[W.w2],d
        else:
            if self.WALLS[W.w1].over(p) and abs(self.WALLS[W.w1].distance(p)[0])<=d:
                return self.WALLS[W.w1],d
        return W,d

    def insertWall(self,w1=None,w2=None):
        assert (w1 is not None) or (w2 is not None)
        if w1 is not None and  w2 is not None:
            assert len(self.WALLS)>w1 and self.WALLS[w1].v and len(self.WALLS)>w2 and self.WALLS[w2].v and w2 == self.WALLS[w1].w2 and w1 == self.WALLS[w2].w1
        elif w1 is not None:
            assert len(self.WALLS)>w1 and self.WALLS[w1].v
            w2 = self.WALLS[w1].w2
        elif w2 is not None:
            assert len(self.WALLS)>w2 and self.WALLS[w2].v
            w1 = self.WALLS[w2].w1
        ID = len(self.WALLS)

        p=self.WALLS[w1].q
        self.WALLS[w1].w2 = ID
        q=self.WALLS[w2].p
        self.WALLS[w2].w1 = ID
        n = None#np.cross(self.WALLS[w1].n,np.array([0,1,0])) if (p-q)@(p-q)<0.01 else np.array([p[2]-q[2],0,q[0]-p[0]])/norm(np.array([p[2]-q[2],0,q[0]-p[0]]))
        self.WALLS.append(wall(p,q,n,w1,w2,ID,scene=self.scne,array=self))
        return ID
    #endregion: operations-------#

    #region: random-related------#
    def maxHeight(self,x,bd=4.0):
        return min([bd]+[(w.p-x.p)@x.n for w in self.WALLS if (w.v and ((w.p-x.p)@x.n > 0.01))]) #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值

    def maxDepth(self,x,bd=-4.0):
        return max([bd]+[(w.p-x.p)@x.n for w in self.WALLS if (w.v and ((w.p-x.p)@x.n <-0.01))]) #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值

    # def breakWall(self,id,rate=None,length=None):
    #     if rate is None: rate = length / self.WALLS[id].length
    
    #     delList = []
    #     for l in self.WALLS[id].linkIndex:
    #         r = self.scne.LINKS[l].rate
    #         if r < rate:
    #             self.scne.LINKS[l].rate = r / rate
    #         else:
    #             self.scne.LINKS[l].src = len(self.WALLS)+1
    #             self.scne.LINKS[l].rate = (r-rate) / (1-rate)
    #             delList.append(l)
    #     for l in delList:
    #         self.WALLS[id].linkIndex.remove(l)

    #     self.WALLS[id].q = np.copy(rate*self.WALLS[id].q + (1-rate)*self.WALLS[id].p)
    #     self.insertWall(id)
    #     self.insertWall(id)

    #     for l in delList:
    #         self.WALLS[len(self.WALLS)-1].linkIndex.append(l)

    # def squeezeWall(self,I):
    #     L1 = self.WALLS[I].length/2.0 if (self.WALLS[I].q-self.WALLS[I].p) @ self.WALLS[self.WALLS[I].w1].n > 0 else -self.WALLS[I].length/2.0
    #     L2 = self.WALLS[I].length/2.0 if (self.WALLS[I].p-self.WALLS[I].q) @ self.WALLS[self.WALLS[I].w2].n > 0 else -self.WALLS[I].length/2.0
    #     self.WALLS[self.WALLS[I].w1].mWall(L1)
    #     self.WALLS[self.WALLS[I].w2].mWall(L2)
    #     self.WALLS[self.WALLS[I].w2].delete() #self.deleteWall(self.WALLS[I].w2,EPS)
    #     self.WALLS[I].delete() #self.deleteWall(I,EPS)

    def squeezeWalls(self):
        from .Logg import dllg
        i=0
        while i<len(self.WALLS):
            if self.WALLS[i].v and self.WALLS[i].length < 0.7:
                self.LOGS.append(dllg(self,i))#{"id":I,"delete":0})#self.LOGS[-1].operate()
                i=-1
            i+=1

    def randomWalls(self):
        from .Logg import rtlg, mvlg
        a,b = 5,3
        while np.random.rand() < 0.8 and a > 0 and b > 0:
            a -= 1
            if np.random.rand() < 0.5:
                b -= 1
                wls = sorted([w.idx for w in self.WALLS if w.v], key=lambda x:-self.WALLS[x].length)
                wid = 0
                while wid < len(wls)-1 and self.WALLS[wls[wid]].length > 1.5 and np.random.rand()<0.5:
                    wid += 1
                wid = wls[wid]
                L = self.WALLS[wid].length
                r = np.random.uniform(0.9 / L, 1.0 - (0.9 / L)) #0.5+(t-0.5)*0.5
                self.LOGS.append(rtlg(self,wid,r))#{"id":wid,"rate":r})
            else: 
                wid = np.random.randint(len(self.WALLS))
            lower = min(self.maxDepth(self.WALLS[wid])+1.0,0.0) # <0
            upper = 0.0#self.maxHeight(WALLS[wid])-1.0# >0
            t = np.random.uniform(lower, upper)
            self.LOGS.append(mvlg(self,wid,t,lower,upper))#{"id":wid,"leng":t,"lower":lower,"upper":upper})
        
            if self.cross():
                J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
                I = self.WALLS[J].w2
                while I != J:
                    I = self.WALLS[I].w2
                    if self.printLog:
                        print(str(self.WALLS[I]))#str(self.WALLS[I].w1)+"<-"+str(self.WALLS[I].idx)+"->"+str(self.WALLS[I].w2),self.WALLS[I].p,self.WALLS[I].n)
                [print(l) for l in self.LOGS if self.printLog]
                return True
        
        self.squeezeWalls()
        self.centerize()
        [print(l) for l in self.LOGS if self.printLog]

    def centerize(self):
        xs,zs = [w.p[0] for w in self.WALLS if w.v],[w.p[2] for w in self.WALLS if w.v]
        cen = np.array([(max(xs)+min(xs))/2.0,0.0,(max(zs)+min(zs))/2.0])
        for w in self.WALLS: w.p,w.q = w.p-cen,w.q-cen

    #endregion: random-related---#

    #region: trash---------------#
    # def deleteWall(self,id,delta=EPS):
    #     w = self.WALLS[self.WALLS[id].w1]
    #     #assert self.WALLS[id].length < delta*2 
    #     assert self.WALLS[id].length < delta*2 or abs(abs(w.n@self.WALLS[id].n)-1.0)<EPS or abs((w.p-self.WALLS[id].q)@self.WALLS[id].n)<EPS or abs((w.p-self.WALLS[id].q)@w.n)<EPS
    #     self.WALLS[id].v = False
    #     w.q = np.copy(self.WALLS[id].q)
    #     #w.lengthh()
    #     #w.resetN()
    #     w.w2 = self.WALLS[id].w2
    #     self.WALLS[self.WALLS[id].w2].w1 = w.idx

    # def minusWall(self,id,delta):
    #     if not self.WALLS[id].v:
    #         return
    #     if abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w2].n + 1.)< EPS:
    #         I,J = id,self.WALLS[id].w2
    #     elif abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w1].n + 1.)< EPS:
    #         I,J = self.WALLS[id].w1,id
    #     else:
    #         if abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w2].n - 1.)< EPS:
    #             self.deleteWall(self.WALLS[id].w2,delta)
    #         elif abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w1].n - 1.)< EPS:
    #             self.deleteWall(id,delta)
    #         return #self.WALLS
    

    #     if self.WALLS[I].length < self.WALLS[J].length:
    #         P,K = self.WALLS[I].p,self.WALLS[I].w1
    #     else:
    #         P,K = self.WALLS[J].q,self.WALLS[J].w2
    #     #print(self)
    #     self.WALLS[I].adjustWall(self.WALLS[I].q,P)
    #     self.WALLS[J].adjustWall(self.WALLS[J].p,P)

    #     # print("I=%d,J=%d,K=%d"%(I,J,K))
    #     # print(self)
        
    #     if self.WALLS[I].length < delta*2:
    #         self.deleteWall(I,delta)
    #     if self.WALLS[J].length < delta*2:
    #         self.deleteWall(J,delta)
        
    #     self.minusWall(K,delta)
    #     self.regularize(I)
    #     self.regularize(J)
    #     self.regularize(self.WALLS[I].w1)
    #     self.regularize(self.WALLS[J].w2)

    # def regularize(self,id):
    #     w = self.WALLS[id]
    #     if not w.v:
    #         return
    #     oldp,oldq  = np.copy(w.p),np.copy(w.q)
    #     if min(abs(w.n[0]),abs(w.n[2]))>=EPS*50:
    #         print("warning: %.3f >= %.3f * %.1f, wall should not be regularized"%(min(abs(w.n[0]),abs(w.n[2])),EPS,50))
    #         #raise NotImplementedError

    #     if abs(w.n[0])<abs(w.n[2]):
    #         b = (w.p[2]+w.q[2])/2.0
    #         w.p[2],w.q[2] = b,b
    #     else:
    #         b = (w.p[0]+w.q[0])/2.0
    #         w.p[0],w.q[0] = b,b
    #     #w.resetN()
    #     self.WALLS[w.w2].adjustWall(oldq,w.q,id)
    #     self.WALLS[w.w1].adjustWall(oldp,w.p,id)

    # def processWithWindoor(self):
    #     raise NotImplementedError
    #     for a in self.windoors:
    #         o = self.windoors[a]
    #         if o.class_name == "door":
    #             a = int(a)

    #             W=self.WALLS[a]

    #             W.q = o.translation+(o.matrix()*np.array([[ o.size[0]+0.05,0,o.size[2]]])).sum(axis=-1)
    #             W.q[1] = 0
    #             self.insertWall(a)
    #             W.q = o.translation+(o.matrix()*np.array([[ o.size[0]+0.05,0,o.size[2]+0.4]])).sum(axis=-1)
    #             W.q[1] = 0
    #             self.insertWall(a)
    #             W.q = o.translation+(o.matrix()*np.array([[-o.size[0]-0.05,0,o.size[2]+0.4]])).sum(axis=-1)
    #             W.q[1] = 0
    #             self.insertWall(a)
    #             W.q = o.translation+(o.matrix()*np.array([[-o.size[0]-0.05,0,o.size[2]]])).sum(axis=-1)
    #             W.q[1] = 0
    #             self.insertWall(a)
            


    #     # X=self.WALLS[W.w2]
    #     # a=W.idx
    #     # for i in range(3,-1,-1):
    #     #     X.p = Spce.corners[(PID+i)%4]
    #     #     a=self.WALLS.insertWall(a)

    #     pass
    
    # def stickWall(self,w,x):
    #     wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
    #     return w.v and x.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

    # def crossWall(self,w,x):
    #     wxp,wxq,xwp,xwq,wwp,xxp = w.n@x.p, w.n@x.q, x.n@w.p, x.n@w.q, w.n@w.p, x.n@x.q
    #     return w.v and x.v and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))
    #endregion: trash------------#

    #region: optFields-----------#
    def optFields(self,sp,o,config):
        if o: #for object samples
            wo, wi, dr = np.array([9.,9.,9.]), np.array([0.,0.,0.]), self.windoors.optFields(sp,o,config["door"])
            for w in self:
                w_o, w_i, out = w.field(sp,config["wall"])
                wo,wi = min(w_o,wo,key=lambda x:norm(x)) if out else wo, max(w_i,wi,key=lambda x:norm(x)) if not out else wi

            from shapely.geometry import Point
            if self.shape().contains(Point(sp.transl[0],sp.transl[2])): #inside
                return (np.array([.0,.0,.0]),wi,dr)
            else:#outside
                return (wo,np.array([.0,.0,.0]),dr)
        else: #for field samples
            wo, wi, dr = np.array([9.,9.,9.]), np.array([0.,0.,0.]), self.windoors.optFields(sp,o,config["door"])
            for w in self:
                w_o, w_i, out = w.field(sp,config["wall"])
                wo,wi = min(w_o,wo,key=lambda x:norm(x)) if out else wo, max(w_i,wi,key=lambda x:norm(x)) if not out else wi

            from shapely.geometry import Point
            if self.shape().contains(Point(sp.transl[0],sp.transl[2])): #inside
                return (np.array([.0,.0,.0]),0.0),(wi,(norm(wi)**2)/2.0),dr
            else:#outside
                return (wo,(norm(wo)**2)/2.0),(np.array([.0,.0,.0]),0.0),dr
    #endregion: optFields--------#
