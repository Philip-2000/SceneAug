#from ...Basic.Wall import *
import numpy as np

class rect():
    def __init__(self,c=None,a=None,c0=None,c1=None):
        if c is None or a is None:
            mins = np.min(np.array([c0,c1]),axis=0)
            maxs = np.max(np.array([c0,c1]),axis=0)
            self.c = (c0+c1)/2.0
            self.a = (maxs-mins)/2.0
        else:
            self.c = c
            self.a = a

    def __str__(self):
        return "[[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f]] at [%.2f,%.2f] with [%.2f,%.2f]"%(self.maxx,self.maxz,self.minx,self.maxz,self.minx,self.minz,self.maxx,self.minz,self.x,self.z,self.a[0],self.a[2])

    @property
    def min(self):
        return self.c-self.a
    
    @property
    def max(self):
        return self.c+self.a
    
    @property
    def corners(self):
        return [np.array(self.max),np.array([self.min[0],.0,self.max[2]]),np.array(self.min),np.array([self.max[0],0.0,self.min[2]])]
        
    @property
    def x(self):
        return self.c[0]
    
    @property
    def z(self):
        return self.c[2]
    
    @property
    def xs(self):
        return self.c[0]-self.a[0], self.c[0]+self.a[0]
    
    @property
    def zs(self):
        return self.c[2]-self.a[2], self.c[2]+self.a[2]
    
    @property
    def minx(self):
        return self.min[0]
    
    @property
    def minz(self):
        return self.min[2]
    
    @property
    def maxx(self):
        return self.max[0]
    
    @property
    def maxz(self):
        return self.max[2]

    @property
    def lenx(self):
        return self.a[0]*2
    
    @property
    def lenz(self):
        return self.a[2]*2
    
    def __add__(self,other):
        return rect(c0=np.array([self.min,other.min]).min(axis=0),c1=np.array([self.max,other.max]).max(axis=0))

from ...Basic.Wall import two23, EPS
class spce():
    def __init__(self,pro,scene,delta,idx=-1):
        self.scene = scene
        self.rect = rect(c=pro.p,a=two23(pro.nearest))
        self.delta=delta
        
        # self.wallsSign = [[],[],[],[]]
        
        # w = pro.toWalls()
        # for i in range(4):
        #     #f = True
        #     for j in pro.areaF[i]:
        #         self.wallsSign[i].append([j[0],j[1],(j[1]<delta),w[i].rate(j[0]),w[i].rate(j[0])*w[i].length])
                #f = f and (j[1]<delta)
            #if f:
            #    onBds.append(i)
        #print(self.wallsSign)
        #print(onBds)
        #raise NotImplementedError

            
        #I,J = 0,1#-1,-1
        # # if pro is not None:
        # #     I,J = 1,2
        # if len(onBds)==1:
        #     J = onBds[0]
        #     I = (J-1)%4
        # elif len(onBds)==2:
        #     #assert abs(onBds[0] - onBds[1])==1
        #     I = 3 if onBds[0] == 0 and onBds[1] == 3 else onBds[0]
        #     J = 0 if onBds[0] == 0 and onBds[1] == 3 else onBds[1]
        # elif len(onBds)==3:
        #     j = [__ for __ in onBds if __ not in [_ for _ in range(4)]]
        #     J = (j[0]+2)%4
        #     I = (J-1)%4
        #这一段核心就是挑IJ，每一个IJ都向两边延伸一下，看看最长的有多长，其中别忘了这里的delta需要设置到门的厚度以上，用以屏蔽门的影响，此处设置为0.42吧
        # Area=-100
        # for j in range(4):
        #     area=0
        #     i = (j-1)%4
        #     idx = len(self.wallsSign[i])-1
        #     while idx > -1:
        #         if self.wallsSign[i][idx][1] < 0.42:
        #             area+=self.wallsSign[i][idx][-1]
        #         else:
        #             break
        #         idx -= 1

        #     idx = 0
        #     while idx < len(self.wallsSign[j]):
        #         if self.wallsSign[j][idx][1] < 0.42:
        #             area+=self.wallsSign[j][idx][-1]
        #         else:
        #             break
        #         idx += 1
        #     if area > Area:
        #         self.I,self.J = (j-1)%4,j
        #         Area = area
        self.pro = pro
        self.I, self.J = pro.AREF.I_J()
        #print(self.I)
        #print(self.J)
        #raise NotImplementedError
        #from ..Basic import obje
        #self.refObj = obje(self.corners[self.J],np.array([1,1,1]),np.array([(2-self.J)*np.math.pi/2])) #if self.J != -1 else  obje((self.corners[self.I]+self.corners[(self.I-1)%4])/2.0,np.array([1,1,1]),self.I*np.math.pi)
        #self.relA = np.array([self.a[0],self.a[1],self.a[2]]) if self.J%2==0 else np.array([self.a[2],self.a[1],self.a[0]])
        self.init_cen = self.corners[self.J] if self.J != -1 else self.corners[self.I]
        self.init_ori = [(2-self.J)*np.math.pi/2] if self.J != -1 else [self.I*np.math.pi]
        from ...Basic import obje
        self.refObj = obje(self.init_cen,np.array([1,1,1]),self.init_ori,i=0)
        self.grup = None
    
    #region: properties----------#
    @property
    def c(self):
        return self.rect.c
    
    @property
    def a(self):
        return self.rect.a

    @property
    def c0(self):
        return self.rect.min
    
    @property
    def c1(self):
        return self.rect.max

    @property
    def corners(self):
        return self.rect.corners
    #endregion: properties-------#

    def __str__(self):
        return "[[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f]] at [%.2f,%.2f] with [%.2f,%.2f]"%(self.corners[0][0],self.corners[0][2],self.corners[1][0],self.corners[1][2],self.corners[2][0],self.corners[2][2],self.corners[3][0],self.corners[3][2],self.c[0],self.c[2],self.a[0],self.a[2])

    def draw(self):
        from matplotlib import pyplot as plt
        scl = [1.0,0.8,0.6,0.4]
        c,a = self.c, self.a
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color="pink")

    #def toWalls(self,eps=0): from ...Basic import walls return walls(c=[self.c[0],self.c[2]],a=[self.a[0]-eps,self.a[2]-eps])

    #region: operations----------#
    def recycle(self):
        self.rect = rect(c0=self.init_cen,c1=2*self.grup.rect.c-self.init_cen)

        from .Prob import prob
        PRO = prob(self.c,self.delta,self.pro.WALLS)
        PRO.res = [(self.a[0]-EPS,True),(self.a[2]-EPS,True),(self.a[0]-EPS,True),(self.a[2]-EPS,True)]
        
        PRO.environ()#WALLS, self.delta)#, True,1)
        self.pro = PRO
        # self.wallsSign = [[],[],[],[]]
        
        # w = PRO.toWalls()
        # for i in range(4):
        #     for j in PRO.areaF[i]:
        #         self.wallsSign[i].append([j[0],j[1],(j[1]<self.delta),w[i].rate(j[0]),w[i].rate(j[0])*w[i].length])

    def viola(self,obj):
        #print("self.rect\n",self.rect)
        bd = obj.rect()
        #print("%s\n"%(obj.class_name),bd)
        grup_bd = self.grup.rect+bd
        #print("grup_bd\n",grup_bd)
        #x
        xmin_to = self.rect.minx-grup_bd.minx #-xmin_to:(>0)space(<0)invalid, max(+xmin_to,0.0): modification, -max(-xmin_to,0.0): space
        xmax_to = self.rect.maxx-grup_bd.maxx #+xmax_to:(>0)space(<0)invalid,-max(-xmax_to,0.0): modification, +max(+xmax_to,0.0): space
        if xmax_to-xmin_to < -self.delta*2: return None,None
        #print("xmin_to=%.3f xmax_to=%.3f"%(xmin_to,xmax_to))
        xmin_mv, xmin_sp = max( xmin_to,0.0), -max(-xmin_to,0.0)
        xmax_mv, xmax_sp =-max(-xmax_to,0.0),  max( xmax_to,0.0)
        #print("xmin_mv=%.3f xmin_sp=%.3f xmax_mv=%.3f, xmax_sp=%.3f"%(xmin_mv,xmin_sp,xmax_mv,xmax_sp))
        assert xmax_mv*xmin_mv >= 0
        if abs(xmin_mv) < abs(xmax_mv):
            x_all = xmax_mv if abs(xmax_mv) < abs(xmin_sp) else xmin_sp
            x_single = xmax_mv - x_all
        else:
            x_all = xmin_mv if abs(xmin_mv) < abs(xmax_sp) else xmax_sp
            x_single = xmin_mv - x_all
        
        #print("x_all=%.3f x_single=%.3f"%(x_all,x_single))
        
        #z
        zmin_to = self.rect.minz-grup_bd.minz
        zmax_to = self.rect.maxz-grup_bd.maxz
        if zmax_to-zmin_to < -self.delta*2: return None,None
        #print("zmin_to=%.3f zmax_to=%.3f"%(zmin_to,zmax_to))
        zmin_mv, zmin_sp = max( zmin_to,0.0), -max(-zmin_to,0.0)
        zmax_mv, zmax_sp =-max(-zmax_to,0.0),  max( zmax_to,0.0)
        #print("zmin_mv=%.3f zmin_sp=%.3f zmax_mv=%.3f, zmax_sp=%.3f"%(zmin_mv,zmin_sp,zmax_mv,zmax_sp))
        assert zmax_mv*zmin_mv >= 0
        if abs(zmin_mv) < abs(zmax_mv):
            z_all = zmax_mv if abs(zmax_mv) < abs(zmin_sp) else zmin_sp
            z_single = zmax_mv - z_all
        else:
            z_all = zmin_mv if abs(zmin_mv) < abs(zmax_sp) else zmax_sp
            z_single = zmin_mv - z_all        
        #print("z_all=%.3f z_single=%.3f"%(z_all,z_single))
        return np.array([x_all,0.0,z_all]), np.array([x_single,0.0,z_single])
        
    def drop(self,pm,nids):
        pm.exp_object(nids[0],self.scene,t=self.init_cen,ori=self.init_ori)
        from ..Plan import grup
        self.scene.GRUPS.append(grup([self.scene.OBJES[-1].idx]))
        self.grup = self.scene.GRUPS[-1]
        self.scene.draw(suffix=str(nids[0]))
        for nid in nids[1:]:
            new_o = pm.exp_object(nid,self.scene,add=False)
            movings,moving = self.viola(new_o)
            if movings is None:
                print("no space for "+str(nid))
                break #unsolved problem
            new_o.translation += moving
            self.scene.addObject(new_o)
            self.grup.append(self.scene.OBJES[-1].idx)
            self.scene.draw(suffix=str(nid)+".5")
            self.grup.move(movings)
            self.scene.draw(suffix=str(nid))
    #endregion: operations-------#

class spces():
    def __init__(self, scne=None, wals=None, name="", flex=1.2, sz=-4.0, drawFolder=""):
        from copy import deepcopy
        self.scene = scne
        #self.iWALLS= deepcopy(scne.WALLS if wals is None else wals) #a copy for visualization
        self.WALLS = scne.WALLS.inward_door()#deepcopy(scne.WALLS if wals is None else wals) #a local data structure for processing
        self.SPCES = []
        self.name = name
        self.flex = flex
        self.delta = 0.2
        self.drawFolder = drawFolder
        import os
        if drawFolder and not os.path.exists(drawFolder): os.makedirs(drawFolder)
    
    def __iter__(self):
        return iter(self.SPCES)

    def __getitem__(self, idx):
        return self.SPCES[idx]

    def __len__(self):
        return len(self.SPCES)

    #region: presentation-----------#
    def draw(self):#,dr=True):
        return [s.draw() for s in self.SPCES]
        #print("here "+folder)
        #print(self.SPCES[0])
        from PIL import Image,ImageOps
        from matplotlib import pyplot as plt
        [s.draw() for s in self.SPCES]
        if self.drawFolder and dr:#print("spces draaw", (folder if folder else self.drawFolder)+str(len(self.SPCES))+".png")
            self.scene.WALLS.draw(color="gray")
            self.WALLS.draw(color="black")
            f=self.drawFolder+str(len(self.SPCES))+".png"
            plt.axis('equal')
            L = max(self.LH)
            plt.xlim(-self.flex*L,self.flex*L)
            plt.ylim(-self.flex*L,self.flex*L)
            plt.savefig(f)
            ImageOps.invert(Image.merge('RGB', Image.open(f).split()[:3])).save(f)
            plt.clf()

    def drawProb(self, probArray, DIR="", order=0, aFlag=True):
        from PIL import Image,ImageDraw
        N,M = 1+2*int(self.flex*self.LH[0]/self.delta),1+2*int(self.flex*self.LH[1]/self.delta)
        #H = int(len(probArray)**0.5) #i*delta-L
        assert N*M == len(probArray)
        img = Image.new("RGB",(N,M))  
        img1 = ImageDraw.Draw(img)  
        for i in range(len(self.WALLS)):
            w = self.WALLS[i]
            if w.v: 
                img1.line([(int((w.p[0]+self.flex*self.LH[0])/self.delta),int((w.p[2]+self.flex*self.LH[1])/self.delta)),(int((w.q[0]+self.flex*self.LH[0])/self.delta),int((w.q[2]+self.flex*self.LH[1])/self.delta))],fill="white",width=4)
        
        pixels = img.load()
    
        for y in range(M):
            for x in range(N):
                pa = probArray[y*N+x]
                if len(pa):
                    p = pa[min(len(pa)-1,order)] if not aFlag else sorted(pa,key=lambda x:-x.area())[0]
                    near = p.nearest()
                    nearx = near[0]
                    nearz = near[1]
                    nearavg = (nearx+nearz)/2.0
                    nearMax = max(nearx,nearz)
                    nearMin = min(nearx,nearz)
                    ratio = p.ratio()
                    
                    pixels[x, y] = (int(nearx*50), int(ratio*50+nearavg*50), int(nearz*50))

        img.save((DIR if DIR else self.drawFolder)+str(len(self.SPCES))+' heat.png')

    def draww(self,suffix=""):
        from matplotlib import pyplot as plt
        import os
        plt.figure(figsize=(10, 8))
        self.WALLS.draw()
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(os.path.join(self.scene.imgDir,self.scene.scene_uid+"_SPCE_WALLS"+suffix+".png"))
        plt.clf()
        plt.close()
    #endregion: presentation--------#

    #region: operations:space-------#
    def eliminate_space(self,Spce,f=False):
        """
        #if f:
        #    print("wallsSign:\n"+'\n'.join([ "→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in ar[:-1]])+"(%.3f,%.3f)"%(ar[-1][0][0],ar[-1][0][2])  for ar in Spce.wallsSign ]))
        
        # from copy import deepcopy
        # #just check the off wall segments of the spce?
        # height = Spce.wallsSign[3][-1][1]
        # idx = [0,0]
        # while True:
        #     if Spce.wallsSign[idx[0]][idx[1]][1]<EPS and height>EPS:
        #         break
        #     height = Spce.wallsSign[idx[0]][idx[1]][1]
        #     if len(Spce.wallsSign[idx[0]]) > idx[1]+2:
        #         idx[1] += 1
        #     else:
        #         idx[1]=0
        #         idx[0]=(idx[0]+1)%4

        # IDX = [idx[0],idx[1]]
        # #print(IDX)
        # offWallSegments = []
        # currentSegment = None#[None,0,None,[]] #startingpoint, length, endingpoint, walls
        # tw = Spce.toWalls()
        # while True:
        #     height = Spce.wallsSign[idx[0]][idx[1]][1]
        #     if len(Spce.wallsSign[idx[0]]) > idx[1]+2:
        #         idx[1] += 1
        #     else:
        #         idx[1]=0
        #         idx[0]=(idx[0]+1)%4
        #         if currentSegment is not None and not (Spce.wallsSign[idx[0]][idx[1]][1]<EPS and height > EPS):
        #             P = Spce.wallsSign[idx[0]][idx[1]][0]
        #             currentSegment[3][-1].q = np.copy(P)
        #             #currentSegment[3][-1].lengthh()
        #             N = currentSegment[3][-1].n
        #             currentSegment[3].append(wall(P,P,np.copy(tw[idx[0]].n),-1,-1,-1))
            
        #     if Spce.wallsSign[idx[0]][idx[1]][1]<EPS*100 and height > EPS*100:
        #         #end an offWallSegment
        #         P = Spce.wallsSign[idx[0]][idx[1]][0]
        #         currentSegment[2] = deepcopy(P)
        #         currentSegment[3][-1].q = np.copy(P)
        #         #currentSegment[3][-1].lengthh()
        #         currentSegment[1] = np.sum([w.length for w in currentSegment[3]])
        #         offWallSegments.append(deepcopy(currentSegment))
        #         currentSegment = None
        #     elif Spce.wallsSign[idx[0]][idx[1]][1]>EPS*100 and height > EPS*100:
        #         P = Spce.wallsSign[idx[0]][idx[1]][0]
        #         currentSegment[3][-1].q = np.copy(P)
        #         #currentSegment[3][-1].lengthh()
        #         #stretch the offWallSegment
        #         pass
        #     elif Spce.wallsSign[idx[0]][idx[1]][1]>EPS*100 and height < EPS*100: 
        #         #start an offWallSegment
        #         P = Spce.wallsSign[idx[0]][idx[1]][0]
        #         currentSegment = [P,0,None,[wall(P,P,tw[idx[0]].n,-1,-1,-1)]] #startingpoint, length, endingpoint, walls
        #         pass

        #     if idx[0] == IDX[0] and idx[1] == IDX[1]:
        #         break 
        
        # if f:
        #     for c in offWallSegments:
        #         print("[%.3f, %.3f] -> %.3f -> [%.3f, %.3f]"%(c[0][0],c[0][2],c[1],c[2][0],c[2][2]))
        #         for w in c[3]:
        #             print(w)
        #         print("\n")
        """
        offWallSegments = Spce.pro.AREF.offWallSegment(f)

        A = sorted(offWallSegments,key=lambda x:-x[1])[0]

        if f:
            print("self.WALLS\n", self.WALLS)
        W = self.WALLS.searchWall(A[0],True,f)[0]
        a=W.idx
        #W.q = A[3][-1].q
        if f:
            print("W",W)
        X = self.WALLS.searchWall(A[2],False,f)[0]
        if f:
            print("X",X)
        W.p = A[3][0].p
        if X.idx == W.idx:
            a = self.WALLS.insertWall(w2 = W.idx)
            X = self.WALLS[a]
        X.q = A[3][-1].q
        W.w1 = X.idx
        X.w2 = W.idx
        if f:
            print("self.WALLS\n", self.WALLS)
        a=self.WALLS.insertWall(X.idx)
        A[3].reverse()
        if f:
            print("self.WALLS\n", self.WALLS)
        for w in A[3][:-1]:#
            self.WALLS[a].q = w.p #self.WALLS[a].resetN()
            a=self.WALLS.insertWall(a)
            if f:
                print("self.WALLS\n", self.WALLS)
        #X.p = A[3][-1].q
        #X.resetN()
        #X.w1 = a
        #self.WALLS[a].w2 = X.idx
        # print(self.WALLS)
        # print("\n")

        valid = [X.idx]
        b=X.w2
        while b!=X.idx:
            valid.append(b)
            b = self.WALLS[b].w2
        for w in self.WALLS:
            w.v = (w.idx in valid)

        if f:
            print("self.WALLS\n", self.WALLS)
        #raise NotImplementedError

        return True

    def extractingSpce(self,DIR="",hint=None):
        from .Prob import prob
        if len([w for w in self.WALLS if w.v])==0:
            return None
        N,M = int(self.flex*self.WALLS.lenx/(2.0*self.delta)),int(self.flex*self.WALLS.lenz/(2.0*self.delta))
        GRIDS = self.delta*np.array([[[i,j] for i in range(-N,N+1)] for j in range(-M,M+1)]).reshape((-1,2)) + np.array([self.WALLS.x,self.WALLS.z]).reshape((1,2))
        GRIDPro = []
        for loc in GRIDS:
            GRIDPro.append([])
            pro = prob(two23(loc),self.delta,self.WALLS)
            pro.updates()
            # for w in self.WALLS:
            #     if w.v and w.over(pro.p):#print("not over")
            #         pro.update(w)#dis,vec = w.distance(pro.p)#print(dis)#print(vec)
                
            i,f = pro.status
            if not i or min(pro.nearest) < 0.4 or pro.ratio < 0.4: continue# or pro.area()<EPS/10
            if not f: print("warning :inconsist in or out status",pro,sep="\n")#,GRIDPro.index(pro))
                #assert f

            if not self.WALLS.cross(pro.toWalls(EPS)):
                #pro.adjust(self.delta) #pro = self.tinyAdjustProb(pro)
                pro.environ()#pro.areaFunctionDetection(self.WALLS,self.delta)
                GRIDPro[-1].append(pro)
            else:
                #print(pro.p,self.WALLS,sep="\n")
                for xi in range(len(self.WALLS)):
                    x = self.WALLS[xi]
                    if x.v and pro.inner(x)[1]:
                        for w in self.WALLS[xi+1:]:
                            if w.v and abs(w.n@x.n)<EPS and pro.inner(w)[1]:
                                qro = prob(pro.p, self.delta, self.WALLS)
                                qro.update(w) #dis,vec=w.distance(qro.p)
                                qro.update(x) #dis,vec=x.distance(qro.p)
                                #qro.adjust(self.delta) #qro = self.tinyAdjustProb(qro)print(qro.toWalls(),qro.toWalls(EPS),sep="\n") #,w,x,pro
                                if min(qro.nearest)>EPS*2 and not self.WALLS.cross(qro.toWalls(EPS/2)):#qro.area()>EPS/10 and 
                                    # print(w,x,qro,"yes",sep="\n")
                                    qro.environ()#qro.areaFunctionDetection(self.WALLS,self.delta)
                                    GRIDPro[-1].append(qro)
                #assert len(GRIDPro[-1])
            
        if DIR or self.drawFolder:
            self.drawProb(GRIDPro,(DIR if DIR else self.drawFolder))

        #Find a pro in pros

        #return sorted([sorted(gg,key=lambda x:-x.key(self.delta,hint))[0] for gg in GRIDPro if len(gg)],key=lambda x:-x.key(self.delta,hint))
        PRO = sorted([sorted(gg,key=lambda x:-x.key(hint))[0] for gg in GRIDPro if len(gg)],key=lambda x:-x.key(hint))[0]
        
        #print(self.WALLS,PRO,PRO.toWalls(),sep="\n")
        #PRO.printConnectivityInfo()
        #PRO.areaFunctionDetection(self.WALLS,self.delta,True)
        #PRO.printConnectivityInfo(self.delta)
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n") #raise NotImplementedError
        #PRO.adjust(self.delta) # PRO = self.tinyAdjustProb(PRO)
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n")
        #raise NotImplementedError
        #print(PRO)

        return spce(PRO,self.scene,self.delta)

    def recycle(self,i=-1):
        self.SPCES[i].recycle()
        self.scene.draw(suffix="rec")
        self.eliminate_space(self.SPCES[i],False)
        self.draww(suffix="1")
    #endregion: operations:space----#

    #region: operations:synthesis---#
    def drop(self,pm,lst=[54,3]): #122
        from ...Operation.Patn import pathses
        from random import randint
        paths = pathses(pm)
        #print(self.scene.imgDir)

        for i,root in enumerate(lst):
            path = paths.paths[root]#randint(paths.segments[root][0],paths.segments[root][1])]
            leaf_scene = pm.leaf_scene(path.path[-1])
            print(leaf_scene.GRUPS[0].size)
            #（1）采样第一条路径 #要边采样边检测空间吗？肯定不要， print("!!!")
            
            #print(self.scene.imgDir)
            #（2）利用路径检测空间
            self.draww(suffix=str(i))
            f = self.extractingSpce(hint=leaf_scene.GRUPS[0].size) #陈年老代码，不知道还能跑吗
            if f is None: return
            self.SPCES.append(f)
            print(self.scene.SPCES[-1])
            #print(self.scene.imgDir)

            #（3）逐物体向空间中添加并整体移动
            self.scene.SPCES[-1].drop(pm,path.path)

            #（4）如果所有物体完全放置了就回收空间，形成一个新的walls
            if i < len(lst)-1: self.recycle()

    #endregion: operations:synthesis#

    # def extractingMoreSpces(self,DIR="",hint=None):#[spce(np.array([-1.8595,0.,-1.332]),np.array([3.3795,0.,1.252]),scne=self)]#
    #     spss = self.extractingSpce()#[None]#
    #     for p in spss[:20]:
    #         #p = prob(np.array([2.84,0,-0.04]))
    #         #p.res = [(0.1385,True),(1.3065,True),(6.2185,True),(1.2910,True)] 
    #         #print(p)
    #         #print(spss.index(p))
    #         #p.printConnectivityInfo(self.delta)
    #         #print("\n")
    #         p.areaFunctionDetection(self.WALLS,self.delta,False)
    #         #print("\n")
    #         #p.printConnectivityInfo(self.delta)
    #         print(str(spss.index(p)/10)+": [p.key() = %.3f ]= [(-1.0 / ratio) = %.3f] + [np.average(self.nearest())*2 = %.3f] -[self.separation()*5 = %.3f] + [self.onWallLength(delta) = %.3f]"%(p.key(self.delta,hint), -1.0/p.ratio(),np.average(p.nearest())*2,p.separation()*5,p.onWallLength(self.delta)))
    #         #print(self.WALLS)
    #         sp = spce(p,self.scene,self.delta)
    #         self.SPCES.append(sp)
    #         sp.scene = self.scene
    #         self.draw(self.drawFolder+" testings "+str(spss.index(p))+" ")
    #         self.SPCES = []

    # def extractingSpces(self,bound=1,DIR=""):
    #     raise AssertionError("this function is not used anymore")
    #     self.SPCES,b = [],self.draw((DIR if DIR else self.drawFolder)) if DIR or self.drawFolder else None
    #     while len(self.SPCES)<bound:
    #         sp = self.extractingSpce(DIR)
    #         self.SPCES.append(sp)
    #         sp.scene = self.scene
    #         a = self.eliminate_space(sp)
    #         b = self.draw((DIR if DIR else self.drawFolder)) if DIR or self.drawFolder else None
    #         if not a:
    #             break
    #         #sp = self.extractingSpce((DIR if DIR else self.drawFolder))

    # def tinyAdjustProb(self,Prob):
    #         if abs(Prob.res[0][0]-Prob.res[2][0])<self.delta*2:
    #             a = (Prob.res[0][0]+Prob.res[2][0])/2.0
    #             b = (Prob.res[0][0]-Prob.res[2][0])/2.0
    #             Prob.res[0],Prob.res[2] = (a,True),(a,True)
    #             Prob.p[0] += b
    #         if abs(Prob.res[1][0]-Prob.res[3][0])<self.delta*2:
    #             a = (Prob.res[1][0]+Prob.res[3][0])/2.0
    #             b = (Prob.res[1][0]-Prob.res[3][0])/2.0
    #             Prob.res[1],Prob.res[3] = (a,True),(a,True)
    #             Prob.p[2] += b
    #         return Prob
