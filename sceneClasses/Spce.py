# from . import SPCES,WALLS
from Wall import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def stickWall(w,x):
    wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
    return w.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

class bond():
    def __init__(self,p,q,idx,spceIdx):
        self.p=p
        self.q=q
        self.n = np.cross(np.array([0,1,0]),p-q)/np.linalg.norm(np.cross(np.array([0,1,0]),p-q))
        #print(self.n) #assert (p[0]-q[0])*self.n[2]<=(p[2]-q[2])*self.n[0]
        self.length=(((self.p-self.q)**2).sum())**0.5
        self.idx=idx
        self.spceIdx=spceIdx
        self.relatedWalls = []
        self.full = False
        #self.checkWalls()

    def addWall(self,w):
        #(self.p-w.p)@(w.q-w.p)/w.length  
        # print("addWall "+str(self.idx)+" "+str(w.idx))
        # print(self.p)
        # print(self.q)
        # print(self.n)
        # print(w.p)
        # print(w.q)
        # print(w.n)
        # print("low")
        # print((self.p-w.p)@(w.q-w.p)/(w.length)**2)

                

        low = max((self.p-w.p)@(w.q-w.p)/(w.length)**2,0.01)
        # print(low)
        # print("upp")
        # print((self.q-w.p)@(w.q-w.p)/(w.length)**2)
        upp = min((self.q-w.p)@(w.q-w.p)/(w.length)**2,0.99)
        # print(upp)
        # print("lower")
        # print((w.p-self.p)@(self.q-self.p)/(self.length)**2)
        lower = max((w.p-self.p)@(self.q-self.p)/(self.length)**2,0.01)
        # print(lower)
        # print("upper")
        # print((w.q-self.p)@(self.q-self.p)/(self.length)**2)
        upper = min((w.q-self.p)@(self.q-self.p)/(self.length)**2,0.99)
        # print(upper)
        self.relatedWalls.append({"idx":w.idx,"low":low,"upp":upp,"lower":lower,"upper":upper})
        if low==0.01 and upp==0.99 and lower==0.01 and upper==0.99 :
            self.full=True

    def checkWalls(self):
        for w in WALLS:
            if stickWall(w,self):
                self.addWall(w)
        pass

    def mWall(self, L, moveFull=True):
        oldp = np.copy(self.p)
        self.p += self.n*L
        oldq = np.copy(self.q)
        self.q += self.n*L
        self.length=(((self.p-self.q)**2).sum())**0.5
        SPCES[self.spceIdx].bounds[(self.idx-1)%4].adjustWall(oldp,self.p,self.idx)
        SPCES[self.spceIdx].bounds[(self.idx+1)%4].adjustWall(oldq,self.q,self.idx)
        if moveFull and self.full:
            WALLS[self.relatedWalls[0]["idx"]].mWall(L)

    def adjustWall(self,oldp,p,hint):
        if hint == (self.idx-1)%4:
            self.p=p
        elif hint == (self.idx+1)%4:
            self.q=p
        else:
            print("false hint")
        if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
            self.n = -self.n
        self.length=(((self.p-self.q)**2).sum())**0.5

#WALLS=[]
class prob():
    def __init__(self,p):
        self.p=p
        self.res=[(1000,False),(1000,False),(1000,False),(1000,False)]
    
    def __str__(self):
        return "[%.2f,%.2f]: xmin: %.2f %s , zmin: %.2f %s , xmax: %.2f %s , zmax: %.2f %s"%(self.p[0],self.p[2],self.res[0][0],("in" if self.res[0][1] else "out"),self.res[1][0],("in" if self.res[1][1] else "out"),self.res[2][0],("in" if self.res[2][1] else "out"),self.res[3][0],("in" if self.res[3][1] else "out"))
    
    def update(self,dis,vec):
        if abs(vec[2])<EPS/10: 
            if vec[0]<EPS:
                if abs(dis)-EPS < self.res[0][0]:
                    self.res[0] = (abs(dis)-EPS,(dis>-EPS))
            if vec[0]>-EPS:
                if abs(dis)-EPS < self.res[2][0]:
                    self.res[2] = (abs(dis)-EPS,(dis>-EPS))
        if abs(vec[0])<EPS/10:
            
            if vec[2]<EPS:
                if abs(dis)-EPS < self.res[1][0]:
                    self.res[1] = (abs(dis)-EPS,(dis>-EPS))
            if vec[2]>-EPS:
                if abs(dis)-EPS < self.res[3][0]:
                    self.res[3] = (abs(dis)-EPS,(dis>-EPS))
        if abs(vec[0])>EPS/10 and abs(vec[2])>EPS/10:
            print("not vertical or horizontal wall, currently not supported by space detection or prob-update")
            raise NotImplementedError
        
    def status(self):
        return self.res[0][1], (self.res[0][1] == self.res[1][1]) and (self.res[1][1] == self.res[2][1]) and (self.res[2][1] == self.res[3][1])

    def nearest(self):
        return [min(self.res[0][0],self.res[2][0])+EPS,min(self.res[1][0],self.res[3][0])+EPS]
    
    def ratio(self):
        a = self.nearest()
        return a[0]/max(a[1],EPS/10) if a[0]<a[1] else a[1]/max(a[0],EPS/10)

    def key(self):
        return self.ratio()+np.average(self.nearest())*0.4 

class spce():
    def __init__(self,c0,c1,idx=-1):
        self.c0 = np.min(np.array([c0,c1]),axis=0)
        self.c1 = np.max(np.array([c0,c1]),axis=0)
        self.c = (c0+c1)/2.0
        self.a = self.c1 - self.c
        self.corners = [self.c0,np.array([self.c1[0],self.c1[1],self.c0[2]]),self.c1,np.array([self.c0[0],self.c0[1],self.c1[2]])]
        
        self.bounds = []
        self.bounds.append(bond(self.c0,np.array([self.c1[0],self.c1[1],self.c0[2]]),0,idx))
        self.bounds.append(bond(np.array([self.c1[0],self.c1[1],self.c0[2]]),self.c1,1,idx))
        self.bounds.append(bond(self.c1,np.array([self.c0[0],self.c0[1],self.c1[2]]),2,idx))
        self.bounds.append(bond(np.array([self.c0[0],self.c0[1],self.c1[2]]),self.c0,3,idx))
        pass

    @classmethod
    def fromProb(cls,c,a):
        return cls(c-a,c+a)

    def __str__(self):
        return "[[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f]] at [%.2f,%.2f] with [%.2f,%.2f]"%(self.corners[0][0],self.corners[0][2],self.corners[1][0],self.corners[1][2],self.corners[2][0],self.corners[2][2],self.corners[3][0],self.corners[3][2],self.c[0],self.c[2],self.a[0],self.a[2])

    def maxZ(self):
        return max(self.corner0[2],self.corner1[2])

    def minZ(self):
        return min(self.corner0[0],self.corner1[0])
    
    def maxX(self):
        return max(self.corner0[2],self.corner1[2])
    
    def minX(self):
        return min(self.corner0[0],self.corner1[0])

    def absoluteBbox(self):
        pass

    def recommendedWalls(self):
        #we are going 
        pass

    def adjustSingle(self,id):
        if id > 3 or (not self.bounds[id].full):
            print("error:noFullBound?")
            return 
        L = 0
        if self.bounds[0].length/self.bounds[1].length < 0.6:
            if id%2 == 0:
                L = self.bounds[1].length - self.bounds[0].length/0.6 #positive
                #reduce myself to reduce self.bounds[1].length
                pass
            else:
                L = self.bounds[0].length - self.bounds[1].length*0.6 #negative
                #add myself to add self.bounds[0].length
                pass
        else:#self.bounds[0].length/self.bounds[1].length > 1.6
            if id%2 == 0:
                L = self.bounds[1].length - self.bounds[0].length/1.6 #negative
                #add myself to add self.bounds[1].length
                pass
            else:
                L = self.bounds[0].length - self.bounds[1].length*1.6 #positive
                #reduce myself to reduce self.bounds[0].length
                pass
            pass 
        self.bounds[id].mWall(L)

    def adjustSize(self):
        raise NotImplementedError
        if abs(self.bounds[0].length/self.bounds[1].length-1.0)<0.4:
            return
        if (self.bounds[0].full or self.bounds[2].full) and (self.bounds[1].full or self.bounds[3].full):
            if(self.bounds[0].length>self.bounds[1].length):
                if self.bounds[0].full:
                    self.adjustSingle(0)
                elif self.bounds[2].full:
                    self.adjustSingle(2)
            else:
                if self.bounds[1].full:
                    self.adjustSingle(1)
                elif self.bounds[3].full:
                    self.adjustSingle(3)
        else:
            id=0
            while id < 4 and (not self.bounds[id].full):
                id += 1
            self.adjustSingle(id)

        #check my own size,
        #find an full-covered wall
        #drag that wall to a propriate place
        # self.c0 = self.bounds[0].p
        # self.c1 = self.bounds[1].q
        # self.c = (self.c0+self.c1)/2.0
        # self.a = self.c0 - self.c
        pass

    def draw(self):
        scl = [1.0,0.8,0.6,0.4]
        c,a = self.c, self.a
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color="pink")


class spces():
    def __init__(self, scne=None, wals=None, name=""):
        self.scne = scne
        self.iWALLS= deepcopy(scne.WALLS if wals is None else wals) #a copy for visualization
        self.WALLS = deepcopy(scne.WALLS if wals is None else wals) #a local data structure for processing
        self.SPCES = []
        self.name = name
        self.L = 5
        self.delta = 0.04
    
    def draw(self,folder=""):
        [s.draw() for s in self.SPCES]
        if folder:
            self.iWALLS.draw(color="gray")
            self.WALLS.draw(color="black")
            
            plt.axis('equal')
            plt.xlim(-self.L,self.L)
            plt.ylim(-self.L,self.L)
            plt.savefig(folder+str(len(self.SPCES))+".png")
            plt.clf()

    def eliminatingSpace(self, Spce):
        #addSpace's walls#print(Spce)#print(self.WALLS)
        PID,W=None,None
        for pid in range(4):
            for w in [w for w in self.WALLS if w.v]:
                #print(w.q-Spce.corners[pid])#print(np.linalg.norm(w.q-Spce.corners[pid]))
                if np.linalg.norm(w.q-Spce.corners[pid])<EPS*5:
                    W=w
                    break
            if W:
                PID=pid
                break

        X=self.WALLS[W.w2]
        a=W.idx
        for i in range(3,-1,-1):
            X.p = Spce.corners[(PID+i)%4]
            a=self.WALLS.insertWall(a)

        self.WALLS.minusWall(W.idx)
        if X.v:
            self.WALLS.minusWall(X.w1)
        #print(self.WALLS)

    def extractingSpce(self,DIR=""):
        if len([w for w in self.WALLS if w.v])==0:
            return None
        #grid on the space
        N = int(self.L/self.delta)
        GRIDS = self.delta*np.array([[[i,j] for i in range(-N,N+1)] for j in range(-N,N+1)]).reshape((-1,2))
        GRIDPro = []
        for loc in GRIDS:
            pro = prob(two23(loc))
            for w in self.WALLS:
                if w.v and w.over(pro.p):#print("not over")
                    dis,vec = w.distance(pro.p)#print(dis)#print(vec)
                    pro.update(dis,vec)
                
            i,f = pro.status()
            GRIDPro.append(pro)
            if not f:
                print(pro)
                print(GRIDPro.index(pro))
                assert f
            #break
        if DIR:
            self.drawProb(GRIDPro,DIR)

        #Find a pro in pros
        PRO = sorted([g for g in GRIDPro if g.status()[0]],key=lambda x:-x.key())[0]
        if abs(PRO.res[0][0]-PRO.res[2][0])<0.05:
            a = (PRO.res[0][0]+PRO.res[2][0])/2.0
            b = (PRO.res[0][0]-PRO.res[2][0])/2.0
            PRO.res[0],PRO.res[2] = (a,True),(a,True)
            PRO.p[0] += b
        if abs(PRO.res[1][0]-PRO.res[3][0])<0.05:
            a = (PRO.res[1][0]+PRO.res[3][0])/2.0
            b = (PRO.res[1][0]-PRO.res[3][0])/2.0
            PRO.res[1],PRO.res[3] = (a,True),(a,True)
            PRO.p[2] += b
        return spce.fromProb(PRO.p,two23(PRO.nearest()))

    def extractingSpces(self,DIR="",bound=1):
        self.SPCES,sp = [],self.extractingSpce(DIR)
        while sp and len(self.SPCES)<bound:
            self.SPCES.append(sp)
            self.eliminatingSpace(sp)
            if DIR:
                self.draw(DIR+self.name+"/")
            sp = self.extractingSpce(DIR)

    def drawProb(self, probArray, DIR):
        from PIL import Image, ImageDraw
        global L
        global delta

        H = int(len(probArray)**0.5) #i*delta-L
        assert H*H == len(probArray)
        img = Image.new("RGB",(H,H))  
        img1 = ImageDraw.Draw(img)  
        for i in range(len(self.WALLS)):
            w = self.WALLS[i]
            if w.v: 
                img1.line([(int((w.p[0]+self.L)/self.delta),int((w.p[2]+self.L)/self.delta)),(int((w.q[0]+self.L)/self.delta),int((w.q[2]+self.L)/self.delta))],fill="white",width=4)
        
        pixels = img.load()
    
        for y in range(H):
            for x in range(H):
                p = probArray[y*H+x]
                if p.status()[0]:
                    near = p.nearest()
                    nearx = near[0]
                    nearz = near[1]
                    nearavg = (nearx+nearz)/2.0
                    nearMax = max(nearx,nearz)
                    nearMin = min(nearx,nearz)
                    ratio = p.ratio()
                    
                    pixels[x, y] = (int(nearx*50), int(ratio*50+nearavg*50), int(nearz*50))
    
        img.save(DIR+self.name+"/"+str(len(self.SPCES))+' heat.png')

import sys,argparse,os
def parse(argv):
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-i','--identity', default="-1")
    parser.add_argument('-n','--new', default=False, action="store_true")
    parser.add_argument('-b','--bound', default=3)
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    DIR,args = "./newRoom/",parse(sys.argv[1:])
    if int(args.identity) == -1:
        for i in range(args.bound):#print(i)
            if args.new:
                wls = walls(name="rand"+str(i))#print(wls.LOGS)
                wls.randomWalls()
                wls.output(DIR)#print("ok")
            wlz = walls.fromLog(f=DIR+"rand"+str(i)+".txt",name="rand"+str(i)+"_") #wlz.draw(DIR)
            if not os.path.exists(DIR+wlz.name):
                os.makedirs(DIR+wlz.name)
            sm = spces(wals=wlz,name=wlz.name)
            sm.extractingSpces(DIR,2)#sm.draw(DIR)
    else:
        wlz = walls.fromLog(f=DIR+"rand"+args.identity+".txt",name="rand"+args.identity+"_") #wlz.draw(DIR)
        if not os.path.exists(DIR+wlz.name):
            os.makedirs(DIR+wlz.name)
        sm = spces(wals=wlz,name=wlz.name)
        sm.extractingSpces(DIR,2)#sm.extractingSpce()
    



# def adjacent(s,Qbox):
#     #还有一些问题就是如果t的展开是可量化的。那能不能直接以s为限制去给出t的量化指标。
#     #比如说，t的一个端点是固定的，另一个端点还未限定，那么由s给出t的另一个端点的范围指标。
#     #
#     Actions = {}
#     if (Qbox["signX"] > 0 and s.minX() > Qbox["fixedX"] and Qbox["maxX"] > s.minX()):
#         Actions["maxX"] = s.minX()

#     if (Qbox["signX"] < 0 and s.maxX() < Qbox["fixedX"] and Qbox["minX"] < s.maxX()):
#         Actions["minX"] = s.maxX()

#     if (Qbox["signZ"] > 0 and s.minZ() > Qbox["fixedZ"] and Qbox["maxZ"] > s.minZ()):
#         Actions["maxZ"] = s.minZ()

#     if (Qbox["signZ"] < 0 and s.maxZ() < Qbox["fixedZ"] and Qbox["minZ"] > s.maxZ()):
#         Actions["minZ"] = s.maxZ()

#     if len(Actions.keys())>0:
#         K = Actions.keys()[0]
#         Square = -1
#         for k in Actions.keys():
#             Pbox = deepcopy(Qbox)
#             Pbox[k] = Actions[k]
#             square = (Pbox["maxX"]-Pbox["minX"])*(Pbox["maxX"]-Pbox["minX"])
#             if square > Square:
#                 Square = square
#                 K = k
#         Qbox[K] = Actions[K]

#     #{"maxX":???,"minX":???,"maxZ":???, "minZ":???}
#     return Qbox