# from . import SPCES,WALLS
from Wall import *
from Obje import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from PIL import Image,ImageOps,ImageDraw

# def stickWall(w,x):
#     wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
#     return w.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

# class bond():
#     def __init__(self,p,q,idx,spceIdx):
#         self.p=p
#         self.q=q
#         self.n = np.cross(np.array([0,1,0]),p-q)/np.linalg.norm(np.cross(np.array([0,1,0]),p-q))
#         #print(self.n) #assert (p[0]-q[0])*self.n[2]<=(p[2]-q[2])*self.n[0]
#         self.length=(((self.p-self.q)**2).sum())**0.5
#         self.idx=idx
#         self.spceIdx=spceIdx
#         self.relatedWalls = []
#         self.full = False
#         #self.checkWalls()

#     def addWall(self,w):
#         #(self.p-w.p)@(w.q-w.p)/w.length  
#         # print("addWall "+str(self.idx)+" "+str(w.idx))
#         # print(self.p)
#         # print(self.q)
#         # print(self.n)
#         # print(w.p)
#         # print(w.q)
#         # print(w.n)
#         # print("low")
#         # print((self.p-w.p)@(w.q-w.p)/(w.length)**2)

                

#         low = max((self.p-w.p)@(w.q-w.p)/(w.length)**2,0.01)
#         # print(low)
#         # print("upp")
#         # print((self.q-w.p)@(w.q-w.p)/(w.length)**2)
#         upp = min((self.q-w.p)@(w.q-w.p)/(w.length)**2,0.99)
#         # print(upp)
#         # print("lower")
#         # print((w.p-self.p)@(self.q-self.p)/(self.length)**2)
#         lower = max((w.p-self.p)@(self.q-self.p)/(self.length)**2,0.01)
#         # print(lower)
#         # print("upper")
#         # print((w.q-self.p)@(self.q-self.p)/(self.length)**2)
#         upper = min((w.q-self.p)@(self.q-self.p)/(self.length)**2,0.99)
#         # print(upper)
#         self.relatedWalls.append({"idx":w.idx,"low":low,"upp":upp,"lower":lower,"upper":upper})
#         if low==0.01 and upp==0.99 and lower==0.01 and upper==0.99 :
#             self.full=True

#     def checkWalls(self):
#         for w in WALLS:
#             if stickWall(w,self):
#                 self.addWall(w)
#         pass

#     def mWall(self, L, moveFull=True):
#         oldp = np.copy(self.p)
#         self.p += self.n*L
#         oldq = np.copy(self.q)
#         self.q += self.n*L
#         self.length=(((self.p-self.q)**2).sum())**0.5
#         SPCES[self.spceIdx].bounds[(self.idx-1)%4].adjustWall(oldp,self.p,self.idx)
#         SPCES[self.spceIdx].bounds[(self.idx+1)%4].adjustWall(oldq,self.q,self.idx)
#         if moveFull and self.full:
#             WALLS[self.relatedWalls[0]["idx"]].mWall(L)

#     def adjustWall(self,oldp,p,hint):
#         if hint == (self.idx-1)%4:
#             self.p=p
#         elif hint == (self.idx+1)%4:
#             self.q=p
#         else:
#             print("false hint")
#         if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
#             self.n = -self.n
#         self.length=(((self.p-self.q)**2).sum())**0.5

#WALLS=[]
class prob():
    def __init__(self,p):
        self.p=np.copy(p)
        self.res=[(20,False),(20,False),(20,False),(20,False)]
        self.areaF = []
        self.Us=[]
        self.straight = []
    
    def __str__(self):
        return "[%.4f,%.4f]: xmin: %.4f %s , zmin: %.4f %s , xmax: %.4f %s , zmax: %.4f %s"%(self.p[0],self.p[2],self.res[0][0],("in" if self.res[0][1] else "out"),self.res[1][0],("in" if self.res[1][1] else "out"),self.res[2][0],("in" if self.res[2][1] else "out"),self.res[3][0],("in" if self.res[3][1] else "out"))
    
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
            print(str(vec)+"not vertical or horizontal wall, currently not supported by space detection or prob-update")
            raise NotImplementedError
        
    def status(self):
        return self.res[0][1], (self.res[0][1] == self.res[1][1]) and (self.res[1][1] == self.res[2][1]) and (self.res[2][1] == self.res[3][1])

    def nearest(self):
        return [min(self.res[0][0],self.res[2][0])+EPS,min(self.res[1][0],self.res[3][0])+EPS]
    
    def ratio(self):
        a = self.nearest()
        return a[0]/max(a[1],EPS/10) if a[0]<a[1] else a[1]/max(a[0],EPS/10)

    def area(self):
        return self.nearest()[0]*self.nearest()[1]
        
    def key(self,delta,hint=None):
        if hint is None:
            return -1.0/self.ratio()+np.average(self.nearest())*2 - self.separation()*5 + self.onWallLength(delta)
        else:
            return -1.0/self.ratio()+np.average(self.nearest())*2 - self.separation()*5 + self.onWallLength(delta)
        
    def areaFunctionDetection(self, walls, delta, f=False, j0=2):
        assert not walls.crossCheck(self.toWalls(EPS/2))#delta))#
        wals = self.toWalls()
        if f:
            print(self.p[0]+self.nearest()[0])
            print(self.p[0]-self.nearest()[0])
            print(wals)    
        self.areaF = [ [ [wals[0].p,200],[wals[0].q,-1]],[[wals[1].p,200],[wals[1].q,-1]],[[wals[2].p,200],[wals[2].q,-1]],[[wals[3].p,200],[wals[3].q,-1]] ]
        for i in range(len(walls)):
            w = walls[i]
            if not w.v:
                continue
            j = [j for j in range(4) if (wals[j].n@w.n > 1-EPS)]
            j = j[0]
            d = (self.areaF[j][0][0]-w.p)@w.n#print(d)
            if d < -EPS:
                continue
            else:
                d = max(d,0)
            if f and j == j0:
                print(i,w,j,wals[j],"matched",sep="\t")
            
            #find the area for w.p and test if we need to set a breakpoint on w.p
            k = 0
            P = self.areaF[j][k][0]
            r = wals[j].rate(P)
            while r<wals[j].rate(w.p)-EPS and k < len(self.areaF[j]):
                k+= 1
                P = None if k==len(self.areaF[j]) else self.areaF[j][k][0] 
                r = 1000 if k==len(self.areaF[j]) else wals[j].rate(P)
            #w.p is in the area between self.areaF[j][k-1][0]=P + epsilon and self.areaF[j][k][0] + epsilon  self.areaF[j][k-1][0]< w.p <=self.areaF[j][k][0] ::: k+1 always exist, which means k<len()-1
            #k = 0: means  w.p <=self.areaF[j][0][0] = 0
            #it means w.p starts at somewhere before wals[j].p or at wals[j].p
            #k = k: means self.areaF[j][k-1][0]< w.p <=self.areaF[j][k][0]
            #it means w.p starts at somewhere in this area, maybe at the back of this area
            #k =len: self.areaF[j][-1][0] < w.p
            if k!= 0 and k != len(self.areaF[j]): #w surely starts in the area of wals[j] means: wals[j].q <     w.p
                if wals[j].rate(w.p) < r-EPS: #w.p is not the same as self.areaF[j][k][0], it's surely in the area; so we should set a breakpoint for it
                    R = wals[j].rate(w.p)
                    di = self.areaF[j][k-1][1]
                    self.areaF[j].insert(k,[wals[j].p*(1-R)+wals[j].q*R,di])
            kP = k
            if f and j == j0:
                print("wall %d start from [%.3f, %.3f], it start in k=%d"%(w.idx,w.p[0],w.p[2],kP))
                print(k)
                print(P)
                #print(self.areaF[j][k][0])
                #print(wals[j].rate(P))
                print(r)
                print(wals[j].rate(self.areaF[j][k-1][0]))
                print(wals[j].rate(w.p))

            #find the area for w.q and test if we need to set a breakpoint on w.q
            k = max(k-1,0)
            P = self.areaF[j][k][0]
            r = wals[j].rate(P)
            while r<wals[j].rate(w.q)-EPS and k < len(self.areaF[j]):
                k+= 1
                P = None if k==len(self.areaF[j]) else self.areaF[j][k][0]
                r = 1000 if k==len(self.areaF[j]) else wals[j].rate(P)
            #w.q is in the area between self.areaF[j][k-1][0] + epsilon and self.areaF[j][k][0]=P + epsilon  self.areaF[j][k-1][0]< w.q <= self.areaF[j][k][0]
            #k = 0: means  w.q <= self.areaF[j][0][0] = 0
            #it means w.p starts at somewhere before wals[j].p or at wals[j].p
            #k = k: means self.areaF[j][k-1][0]< w.q <=self.areaF[j][k][0]
            #it means w.p starts at somewhere in this area, maybe at the back of this area
            #k =len: self.areaF[j][-1][0] < w.q
            if k != 0 and k != len(self.areaF[j]): #w surely ends in the area of wals[j] means: self.areaF[j][k-1][0]< w.q <=self.areaF[j][k][0]
                if wals[j].rate(w.q) < r-EPS: #w.q is not the same as self.areaF[j][k][0], it's surely in the area; so we should set a breakpoint for it
                    R = wals[j].rate(w.q)
                    di = self.areaF[j][k-1][1]
                    self.areaF[j].insert(k,[wals[j].p*(1-R)+wals[j].q*R,di])
            kQ = k
            if f and j == j0:
                print("wall %d ends at [%.3f, %.3f], it ends at k=%d"%(w.idx,w.q[0],w.q[2],kQ))
                print(k)
                print(P)
                #print(self.areaF[j][k][0])
                #print(wals[j].rate(P))
                print(r)
                print(wals[j].rate(self.areaF[j][k-1][0]))
                print(wals[j].rate(w.q))
                if kP<kQ:
                    print("d is %.3f"%(d))
                
            #traverse from w.p's area to w.q's area
            kk = kP
            for t in range(kP,kQ):
                if self.areaF[j][kk][1] > d:
                    if kk>1 and abs(self.areaF[j][kk-1][1]-d)<EPS: #merge it with the area before me
                        del self.areaF[j][kk]
                    else: #cover this area
                        self.areaF[j][kk][1]=d
                        kk+=1

            if f and j == j0:
                print("→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in self.areaF[j][:-1]])+"(%.3f,%.3f)"%(self.areaF[j][-1][0][0],self.areaF[j][-1][0][2]))
            
            if False:

                k = 0
                P = self.areaF[j][k][0]
                r = wals[j].rate(P)
                while r<wals[j].rate(w.p)+EPS and k < len(self.areaF[j])-1:
                    k+= 1
                    P = self.areaF[j][k][0]
                    r = wals[j].rate(P)
                
                if f and j == j0:
                    print(k)
                    print(P)
                    print(self.areaF[j][k][0])
                    print(wals[j].rate(P))
                    print(r)
                    print(wals[j].rate(self.areaF[j][k-1][0]))
                    print(wals[j].rate(w.p))
                if k == len(self.areaF[j])-1:
                    if f and j == j0:
                        print("w.p is behind wals[j].q, make no effect to me")
                    continue
                
                if k>0 and abs(wals[j].rate(self.areaF[j][k-1][0])-wals[j].rate(w.p))> EPS: #P is behind w.q, #stop here
                    R = wals[j].rate(w.p)
                    di = self.areaF[j][k-1][1]
                    self.areaF[j].insert(k,[wals[j].p*(1-R)+wals[j].q*R,di])
                    if f and j == j0:
                        print("p self.areaF["+str(j)+"]" +"→".join(["(%.2f,%.2f)←↑%.2f"%(a[0][0],a[0][2],a[1]) for a in self.areaF[j][:-1]])+"(%.2f,%.2f)"%(self.areaF[j][-1][0][0],self.areaF[j][-1][0][2]))
                
                while r<1.0+EPS:
                    if f and j == j0:
                        print(P,r,"r = wals[j].rate(P)",sep="\t")
                    
                    if wals[j].rate(w.q)<r and k>0: #P is behind w.q, #stop here
                        r = wals[j].rate(w.q)
                        di = self.areaF[j][k-1][1]
                        self.areaF[j][k-1][1]=d
                        self.areaF[j].insert(k,[wals[j].p*(1-r)+wals[j].q*r,di])
                        if f and j == j0:
                            print("q self.areaF["+str(j)+"]" +"→".join(["(%.2f,%.2f)←↑%.2f"%(a[0][0],a[0][2],a[1]) for a in self.areaF[j][:-1]])+"(%.2f,%.2f)"%(self.areaF[j][-1][0][0],self.areaF[j][-1][0][2]))
                        break
                    else:  #w can cover this area
                        if k>2 and abs(self.areaF[j][k-2][1]-d)<EPS: #merge it with the area before me
                            del self.areaF[j][k-1]
                            k-=1
                        else: #cover this area
                            self.areaF[j][k-1][1]=d
                    if k == len(self.areaF[j])-1:
                        break
                    k += 1
                    P = self.areaF[j][k][0]
                    r = wals[j].rate(P)
                    di = self.areaF[j][k][1]
            

        self.Us=[]
        self.straight = []
        for i in range(4):
            for j in self.areaF[i][:-1]:#print(wals[i].rate(j[0]))
                if (wals[i].rate(j[0])<EPS and i == 0) or not(wals[i].rate(j[0])<EPS and max(j[1],delta)==self.straight[-1][1]):
                    self.straight.append([wals[i].rate(j[0])+i,max(j[1],delta),0])
        if self.straight[-1][1]==self.straight[0][1]:
            if len(self.straight)==1:
                self.straight[0][2]=wals[0].length+wals[1].length+wals[2].length+wals[3].length
                return
            del self.straight[0]
            
        s = 0
        a = 0
        #print(self.straight)#raise NotImplementedError
        self.straight.append([self.straight[0][0]+4,self.straight[0][1],0])
        if f:
            print("straight: "+", ".join([ "(%.2f,%.2f,%.2f)"%(a[0],a[1],a[2]) for a in self.straight]))
        
        for i in range(8):
            while s<len(self.straight) and self.straight[s][0] < i+1:
                self.straight[s-1][2] += (self.straight[s][0]-a)*wals[i%4].length
                a = self.straight[s][0]
                s += 1
            a = i+1
            self.straight[s-1][2] += (a-max(self.straight[s-1][0],i))*wals[i%4].length
        if f:
            print("straight: "+", ".join([ "(%.2f,%.2f,%.2f)"%(a[0],a[1],a[2]) for a in self.straight]))
        
        self.straight = self.straight[:-1]
        
        
        for s in range(len(self.straight)):
            if self.straight[s][1]>max(self.straight[s-1][1],self.straight[(s+1)%len(self.straight)][1]):
                l = self.straight[s][2]
                #try:
                self.Us.append(l - l*np.math.exp(-0.05*(2.0*self.straight[s][1])/(self.straight[s-1][1]+self.straight[(s+1)%len(self.straight)][1])))
                #except:
                #    pass

        self.Us = sorted(self.Us,key=lambda x:-x)
            #go from w.p to w.q

    def onWallLength(self, delta):
        return np.sum([s[2]*int(s[1]<delta+EPS) for s in self.straight])

    def onWallSegment(self, delta):
        return np.sum([int(s[1]<delta+EPS) for s in self.straight])

    def avgOnWallLength(self,delta):
        return self.onWallLength(delta)/max(1,self.onWallSegment(delta))

    def separation(self):
        return 0 if len(self.Us)<2 else np.sum(self.Us[1:])

    def printConnectivityInfo(self,delta):
        #self.areaF = [ [ (wals[0].p,20),(wals[0].q,-1)],[(wals[1].p,20),(wals[1].q,-1)],[(wals[2].p,20),(wals[2].q,-1)],[(wals[3].p,20),(wals[3].q,-1)] ]
        #self.straight = [relLength,dis,absLength]
        print(self)
        print("areaF:\n"+'\n'.join([ "→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in ar[:-1]])+"(%.3f,%.3f)"%(ar[-1][0][0],ar[-1][0][2])  for ar in self.areaF ]))
        print("\nstraight: "+", ".join([ "(%.3f,%.3f,%.3f)"%(a[0],a[1],a[2]) for a in self.straight]))
        print("Us:       "+str(self.Us))
        print("onWallLength: "+str(self.onWallLength(delta))+"\tonWallSegment:"+str(self.onWallSegment(delta))+"\tseparation:   "+str(self.separation()))
        return

    def inner(self,w):
        dis,vec = w.distance(self.p)
        if abs(vec[2])<EPS/10: 
            if vec[0]<EPS:
                if abs(dis)-EPS*2 < self.res[0][0] and dis > -EPS: #
                    return vec,True
            if vec[0]>-EPS:
                if abs(dis)-EPS*2 < self.res[2][0] and dis > -EPS:
                    return vec,True
        if abs(vec[0])<EPS/10:
            if vec[2]<EPS:
                if abs(dis)-EPS*2 < self.res[1][0] and dis > -EPS:
                    return vec,True
            if vec[2]>-EPS:
                if abs(dis)-EPS*2 < self.res[3][0] and dis > -EPS:
                    return vec,True
        if abs(vec[0])>EPS/10 and abs(vec[2])>EPS/10:
            print(str(vec)+"not vertical or horizontal wall, currently not supported by space detection or prob-update")
            raise NotImplementedError
        return [0,0],False

    def toWalls(self,eps=0): #,w=None,x=None,qro=None
        #if self.nearest()[0]< eps or self.nearest()[1]<eps:print(self,self.area(),"toWalls error",w,x,qro,"yes",sep="\n")
        return walls(c=[self.p[0],self.p[2]],a=[self.nearest()[0]-eps,self.nearest()[1]-eps])

class spce():
    def __init__(self,c0,c1,pro=None,scne=None,delta=None,idx=-1):
        self.c0 = np.min(np.array([c0,c1]),axis=0)
        self.c1 = np.max(np.array([c0,c1]),axis=0)
        self.c = (c0+c1)/2.0
        self.a = self.c1 - self.c
        self.delta=delta
        self.corners = [np.array(self.c1),np.array([self.c0[0],self.c0[1],self.c1[2]]),np.array(self.c0),np.array([self.c1[0],self.c1[1],self.c0[2]])]
        
        # self.bounds = [bond(self.c0,np.array([self.c1[0],self.c1[1],self.c0[2]]),0,idx),
        #                bond(np.array([self.c1[0],self.c1[1],self.c0[2]]),self.c1,1,idx)),
        #                bond(self.c1,np.array([self.c0[0],self.c0[1],self.c1[2]]),2,idx)),
        #                bond(np.array([self.c0[0],self.c0[1],self.c1[2]]),self.c0,3,idx))]

        self.wallsSign = [[],[],[],[]]
        
        #onBds = []
        #pro.printConnectivityInfo(delta)
        w = pro.toWalls()
        for i in range(4):
            #f = True
            for j in pro.areaF[i]:
                self.wallsSign[i].append([j[0],j[1],(j[1]<delta),w[i].rate(j[0]),w[i].rate(j[0])*w[i].length])
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
        Area=-100
        for j in range(4):
            area=0
            i = (j-1)%4
            idx = len(self.wallsSign[i])-1
            while idx > -1:
                if self.wallsSign[i][idx][1] < 0.42:
                    area+=self.wallsSign[i][idx][-1]
                else:
                    break
                idx -= 1

            idx = 0
            while idx < len(self.wallsSign[j]):
                if self.wallsSign[j][idx][1] < 0.42:
                    area+=self.wallsSign[j][idx][-1]
                else:
                    break
                idx += 1
            if area > Area:
                self.I,self.J = (j-1)%4,j
                Area = area
        #print(self.I)
        #print(self.J)
        #raise NotImplementedError
        self.refObj = obje(self.corners[self.J],np.array([1,1,1]),np.array([(2-self.J)*np.math.pi/2])) #if self.J != -1 else  obje((self.corners[self.I]+self.corners[(self.I-1)%4])/2.0,np.array([1,1,1]),self.I*np.math.pi)
        self.relA = np.array([self.a[0],self.a[1],self.a[2]]) if self.J%2==0 else np.array([self.a[2],self.a[1],self.a[0]])

    def transformInward(self, objes):
        return [self.refObj.rela(o) for o in objes]

    def transformOutward(self, objes):
        return [self.refObj.rely(o) for o in objes]
        
    def recycle(self,BD,WALLS):
        print(BD)
        self.corners[(self.J+1)%4] = self.corners[self.J] + self.refObj.matrix(1)@np.array([BD[0],0,0])#(self.refObj.matrix(1)*np.array([[BD[0],0,0]])).sum(axis=-1)
        self.corners[(self.J+2)%4] = self.corners[self.J] + self.refObj.matrix(1)@np.array([BD[0],0,BD[2]])#(self.refObj.matrix(1)*np.array([[BD[0],0,BD[2]]])).sum(axis=-1)
        self.corners[(self.J+3)%4] = self.corners[self.J] + self.refObj.matrix(1)@np.array([0,0,BD[2]])#(self.refObj.matrix(1)*np.array([[0,0,BD[2]]])).sum(axis=-1)

        self.c0 = np.array(self.corners).min(axis=0)
        self.c1 = np.array(self.corners).max(axis=0)
        self.c = (self.c0+self.c1)/2.0
        self.a = self.c1 - self.c

        PRO = prob(self.c)
        PRO.res = [(self.a[0]-EPS,True),(self.a[2]-EPS,True),(self.a[0]-EPS,True),(self.a[2]-EPS,True)]
        # print(WALLS)
        # print(PRO)
        # print(self.c0)
        # print(self.c1)
        # print(self.c)
        # print(self.a)
        PRO.areaFunctionDetection(WALLS, self.delta)#, True,1)
        self.wallsSign = [[],[],[],[]]
        
        w = PRO.toWalls()
        for i in range(4):
            for j in PRO.areaF[i]:
                self.wallsSign[i].append([j[0],j[1],(j[1]<self.delta),w[i].rate(j[0]),w[i].rate(j[0])*w[i].length])



    @classmethod
    def fromProb(cls,c,a):
        return cls(c-a,c+a)
    
    @classmethod
    def fromPro(cls,pro,scne,delta):
        c = pro.p
        a = two23(pro.nearest())
        return cls(c-a,c+a,pro,scne,delta)

    def __str__(self):
        return "[[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f],[%.2f,%.2f]] at [%.2f,%.2f] with [%.2f,%.2f]"%(self.corners[0][0],self.corners[0][2],self.corners[1][0],self.corners[1][2],self.corners[2][0],self.corners[2][2],self.corners[3][0],self.corners[3][2],self.c[0],self.c[2],self.a[0],self.a[2])

    def draw(self):
        scl = [1.0,0.8,0.6,0.4]
        c,a = self.c, self.a
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color="pink")

    def backToWall(self, o):
        #lots of problems here.
        #Isolating each object may cause global problems
        #What if the door occurs in the self.J wall?
        #so is that means, when we are selecting the self.J's wall, the door should be considered already
        #how should we organize such data structure. fuck. It's the problem
        
        #another idea is that if we can imitate the main "againsting wall behaviour" through the sampling process: generate(useWalls=True)
        #what if the "preserving area in front of the door"
        #if we do not let the back on the wall operation,
        #then "preserving area in the front of the door" should be implemented through extractingSpce()?
        #what if squeeze the area in front
        #but the problem turns out to be, we should allow some sort of alone areas.
        #the eliminating() is more complex then

        #we should also allow some sort of alone while selecting?
        #we should also talk about the length of high areas when talking about the Us
        pass

    def toWalls(self,eps=0):
        return walls(c=[self.c[0],self.c[2]],a=[self.a[0]-eps,self.a[2]-eps])

class spces():
    def __init__(self, scne=None, wals=None, name="", flex=1.2, sz=-4.0, drawFolder=""):
        self.scne = scne
        self.iWALLS= deepcopy(scne.WALLS if wals is None else wals) #a copy for visualization
        self.WALLS = deepcopy(scne.WALLS if wals is None else wals) #a local data structure for processing
        self.SPCES = []
        self.name = name
        try:
            self.LH = self.iWALLS.LH() if sz < 0 else [sz,sz]
        except:
            self.LH = None
        self.flex = flex
        self.delta = 0.04
        self.drawFolder = drawFolder
        if drawFolder and not os.path.exists(drawFolder):
            os.makedirs(drawFolder)
    
    def __iter__(self):
        return iter(self.SPCES)

    def draw(self,folder="",dr=True):
        #print("here "+folder)
        #print(self.SPCES[0])
        [s.draw() for s in self.SPCES]
        if (folder or self.drawFolder) and dr:#print("spces draaw", (folder if folder else self.drawFolder)+str(len(self.SPCES))+".png")
            self.iWALLS.draw(color="gray")
            self.WALLS.draw(color="black")
            f=(folder if folder else self.drawFolder)+str(len(self.SPCES))+".png"
            plt.axis('equal')
            L = max(self.LH)
            plt.xlim(-self.flex*L,self.flex*L)
            plt.ylim(-self.flex*L,self.flex*L)
            plt.savefig(f)
            ImageOps.invert(Image.merge('RGB', Image.open(f).split()[:3])).save(f)
            plt.clf()

    def tinyAdjustProb(self,Prob):
        if abs(Prob.res[0][0]-Prob.res[2][0])<self.delta*2:
            a = (Prob.res[0][0]+Prob.res[2][0])/2.0
            b = (Prob.res[0][0]-Prob.res[2][0])/2.0
            Prob.res[0],Prob.res[2] = (a,True),(a,True)
            Prob.p[0] += b
        if abs(Prob.res[1][0]-Prob.res[3][0])<self.delta*2:
            a = (Prob.res[1][0]+Prob.res[3][0])/2.0
            b = (Prob.res[1][0]-Prob.res[3][0])/2.0
            Prob.res[1],Prob.res[3] = (a,True),(a,True)
            Prob.p[2] += b
        return Prob

    def eliminatedSpace(self,Spce,f=False):
        if f:
            print("wallsSign:\n"+'\n'.join([ "→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in ar[:-1]])+"(%.3f,%.3f)"%(ar[-1][0][0],ar[-1][0][2])  for ar in Spce.wallsSign ]))
        
        #just check the off wall segments of the spce?
        height = Spce.wallsSign[3][-1][1]
        idx = [0,0]
        while True:
            if Spce.wallsSign[idx[0]][idx[1]][1]<EPS and height>EPS:
                break
            height = Spce.wallsSign[idx[0]][idx[1]][1]
            if len(Spce.wallsSign[idx[0]]) > idx[1]+2:
                idx[1] += 1
            else:
                idx[1]=0
                idx[0]=(idx[0]+1)%4

        IDX = [idx[0],idx[1]]
        #print(IDX)
        offWallSegments = []
        currentSegment = None#[None,0,None,[]] #startingpoint, length, endingpoint, walls
        tw = Spce.toWalls()
        while True:
            height = Spce.wallsSign[idx[0]][idx[1]][1]
            if len(Spce.wallsSign[idx[0]]) > idx[1]+2:
                idx[1] += 1
            else:
                idx[1]=0
                idx[0]=(idx[0]+1)%4
                if currentSegment is not None and not (Spce.wallsSign[idx[0]][idx[1]][1]<EPS and height > EPS):
                    P = Spce.wallsSign[idx[0]][idx[1]][0]
                    currentSegment[3][-1].q = np.copy(P)
                    currentSegment[3][-1].lengthh()
                    N = currentSegment[3][-1].n
                    currentSegment[3].append(wall(P,P,np.copy(tw[idx[0]].n),-1,-1,-1))
            
            if Spce.wallsSign[idx[0]][idx[1]][1]<EPS and height > EPS:
                #end an offWallSegment
                P = Spce.wallsSign[idx[0]][idx[1]][0]
                currentSegment[2] = deepcopy(P)
                currentSegment[3][-1].q = np.copy(P)
                currentSegment[3][-1].lengthh()
                currentSegment[1] = np.sum([w.length for w in currentSegment[3]])
                offWallSegments.append(deepcopy(currentSegment))
                currentSegment = None
            elif Spce.wallsSign[idx[0]][idx[1]][1]>EPS and height > EPS:
                P = Spce.wallsSign[idx[0]][idx[1]][0]
                currentSegment[3][-1].q = np.copy(P)
                currentSegment[3][-1].lengthh()
                #stretch the offWallSegment
                pass
            elif Spce.wallsSign[idx[0]][idx[1]][1]>EPS and height < EPS: 
                #start an offWallSegment
                P = Spce.wallsSign[idx[0]][idx[1]][0]
                currentSegment = [P,0,None,[wall(P,P,tw[idx[0]].n,-1,-1,-1)]] #startingpoint, length, endingpoint, walls
                pass

            if idx[0] == IDX[0] and idx[1] == IDX[1]:
                break 
        
        if f:
            for c in offWallSegments:
                print("[%.3f, %.3f] -> %.3f -> [%.3f, %.3f]"%(c[0][0],c[0][2],c[1],c[2][0],c[2][2]))
                for w in c[3]:
                    print(w)
                print("\n")

        

        A = sorted(offWallSegments,key=lambda x:-x[1])[0]

        if f:
            print(self.WALLS)
            print("\n")
        W = self.WALLS.searchWall(A[0],False)[0]
        a=W.idx
        #W.q = A[3][-1].q
        X = self.WALLS.searchWall(A[2],True)[0]
        if f:
            print("W")
            print(W)
        W.p = A[3][0].p
        if f:
            print("X")
            print(X)
        if X.idx == W.idx:
            a = self.WALLS.insertWall(w2 = W.idx)
            X = self.WALLS[a]
        X.q = A[3][-1].q
        W.w1 = X.idx
        X.w2 = W.idx
        if f:
            print(self.WALLS)
            print("\n")
        a=self.WALLS.insertWall(X.idx)
        A[3].reverse()
        if f:
            print(self.WALLS)
            print("\n")
        for w in A[3][:-1]:#
            self.WALLS[a].q = w.p
            self.WALLS[a].resetN()
            a=self.WALLS.insertWall(a)
            if f:
                print(self.WALLS)
                print("\n")
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
            print(self.WALLS)
        #raise NotImplementedError

        return True



    def eliminatingSpace(self, Spce):
        return self.eliminatedSpace(Spce)
        #addSpace's walls#print(Spce)#print(self.WALLS)
        PID,W=None,None
        for pid in range(4):
            for w in [w for w in self.WALLS if w.v]:
                #print(w.q-Spce.corners[pid])#print(np.linalg.norm(w.q-Spce.corners[pid]))
                if np.linalg.norm(w.q-Spce.corners[pid])<self.delta/2:
                    W=w
                    break
            if W:
                PID=pid
                break
        if W is None:
            print(self.WALLS,Spce,sep="\n")
            return False
        #else:
            #print(W.q,Spce.corners[pid],sep="\n")

        X=self.WALLS[W.w2]
        a=W.idx
        for i in range(3,-1,-1):
            X.p = Spce.corners[(PID+i)%4]
            a=self.WALLS.insertWall(a)
        #print(self.WALLS)

        self.WALLS.minusWall(W.idx,self.delta)
        if X.v:
            self.WALLS.minusWall(X.w1,self.delta)
        #print(self.WALLS)
        return True

    def extractingSpce(self,DIR="",hint=None):
        if len([w for w in self.WALLS if w.v])==0:
            return None
        #grid on the space
        N,M = int(self.flex*self.LH[0]/self.delta),int(self.flex*self.LH[1]/self.delta)
        GRIDS = self.delta*np.array([[[i,j] for i in range(-N,N+1)] for j in range(-M,M+1)]).reshape((-1,2))
        GRIDPro = []
        for loc in GRIDS:
            GRIDPro.append([])
            pro = prob(two23(loc))
            for w in self.WALLS:
                if w.v and w.over(pro.p):#print("not over")
                    dis,vec = w.distance(pro.p)#print(dis)#print(vec)
                    pro.update(dis,vec)
                
            i,f = pro.status()
            if not i or min(pro.nearest())<EPS*10: # or pro.area()<EPS/10
                continue
            if not f:
                print("warning :inconsist in or out status",pro,sep="\n")#,GRIDPro.index(pro))
                #assert f

            if not self.WALLS.crossCheck(pro.toWalls(EPS)):
                #pro = self.tinyAdjustProb(pro)
                pro.areaFunctionDetection(self.WALLS,self.delta)
                GRIDPro[-1].append(pro)
            else:
                #print(pro.p,self.WALLS,sep="\n")
                for xi in range(len(self.WALLS)):
                    x = self.WALLS[xi]
                    if x.v and pro.inner(x)[1]:
                        for w in self.WALLS[xi+1:]:
                            if w.v and abs(w.n@x.n)<EPS and pro.inner(w)[1]:
                                qro = prob(pro.p)
                                dis,vec=w.distance(qro.p)
                                qro.update(dis,vec)
                                dis,vec=x.distance(qro.p)
                                qro.update(dis,vec)
                                #qro = self.tinyAdjustProb(qro)print(qro.toWalls(),qro.toWalls(EPS),sep="\n") #,w,x,pro
                                if min(qro.nearest())>EPS*2 and not self.WALLS.crossCheck(qro.toWalls(EPS/2)):#qro.area()>EPS/10 and 
                                    # print(w,x,qro,"yes",sep="\n")
                                    qro.areaFunctionDetection(self.WALLS,self.delta)
                                    GRIDPro[-1].append(qro)
                assert len(GRIDPro[-1])
            
        if DIR or self.drawFolder:
            self.drawProb(GRIDPro,(DIR if DIR else self.drawFolder))

        #Find a pro in pros

        #return sorted([sorted(gg,key=lambda x:-x.key(self.delta,hint))[0] for gg in GRIDPro if len(gg)],key=lambda x:-x.key(self.delta,hint))
        PRO = sorted([sorted(gg,key=lambda x:-x.key(self.delta,hint))[0] for gg in GRIDPro if len(gg)],key=lambda x:-x.key(self.delta,hint))[0]
        
        #print(self.WALLS,PRO,PRO.toWalls(),sep="\n")
        #PRO.printConnectivityInfo()
        #PRO.areaFunctionDetection(self.WALLS,self.delta,True)
        #PRO.printConnectivityInfo(self.delta)
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n") #raise NotImplementedError
        #PRO = self.tinyAdjustProb(PRO)
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n")
        #raise NotImplementedError
        #print(PRO)

        return spce.fromPro(PRO,self.scne,self.delta)#spce.fromProb(PRO.p,two23(PRO.nearest()))

    def extractingMoreSpces(self,DIR="",hint=None):#[spce(np.array([-1.8595,0.,-1.332]),np.array([3.3795,0.,1.252]),scne=self)]#
        spss = self.extractingSpce()#[None]#
        for p in spss[:20]:
            #p = prob(np.array([2.84,0,-0.04]))
            #p.res = [(0.1385,True),(1.3065,True),(6.2185,True),(1.2910,True)] 
            #print(p)
            #print(spss.index(p))
            #p.printConnectivityInfo(self.delta)
            #print("\n")
            p.areaFunctionDetection(self.WALLS,self.delta,False)
            #print("\n")
            #p.printConnectivityInfo(self.delta)
            print(str(spss.index(p)/10)+": [p.key() = %.3f ]= [(-1.0 / ratio) = %.3f] + [np.average(self.nearest())*2 = %.3f] -[self.separation()*5 = %.3f] + [self.onWallLength(delta) = %.3f]"%(p.key(self.delta,hint), -1.0/p.ratio(),np.average(p.nearest())*2,p.separation()*5,p.onWallLength(self.delta)))
            #print(self.WALLS)
            sp = spce.fromPro(p,self.scne,self.delta)
            self.SPCES.append(sp)
            sp.scne = self.scne
            self.draw(self.drawFolder+" testings "+str(spss.index(p))+" ")
            self.SPCES = []

    def extractingSpces(self,bound=1,DIR=""):
        self.SPCES,sp,b = [],self.extractingSpce(DIR),self.draw((DIR if DIR else self.drawFolder)) if DIR or self.drawFolder else None
        while sp and len(self.SPCES)<bound:
            self.SPCES.append(sp)
            sp.scne = self.scne
            a = self.eliminatingSpace(sp)
            b = self.draw((DIR if DIR else self.drawFolder)) if DIR or self.drawFolder else None
            if not a:
                break
            sp = self.extractingSpce((DIR if DIR else self.drawFolder))

    def drawProb(self, probArray, DIR="", order=0, aFlag=True):
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

import sys,argparse,os
def parse(argv):
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-i','--identity', default="3")
    parser.add_argument('-n','--new', default=False, action="store_true")
    parser.add_argument('-b','--bound', default=19)
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    DIR,args = "./newRoom/",parse(sys.argv[1:])
    for i in (range(9,args.bound) if int(args.identity) == -1 else range(int(args.identity),int(args.identity)+1)):
        if args.new:
            wls = walls(name="rand"+str(i))#print(wls.LOGS)
            wls.randomWalls()
            wls.output()
        
        #print(i)
        wlz = walls.fromLog(f=DIR+"rand"+str(i)+".txt",name="rand"+str(i)+"_",drawFolder=DIR) #wlz.draw(DIR)
        #print(wlz)
        sm = spces(wals=wlz,name=wlz.name,drawFolder=DIR+wlz.name+"/")
        sm.extractingSpces(2)
        #sm.extractingMoreSpces()