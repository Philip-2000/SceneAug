# from . import SPCES,WALLS
from Wall import *
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from PIL import Image,ImageOps,ImageDraw

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
        self.p=np.copy(p)
        self.res=[(20,False),(20,False),(20,False),(20,False)]
        self.areaF = []
        self.Us=[]
        self.straight = []
    
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
        
    def key(self,hint=None):
        return self.ratio()+np.average(self.nearest())*0.4 

    #moreAttributes
    #stick on wall length,
    #stick on wall number
    #area function
    #
    def areaFunctionDetection(self, walls):
        assert not walls.crossCheck(self.toWalls(EPS/2))
        wals = self.toWalls()
        #print(wals)
        self.areaF = [ [ (wals[0].p,20),(wals[0].q,-1)],[(wals[1].p,20),(wals[1].q,-1)],[(wals[2].p,20),(wals[2].q,-1)],[(wals[3].p,20),(wals[3].q,-1)] ]
        for i in range(len(walls)):
            w = walls[i]
            j = [j for j in range(4) if wals[j].n@walls[i].n > 1-EPS]
            j = j[0]
            d = (self.areaF[j][0][0]-w.p)@w.n#print(d)
            P = self.areaF[j][0][0]
            r = wals[j].rate(P)#print(r)
            k = 0
            while w.rate(P)<1:
                k+=1
                P = self.areaF[j][k][0]
                r = wals[j].rate(P)
                if r > 1+EPS:
                    break#print(r)
                if self.areaF[j][k-1][1]>d:
                    if k>2 and abs(self.areaF[k-2][1]-d)<EPS:
                        del self.areaF[j][k-1]
                        k-=1
                    else:
                        self.areaF[j][k-1]=(self.areaF[j][k-1][0],d)
                if k == len(self.areaF[j])-1:
                    break
            if k+1 < len(self.areaF[j])-1:
                r = wals[j].rate(w.q)
                self.areaF[j].insert(k+1,(wals[j].p*r+wals[j].q*(1-r),d))

        self.Us=[]
        self.straight = []
        for i in range(4):
            # print(self.areaF[i][0][0])
            # print(self.areaF[i][0][1])
            # print(self.areaF[i][1][0])
            # print(self.areaF[i][1][1])
            for j in self.areaF[i][:-1]:#print(wals[i].rate(j[0]))
                if (wals[i].rate(j[0])<EPS and i == 0) or not(wals[i].rate(j[0])<EPS and j[1]==self.straight[-1][1]):
                    self.straight.append([wals[i].rate(j[0])+i,j[1],0])
        if self.straight[-1][1]==self.straight[0][1]:
            if len(self.straight)==1:
                self.straight[0][2]=wals[0].length+wals[1].length+wals[2].length+wals[3].length
                return
            del self.straight[0]
            
        s = 0
        a = 0
        #print(self.straight)#raise NotImplementedError
        for i in range(4):
            while s<len(self.straight) and self.straight[s][0] < i+1:
                self.straight[s-1][2] += (self.straight[s][0]-a)*wals[i].length
                a = self.straight[s][0]
                s += 1
            a = i+1
            self.straight[s-1][2] += (a-self.straight[s-1][0])*wals[i].length

        
        for s in range(len(self.straight)):
            if self.straight[s][1]<min(self.straight[s-1][1],self.straight[(s+1)%len(self.straight)][1]):
                try:
                    self.Us.append(1 - np.math.exp(-(self.straight[s-1][1]+self.straight[(s+1)%len(self.straight)][1])/(2.0*max(self.straight[s][1],EPS))))
                except:
                    pass

        self.Us = sorted(self.Us,key=lambda x:-x)
            #go from w.p to w.q

    def onWallLength(self):
        return np.sum([s[2]*int(s[1]<EPS/10) for s in self.straight])

    def onWallSegment(self):
        return np.sum([int(s[1]<EPS/10) for s in self.straight])

    def separation(self):
        return 0 if len(self.Us)<2 else np.sum(self.Us[1:])

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
    def __init__(self, scne=None, wals=None, name="", flex=1.2, sz=-4.0, drawFolder=""):
        self.scne = scne
        self.iWALLS= deepcopy(scne.WALLS if wals is None else wals) #a copy for visualization
        self.WALLS = deepcopy(scne.WALLS if wals is None else wals) #a local data structure for processing
        self.SPCES = []
        self.name = name
        self.LH = self.iWALLS.LH() if sz < 0 else [sz,sz]
        self.flex = flex
        self.delta = 0.04
        self.drawFolder = drawFolder
        if drawFolder and not os.path.exists(drawFolder):
            os.makedirs(drawFolder)
    
    def draw(self,folder=""):
        [s.draw() for s in self.SPCES]
        if folder or self.drawFolder:#print("spces draaw", (folder if folder else self.drawFolder)+str(len(self.SPCES))+".png")
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

    def eliminatingSpace(self, Spce):
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

    def extractingSpce(self,DIR=""):
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
            if not i or min(pro.nearest())<EPS*1.5: # or pro.area()<EPS/10
                continue
            if not f:
                print(self.WALLS,pro,"inconsist in or out status",sep="\n")#,GRIDPro.index(pro))
                assert f

            if not self.WALLS.crossCheck(pro.toWalls(EPS)):
                pro = self.tinyAdjustProb(pro)
                pro.areaFunctionDetection(self.WALLS)
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
                                if min(qro.nearest())>EPS*2 and not self.WALLS.crossCheck(qro.toWalls(EPS)):#qro.area()>EPS/10 and 
                                    # print(w,x,qro,"yes",sep="\n")
                                    qro.areaFunctionDetection(self.WALLS)
                                    GRIDPro[-1].append(qro)
                assert len(GRIDPro[-1])
            
        if DIR or self.drawFolder:
            self.drawProb(GRIDPro,(DIR if DIR else self.drawFolder))

        #Find a pro in pros
        PRO = sorted([sorted(gg,key=lambda x:-x.key())[0] for gg in GRIDPro if len(gg)],key=lambda x:-x.key())[0]
        
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n") #raise NotImplementedError
        PRO = self.tinyAdjustProb(PRO)
        # print(PRO.toWalls(),PRO.toWalls(EPS),sep="\n")
        return spce.fromProb(PRO.p,two23(PRO.nearest()))

    def extractingSpces(self,bound=1,DIR=""):
        self.SPCES,sp,b = [],self.extractingSpce(DIR),self.draw((DIR if DIR else self.drawFolder)) if DIR or self.drawFolder else None
        while sp and len(self.SPCES)<bound:
            self.SPCES.append(sp)
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
    parser.add_argument('-i','--identity', default="72")
    parser.add_argument('-n','--new', default=False, action="store_true")
    parser.add_argument('-b','--bound', default=3)
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    DIR,args = "./newRoom/",parse(sys.argv[1:])
    for i in (range(args.bound) if int(args.identity) == -1 else range(int(args.identity),int(args.identity)+1)):
        if args.new:
            wls = walls(name="rand"+str(i))#print(wls.LOGS)
            wls.randomWalls()
            wls.output()
        wlz = walls.fromLog(f=DIR+"rand"+str(i)+".txt",name="rand"+str(i)+"_",drawFolder=DIR) #wlz.draw(DIR)
        wlz.draw(True)
        sm = spces(wals=wlz,name=wlz.name,drawFolder=DIR+wlz.name+"/")
        sm.extractingSpces(2)

"""
总结一下怎恶魔做。
每个点，多种方案。对。每个方案都记，肯定是都记录下来的
每个方案在生成的时候，其实还需要把更复杂的信息探测到。比如贴边总长度，贴边联通数。
可是还容易出现一个问题，那就是空间切碎的问题。这个事情该怎么讨论呢？
就是说有在


然后去挑一个方案？


The problem is to visualize all these things first? Temporarily
But how? i DONT KNOW



以后：挑的时候其实也可以给一个这个hint，说我需要的是一个多大的一个空间？

New problems cames out. Always.
"""