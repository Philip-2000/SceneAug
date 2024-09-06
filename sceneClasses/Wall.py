import numpy as np
from matplotlib import pyplot as plt 
from Logg import *
WALLSTATUS = True
EPS = 0.001
class wall():
    def __init__(self, p, q, n, w1, w2, idx, v=True, spaceIn=False, sig=-1, scne=None, array=None):
        global WALLSTATUS
        if v and abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) > 0.01: #assert abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) < 0.01
            WALLSTATUS = False
            print("not straight " + str(sig))
        if (p[0]-q[0])*n[2]>(p[2]-q[2])*n[0]: #assert (p[0]-q[0])*n[2]<=(p[2]-q[2])*n[0]
            WALLSTATUS = False
            print("not right-hand " + str(sig))
        self.linkIndex=[]
        self.idx=idx
        self.p=np.copy(p)
        self.q=np.copy(q)
        self.n=np.copy(n)
        self.length=(((self.p-self.q)**2).sum())**0.5
        self.w1=w1
        self.w2=w2
        self.v=v
        self.spaceIn=spaceIn
        self.scne=scne
        self.array=scne.WALLS if scne is not None else array#return WALLSTATUS
        
    def __str__(self):
        return (" " if self.v else "              ")+str(self.w1)+"<-"+str(self.idx)+"->"+str(self.w2)+"\t"+str(self.p)+"\t"+str(self.q)+"\t"+str(self.n)

    def lengthh(self):
        self.length=(((self.p-self.q)**2).sum())**0.5

    def mWall(self,L): 
        oldp = np.copy(self.p)
        self.p += self.n*L
        oldq = np.copy(self.q)
        self.q += self.n*L
        self.array[self.w1].adjustWall(oldp,np.copy(self.p),self.idx)
        self.array[self.w2].adjustWall(oldq,np.copy(self.q),self.idx)
        for i in self.linkIndex:
            self.scne.LINKS[i].adjust(self.n*L)
        
    def adjustWall(self,oldp,p,hint=-1):
        oldn = self.n
        if hint<0:
            if ((self.p-oldp)**2).sum()<0.001:
                oldq = self.q
                self.p=p
            elif ((self.q-oldp)**2).sum()<0.001:
                oldq = oldp
                oldp = self.p
                self.q=p
            else:
                print("adjustWall error")
                return
        else:
            if hint == self.w1:
                oldq = self.q
                self.p=p
            elif hint == self.w2:
                oldq = oldp
                oldp = self.p
                self.q=p
            else:
                print("false hint")
                return
        if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
            self.n = -self.n
        self.lengthh()#=(((self.p-self.q)**2).sum())**0.5

        for i in self.linkIndex:
            self.scne.LINKS[i].modify(oldp, oldq, oldn)
    
    def distance(self,P):
        return (P-self.p)@(self.n), (P-self.p)@(self.n)*self.n

    def over(self,P):
        return (P-self.q)@(self.p-self.q) > -EPS and (P-self.p)@(self.q-self.p) > -EPS

    def resetN(self):
        self.n = np.array([self.p[2]-self.q[2],0,self.q[0]-self.p[0]])/np.linalg.norm(np.array([self.p[2]-self.q[2],0,self.q[0]-self.p[0]]))

def two23(a):
    return np.array([a[0],0,a[1]])

class walls():
    def __init__(self, Walls=[], c_e=0, scne=None, l=2.0, printLog=False, name=""):
        if len(Walls)>0:
            self.WALLS = [wall(two23(walls[j][:2])-c_e,two23(walls[(j+1)%len(walls)][:2])-c_e,np.array([walls[j][3],0,walls[j][2]]),(j-1)%len(walls),(j+1)%len(walls),j,scne=scne) for j in range(len(walls))]
        else:
            self.WALLS = [wall(two23([l-2*l*int((i+1)%4>1),l-2*l*int(i%4>1)]),two23([l-2*l*int(i%4<2),l-2*l*int((i+1)%4>1)]),two23([(2.0-i)*(i%2),(-1.0+i)*(1-i%2)]),(i-1)%4,(i+1)%4,i,array=self) for i in range(4)]
        self.LOGS = []
        self.printLog = printLog
        self.scne = scne
        self.name = name

    def __getitem__(self, idx):
        return self.WALLS[idx]

    def __len__(self):
        return len(self.WALLS)

    def __str__(self):
        return '\n'.join([str(w) for w in self.WALLS])

    @classmethod
    def fromLog(cls,f,name=""):
        a = cls(name=name)
        a.LOGS = [distribute(a,l) for l in open(f,"r").readlines()]
        a.centerize()
        [print(l) for l in a.LOGS if a.printLog]
        return a

    def stickWall(self,w,x):
        wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
        return w.v and x.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

    def crossWall(self,w,x):
        wxp,wxq,xwp,xwq,wwp,xxp = w.n@x.p, w.n@x.q, x.n@w.p, x.n@w.q, w.n@w.p, x.n@x.q
        return w.v and x.v and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))

    def crossCheck(self):
        for i in range(len(self.WALLS)):
            for j in range(i):
                if self.crossWall(self.WALLS[i], self.WALLS[j]):
                    return True
        return False

    def maxHeight(self,x,bd=4.0):
        return min([bd]+[(w.p-x.p)@x.n for w in self.WALLS if (w.v and ((w.p-x.p)@x.n > 0.01))]) #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值

    def maxDepth(self,x,bd=-4.0):
        return max([bd]+[(w.p-x.p)@x.n for w in self.WALLS if (w.v and ((w.p-x.p)@x.n <-0.01))]) #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值

    def breakWall(self,id,rate):
    
        delList = []
        for l in self.WALLS[id].linkIndex:
            r = self.scne.LINKS[l].rate
            if r < rate:
                self.scne.LINKS[l].rate = r / rate
            else:
                self.scne.LINKS[l].src = len(self.WALLS)+1
                self.scne.LINKS[l].rate = (r-rate) / (1-rate)
                delList.append(l)
        for l in delList:
            self.WALLS[id].linkIndex.remove(l)

        self.WALLS[id].q = np.copy(rate*self.WALLS[id].q + (1-rate)*self.WALLS[id].p)
        self.insertWall(id)
        self.insertWall(id)

        for l in delList:
            self.WALLS[len(self.WALLS)-1].linkIndex.append(l)

    def squeezeWall(self,I):
        L1 = self.WALLS[I].length/2.0 if (self.WALLS[I].q-self.WALLS[I].p) @ self.WALLS[self.WALLS[I].w1].n > 0 else -self.WALLS[I].length/2.0
        L2 = self.WALLS[I].length/2.0 if (self.WALLS[I].p-self.WALLS[I].q) @ self.WALLS[self.WALLS[I].w2].n > 0 else -self.WALLS[I].length/2.0
        self.WALLS[self.WALLS[I].w1].mWall(L1)
        self.WALLS[self.WALLS[I].w2].mWall(L2)
        self.deleteWall(self.WALLS[I].w2)
        self.deleteWall(I)

    def squeezeWalls(self):
        i=0
        while i<len(self.WALLS):
            if self.WALLS[i].v and self.WALLS[i].length < 0.7:
                self.LOGS.append(dllg(self,i))#{"id":I,"delete":0})#self.LOGS[-1].operate()
                i=-1
            i+=1

    def randomWalls(self):
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
        
            if self.crossCheck():
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
        for w in self.WALLS:
            w.p -= cen
            w.q -= cen

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
        # print(p)
        # print(q)
        # print(p-q)
        # print((p-q)@(p-q))
        n = np.cross(self.WALLS[w1].n,np.array([0,1,0])) if (p-q)@(p-q)<0.01 else np.array([p[2]-q[2],0,q[0]-p[0]])/np.linalg.norm(np.array([p[2]-q[2],0,q[0]-p[0]]))
        # if (p-q)@(p-q)>0.01:
        #     n = np.array([p[2]-q[2],0,q[0]-p[0]])/np.linalg.norm(np.array([p[2]-q[2],0,q[0]-p[0]]))
        #     #print(n)
        # else:
        #     n = np.cross(self.WALLS[w1].n,np.array([0,1,0]))
        # print(n)
        self.WALLS.append(wall(p,q,n,w1,w2,ID,scne=self.scne,array=self))
        return ID

    def deleteWall(self,id):
        w = self.WALLS[self.WALLS[id].w1]
        assert self.WALLS[id].length < 6*EPS or abs((w.p-self.WALLS[id].q)@self.WALLS[id].n)<0.001 or abs((w.p-self.WALLS[id].q)@w.n)<0.001
        self.WALLS[id].v = False
        w.q = np.copy(self.WALLS[id].q)
        w.lengthh()
        w.resetN()
        w.w2 = self.WALLS[id].w2
        self.WALLS[self.WALLS[id].w2].w1 = w.idx

    def minusWall(self,id):
        if not self.WALLS[id].v:
            return
        if abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w2].n + 1.)< EPS*6:
            I = id
            J = self.WALLS[id].w2
        elif abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w1].n + 1.)< EPS*6:
            I = self.WALLS[id].w1
            J = id
        else:
            return #self.WALLS  print(I,J)
        
        if self.WALLS[I].length < self.WALLS[J].length:
            P = self.WALLS[I].p
            K = self.WALLS[I].w1
        else:
            P = self.WALLS[J].q
            K = self.WALLS[J].w2
        self.WALLS[I].adjustWall(self.WALLS[I].q,P)
        self.WALLS[J].adjustWall(self.WALLS[J].p,P)
        
        if self.WALLS[I].length < EPS*6:
            self.deleteWall(I)
        if self.WALLS[J].length < EPS*6:
            self.deleteWall(J)
        
        self.minusWall(K)

    def field():
        #what about those serial version of Scene Fields?
        #we can debug with ourself.

        pass

    def draw(self,folder="",suffix=".png",color="black"):
        if len([w.idx for w in self.WALLS if w.v]):
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            contour,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]], self.WALLS[J].w2
            while w != J:
                contour.append([self.WALLS[w].p[0],self.WALLS[w].p[2]])
                w = self.WALLS[w].w2
            contour = np.array(contour)
            plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o", color=color)
            if folder:
                plt.savefig(folder+self.name+suffix)
                plt.clf()

    def writeLog(self,folder):
        with open(folder+self.name+".txt","w") as f:
            [f.write(str(l)) for l in self.LOGS]
        
    def output(self,folder):
        self.draw(folder)
        self.writeLog(folder)