import numpy as np
from matplotlib import pyplot as plt 
from Logg import *
from Obje import *
import json
WALLSTATUS = True
EPS = 0.001
class wall():
    def __init__(self, p, q, n, w1, w2, idx, v=True, spaceIn=False, sig=-1, scene=None, array=None):
        global WALLSTATUS
        if v and abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) > 0.01: #assert abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) < 0.01
            WALLSTATUS = False
            print("not straight " + str(sig))
        if (p[0]-q[0])*n[2]>(p[2]-q[2])*n[0]: #assert (p[0]-q[0])*n[2]<=(p[2]-q[2])*n[0]
            WALLSTATUS = False
            #print(traceback.format_stack())
            print("not right-hand " + str(sig))
            raise NotImplementedError
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
        self.scne=scene
        self.array=array#scne.WALLS if scne is not None else array#return WALLSTATUS
        
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

    def rate(self,p,checkOn=False):
        r = ((p-self.p)@(self.q-self.p)) / ((self.q-self.p)@(self.q-self.p))
        assert (not checkOn) or (abs((p-self.p)@self.n)<EPS and 0<=r and r<=1)
        return r


def two23(a):
    return np.array([a[0],0,a[1]])

class walls():
    def __init__(self, Walls=[], c_e=0, scne=None, c=[0,0], a=[2.0,2.0], printLog=False, name="", flex=1.2, drawFolder="", keepEmptyWL = False):
        #print(keepEmptyWL)
        if len(Walls)>0:
            self.WALLS = [wall(two23(walls[j][:2])-c_e,two23(walls[(j+1)%len(walls)][:2])-c_e,np.array([walls[j][3],0,walls[j][2]]),(j-1)%len(walls),(j+1)%len(walls),j,scene=scne,array=self) for j in range(len(Walls))]
        else:#if a[0]<0 or a[1]<0:print(a)
            self.WALLS = [wall(two23([c[0]+a[0]-2*a[0]*int((i+1)%4>1),c[1]+a[1]-2*a[1]*int(i%4>1)]),two23([c[0]+a[0]-2*a[0]*int(i%4<2),c[1]+a[1]-2*a[1]*int((i+1)%4>1)]),two23([(2.0-i)*(i%2),(-1.0+i)*(1-i%2)]),(i-1)%4,(i+1)%4,i,array=self) for i in range(4)] if not keepEmptyWL else []
        self.LOGS = []
        self.printLog = printLog
        self.scne = scne
        self.name = name
        self.flex = flex
        self.drawFolder = drawFolder

        self.windoors = {}

    def __getitem__(self, idx):
        return self.WALLS[idx]

    def __len__(self):
        return len(self.WALLS)

    def __str__(self):
        return '\n'.join([str(w) for w in self.WALLS])

    def __iter__(self):
        return iter(self.WALLS)

    @classmethod
    def fromLog(cls,f,name="",drawFolder=""):
        a = cls(name=name,drawFolder=drawFolder)
        a.LOGS = [distribute(a,l) for l in open(f,"r").readlines()]
        a.centerize()
        [print(l) for l in a.LOGS if a.printLog]
        
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
    
    def toWallsJson(self):
        sha,nor,ori = [],[],[]
        if len([w.idx for w in self.WALLS if w.v]):
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            sha,nor,ori,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]],[[self.WALLS[J].n[0],self.WALLS[J].n[2]]],[np.math.atan2(self.WALLS[J].n[0],self.WALLS[J].n[2])], self.WALLS[J].w2
            while w != J:
                sha.append([float(self.WALLS[w].p[0]),float(self.WALLS[w].p[2])])
                nor.append([float(self.WALLS[w].n[0]),float(self.WALLS[w].n[2])])
                ori.append(float(np.math.atan2(self.WALLS[J].n[0],self.WALLS[J].n[2])))
                w = self.WALLS[w].w2
        return sha, nor, ori
    
    @classmethod
    def fromWallsJson(cls,sha,nor):
        o = walls(np.array([[sha[i][0],sha[i][1],nor[i][0],nor[i][1]] for i in range(len(sha))]))
        return o

    def processWithWindoor(self):
        for a in self.windoors:
            o = self.windoors[a]
            if o.class_name() == "door":
                a = int(a)

                W=self.WALLS[a]

                W.q = o.translation+(o.matrix()*np.array([[ o.size[0]+0.05,0,o.size[2]]])).sum(axis=-1)
                W.q[1] = 0
                self.insertWall(a)
                W.q = o.translation+(o.matrix()*np.array([[ o.size[0]+0.05,0,o.size[2]+0.4]])).sum(axis=-1)
                W.q[1] = 0
                self.insertWall(a)
                W.q = o.translation+(o.matrix()*np.array([[-o.size[0]-0.05,0,o.size[2]+0.4]])).sum(axis=-1)
                W.q[1] = 0
                self.insertWall(a)
                W.q = o.translation+(o.matrix()*np.array([[-o.size[0]-0.05,0,o.size[2]]])).sum(axis=-1)
                W.q[1] = 0
                self.insertWall(a)
            


        # X=self.WALLS[W.w2]
        # a=W.idx
        # for i in range(3,-1,-1):
        #     X.p = Spce.corners[(PID+i)%4]
        #     a=self.WALLS.insertWall(a)

        pass

    def searchWall(self,p,back=False):
        d,W=1000,None
        for w in self.WALLS:
            if w.v and w.over(p):
                print(w.idx)
                print(w.distance(p)[0])
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

    def stickWall(self,w,x):
        wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
        return w.v and x.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

    def crossWall(self,w,x):
        wxp,wxq,xwp,xwq,wwp,xxp = w.n@x.p, w.n@x.q, x.n@w.p, x.n@w.q, w.n@w.p, x.n@x.q
        return w.v and x.v and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))

    def crossCheck(self,other=None):
        otherWALLS = self.WALLS if other is None else other.WALLS
        for i in range(len(self.WALLS)):
            for j in range(i if other is None else len(otherWALLS)):
                if self.crossWall(self.WALLS[i], otherWALLS[j]):
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
        self.deleteWall(self.WALLS[I].w2,EPS)
        self.deleteWall(I,EPS)

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

    def deleteWall(self,id,delta):
        w = self.WALLS[self.WALLS[id].w1]
        #assert self.WALLS[id].length < delta*2 
        assert self.WALLS[id].length < delta*2 or abs(abs(w.n@self.WALLS[id].n)-1.0)<EPS or abs((w.p-self.WALLS[id].q)@self.WALLS[id].n)<EPS or abs((w.p-self.WALLS[id].q)@w.n)<EPS
        self.WALLS[id].v = False
        w.q = np.copy(self.WALLS[id].q)
        w.lengthh()
        w.resetN()
        w.w2 = self.WALLS[id].w2
        self.WALLS[self.WALLS[id].w2].w1 = w.idx

    def minusWall(self,id,delta):
        if not self.WALLS[id].v:
            return
        if abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w2].n + 1.)< EPS:
            I,J = id,self.WALLS[id].w2
        elif abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w1].n + 1.)< EPS:
            I,J = self.WALLS[id].w1,id
        else:
            if abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w2].n - 1.)< EPS:
                self.deleteWall(self.WALLS[id].w2,delta)
            elif abs(self.WALLS[id].n @ self.WALLS[self.WALLS[id].w1].n - 1.)< EPS:
                self.deleteWall(id,delta)
            return #self.WALLS
    

        if self.WALLS[I].length < self.WALLS[J].length:
            P,K = self.WALLS[I].p,self.WALLS[I].w1
        else:
            P,K = self.WALLS[J].q,self.WALLS[J].w2
        #print(self)
        self.WALLS[I].adjustWall(self.WALLS[I].q,P)
        self.WALLS[J].adjustWall(self.WALLS[J].p,P)

        # print("I=%d,J=%d,K=%d"%(I,J,K))
        # print(self)
        
        if self.WALLS[I].length < delta*2:
            self.deleteWall(I,delta)
        if self.WALLS[J].length < delta*2:
            self.deleteWall(J,delta)
        
        self.minusWall(K,delta)
        self.regularize(I)
        self.regularize(J)
        self.regularize(self.WALLS[I].w1)
        self.regularize(self.WALLS[J].w2)

    def regularize(self,id):
        w = self.WALLS[id]
        if not w.v:
            return
        oldp,oldq  = np.copy(w.p),np.copy(w.q)
        if min(abs(w.n[0]),abs(w.n[2]))>=EPS*50:
            print("warning: %.3f >= %.3f * %.1f, wall should not be regularized"%(min(abs(w.n[0]),abs(w.n[2])),EPS,50))
            #raise NotImplementedError

        if abs(w.n[0])<abs(w.n[2]):
            b = (w.p[2]+w.q[2])/2.0
            w.p[2],w.q[2] = b,b
        else:
            b = (w.p[0]+w.q[0])/2.0
            w.p[0],w.q[0] = b,b
        w.resetN()
        self.WALLS[w.w2].adjustWall(oldq,w.q,id)
        self.WALLS[w.w1].adjustWall(oldp,w.p,id)
            
    def LH(self):
        return [max([abs(w.q[0]) for w in self.WALLS if w.v]),max([abs(w.q[2]) for w in self.WALLS if w.v])]

    def bbox(self):
        return np.array([[min([w.p[0] for w in self.WALLS if w.v]),0,min([w.p[2] for w in self.WALLS if w.v])],
                       [max([w.p[0] for w in self.WALLS if w.v]),3,max([w.p[2] for w in self.WALLS if w.v])]])

    def field():
        #what about those serial version of Scene Fields?
        #we can debug with ourself.

        pass

    def draw(self,end=False,suffix=".png",color="black"):
        print(self)
        if len([w.idx for w in self.WALLS if w.v]):
            J = min([w.idx for w in self.WALLS if w.v])#WALLS[0].w2
            contour,w =[[self.WALLS[J].p[0],self.WALLS[J].p[2]]], self.WALLS[J].w2
            while w != J:
                contour.append([self.WALLS[w].p[0],self.WALLS[w].p[2]])
                w = self.WALLS[w].w2
            contour = np.array(contour)
            plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o", color=color)
            [self.windoors[o].draw() for o in self.windoors]
            if end and self.drawFolder:
                plt.axis('equal')
                L = max(self.LH())
                plt.xlim(-self.flex*L,self.flex*L)
                plt.ylim(-self.flex*L,self.flex*L)
                plt.savefig(self.drawFolder+self.name+suffix)
                plt.clf()

    def writeLog(self,):
        with open(self.drawFolder+self.name+".txt","w") as f:
            [f.write(str(l)) for l in self.LOGS]
        
    def output(self):
        self.draw(True)
        self.writeLog()



if __name__ == "__main__": #load="testings",
    
    DIR = "./newRoom/"
    W = walls.fromLog(f=DIR+"rand3.txt",name="",drawFolder=DIR) #wlz.draw(DIR)
    print(W)
    W.draw(True)