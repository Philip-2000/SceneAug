from . import LINKS
from . import WALLS
import numpy as np
#WALLS=[]
WALLSTATUS = True
class wall():
    def __init__(self, p, q, n, w1, w2, idx, v=True, spaceIn=False, sig=-1):
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
        #return WALLSTATUS
        

    def __str__(self):
        return (" " if self.v else "              ")+str(self.w1)+"<-"+str(self.idx)+"->"+str(self.w2)+"\t"+str(self.p)+"\t"+str(self.q)+"\t"+str(self.n)

    def lengthh(self):
        self.length=(((self.p-self.q)**2).sum())**0.5

    def mWall(self,L): 
        oldp = np.copy(self.p)
        self.p += self.n*L
        oldq = np.copy(self.q)
        self.q += self.n*L
        self.length=(((self.p-self.q)**2).sum())**0.5
        WALLS[self.w1].adjustWall(oldp,self.p,self.idx)
        WALLS[self.w2].adjustWall(oldq,self.q,self.idx)
        
        for i in self.linkIndex:
            LINKS[i].adjust(self.n*L)
        
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
        self.length=(((self.p-self.q)**2).sum())**0.5

        for i in self.linkIndex:
            LINKS[i].modify(oldp, oldq, oldn)
        pass

def breakWall(id,rate):
    #load all the links of 
    
    delList = []
    for l in WALLS[id].linkIndex:
        r = LINKS[l].rate
        if r < rate:
            LINKS[l].rate = r / rate
        else:
            LINKS[l].src = len(WALLS)+1
            LINKS[l].rate = (r-rate) / (1-rate)
            delList.append(l)
    for l in delList:
        WALLS[id].linkIndex.remove(l)
    #WALLS[id].linkIndex

    cutP = rate*WALLS[id].q + (1-rate)*WALLS[id].p
    A = len(WALLS)
    WALLS.append(
        wall(cutP,cutP,np.cross(WALLS[id].n,np.array([0,1,0])),id,A+1,A)
    )
    WALLS.append(
        wall(cutP,WALLS[id].q,WALLS[id].n,A,WALLS[id].w2,A+1)
    )
    for l in delList:
        WALLS[A+1].linkIndex.append(l)
    WALLS[WALLS[id].w2].w1= A+1
    WALLS[id].q = np.copy(cutP)
    WALLS[id].w2= A
    pass

def deleteWall(id,lst=WALLS):
    assert lst[id].length < 0.001
    lst[id].v = False
    lst[lst[id].w1].q = np.copy(lst[lst[id].w2].p)
    lst[lst[id].w1].w2 = lst[id].w2
    lst[lst[id].w2].w1 = lst[id].w1
    pass

def minusWall(id,lst=WALLS):
    if not lst[id].v:
        return lst
    # print(str(id) + " local")
    # for I in range(len(lst)):
    #     print(lst[I])
    if abs(lst[id].n @ lst[lst[id].w2].n + 1.)< 0.001:
        I = id
        J = lst[id].w2
    elif abs(lst[id].n @ lst[lst[id].w1].n + 1.)< 0.001:
        I = lst[id].w1
        J = id
    else:
        return lst # print(I,J)
    
    if lst[I].length < lst[J].length:
        P = lst[I].p
        K = lst[I].w1
    else:
        P = lst[J].q
        K = lst[J].w2
    lst[I].adjustWall(lst[I].q,P)
    lst[J].adjustWall(lst[J].p,P)
    
    if lst[I].length < 0.001 and lst[J].length < 0.001:
        lst[I].v = False
        lst[J].v = False
        lst[lst[J].w2].v=False
        lst[lst[I].w1].q = np.copy(lst[lst[J].w2].q)
        lst[lst[I].w1].w2 = lst[J].w2
        lst[lst[J].w2].w1 = lst[I].w1
        lst[lst[I].w1].lengthh()
    elif lst[I].length < 0.001:
        deleteWall(I,lst)
    elif lst[J].length < 0.001:
        deleteWall(J,lst)
    # print(str(id) + " local")
    # for I in range(len(lst)):
    #     print(lst[I])
    lst = minusWall(K,lst)
    return lst

    