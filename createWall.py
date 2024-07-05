import numpy as np
from statistics import mean
import json
import os
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Grup import *
from classes.Spce import *
from classes.Wall import *
from classes import *
from util import two23,fullLoadScene,storeScene,storedDraw,clearScene

#dir = "C:/Users/win/Downloads/3d_front_processed/livingrooms_objfeats_32_64/"
dir = "../novel3DFront/"
#construct the scene graph based on wall,

# wall0                  wall1                 wall2               wall3
# obj1-   obj4-          ...                    obj2-               obj3-

common_links={
    "Dining Chair":["Dining Table"],
    "Chaise Longue Sofa":["Coffee Table"],
    "Coffee Table":["Lazy Sofa","Three-seat / Multi-seat Sofa","Loveseat Sofa","L-shaped Sofa"],
    "Dressing Chair":["Dressing Table"],
    "Pendant Lamp":["King-size Bed", "Kids Bed", "Single bed", "Coffee Table", "Dining Table"],
    "Nightstand":["King-size Bed", "Kids Bed", "Single bed"]
}

#create empty room considering blank spaces, doorways, and windows
#what is blank space
#no, if you go through this aspect, it's probably the same with the warpWall
#so we should go through another way

#from [-1,1], [1,1], [1,-1] and [-1,-1]
#a 

#what is doorway

#what is light, it is a part of the wall

def init():
    WALLS.append(wall(np.array([2.,0.,2.]),np.array([-2.,0.,2.]),np.array([0.,0.,-1.]),3,1,0))
    WALLS.append(wall(np.array([-2.,0.,2.]),np.array([-2.,0.,-2.]),np.array([1.,0.,0.]),0,2,1))
    WALLS.append(wall(np.array([-2.,0.,-2.]),np.array([ 2.,0.,-2.]),np.array([0.,0.,1.]),1,3,2))
    WALLS.append(wall(np.array([2.,0.,-2.]),np.array([ 2.,0.,2.]),np.array([-1.,0.,0.]),2,0,3))


def crossWall(w,x):
    wxp,wxq,xwp,xwq,wwp,xxp = w.n@x.p, w.n@x.q, x.n@w.p, x.n@w.q, w.n@w.p, x.n@x.q
    return w.valid and x.valid and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))

def crossCheck():
    for i in range(len(WALLS)):
        for j in range(i):
            if crossWall(WALLS[i], WALLS[j]):
                return True
    return False

F=-1
def postProcess():
    I = -1
    for i in range(len(WALLS)):
        if WALLS[i].valid and WALLS[i].length < 0.7:
            I=i
    while I > 0:#print(F)
        L1 = WALLS[I].length/2.0 if (WALLS[I].p-WALLS[I].q) @ WALLS[WALLS[I].w2].n < 0 else -WALLS[I].length/2.0
        L2 = WALLS[I].length/2.0 if (WALLS[I].p-WALLS[I].q) @ WALLS[WALLS[I].w1].n > 0 else -WALLS[I].length/2.0
        WALLS[WALLS[I].w1].mWall(L1)
        WALLS[WALLS[I].w2].mWall(L2)
        w = WALLS[WALLS[I].w1]
        w.q = WALLS[WALLS[I].w2].p
        w.w2= WALLS[WALLS[I].w2].w2
        WALLS[w.w2].w1 = w.idx
        if (w.p-w.q)[0]*w.n[2] > (w.p-w.q)[2]*w.n[0]:
            w.n = -w.n
        WALLS[I].valid=False
        WALLS[WALLS[I].w2].valid=False
        
        I = -1
        for i in range(len(WALLS)):
            if WALLS[i].valid and WALLS[i].length < 0.7:
                I=i

LOGS=[]
def change():
    global LOGS
    LOGS = []
    a,b = 5,3
    while np.random.rand() < 0.8 and a > 0 and b > 0:
        a -= 1
        if np.random.rand() < 0.5:
            b -= 1
            wls = sorted([w.idx for w in WALLS if w.valid], key=lambda x:-WALLS[x].length)
            wid = 0
            while wid < len(wls)-1 and WALLS[wls[wid]].length > 1.5 and np.random.rand()<0.5:
                wid += 1
            wid = wls[wid]
            L = WALLS[wid].length
            r = np.random.uniform(0.9 / L, 1.0 - (0.9 / L)) #0.5+(t-0.5)*0.5
            breakWall(wid,r)
            LOGS.append({"id":wid,"rate":r})
            if False and (WALLS[WALLS[wid].w1].p - WALLS[wid].p) @ WALLS[wid].n < 0:
                assert -WALLS[WALLS[wid].w1].length+1.0 < -1.0
                t = np.random.uniform(-WALLS[WALLS[wid].w1].length+1.0, -1.0)
            else:
                t = np.random.uniform(-4.0, -1.0)
            WALLS[wid].mWall(t)
            LOGS.append({"id":wid,"leng":t, "lower":0, "upper":0})
        else: 
            wid = np.random.randint(len(WALLS))
            if WALLS[WALLS[wid].w1].n @ WALLS[WALLS[wid].w2].n > 0:
                if (WALLS[WALLS[wid].w1].p - WALLS[wid].p) @ WALLS[wid].n > 0:
                    t = np.random.uniform(-WALLS[WALLS[wid].w2].length+1.0, WALLS[WALLS[wid].w1].length-1.0)
                    #assert -WALLS[WALLS[wid].w2].length+1.0 < WALLS[WALLS[wid].w1].length-1.0
                    LOGS.append({"id":wid,"leng":t, "lower":-WALLS[WALLS[wid].w2].length+1.0, "upper":WALLS[WALLS[wid].w1].length-1.0})
                else:
                    t = np.random.uniform(-WALLS[WALLS[wid].w1].length+1.0, WALLS[WALLS[wid].w2].length-1.0)
                    #assert -WALLS[WALLS[wid].w1].length+1.0 < WALLS[WALLS[wid].w2].length-1.0
                    LOGS.append({"id":wid,"leng":t, "lower":-WALLS[WALLS[wid].w1].length+1.0, "upper":WALLS[WALLS[wid].w2].length-1.0})
                WALLS[wid].mWall(t)
            else:
                t = np.random.uniform(-4.0, -1.0)
                WALLS[wid].mWall(t)
                LOGS.append({"id":wid,"leng":t, "lower":0, "upper":0})
        
        if crossCheck():
            J = min([w.idx for w in WALLS if w.valid])#WALLS[0].w2
            I = WALLS[J].w2
            while I != J:
                I = WALLS[I].w2
                print(str(WALLS[I].w1)+"<-"+str(WALLS[I].idx)+"->"+str(WALLS[I].w2),WALLS[I].p,WALLS[I].n)
                
            for l in LOGS:
                if "rate" in l:
                    print(str(l["id"])+"::"+str(l["rate"]))
                else:
                    print(str(l["id"])+"=="+str(l["lower"])+"<<"+str(l["leng"])+">>"+str(l["upper"]))
            return True
    
    postProcess()

    return False

def spaceExtract():
    #我觉得就是找个U型的东西
    #找一下
    candidates = []
    J = min([w.idx for w in WALLS if w.valid])#WALLS[0].w2
    I = WALLS[J].w2
    while I != J:
        I = WALLS[I].w2
        if (not WALLS[I].spaceIn)and (WALLS[WALLS[I].w1].n @ WALLS[WALLS[I].w2].n < 0) and (WALLS[I].n @ (WALLS[WALLS[I].w1].p-WALLS[WALLS[I].w1].q)>0):
            if max(WALLS[WALLS[I].w1].length, WALLS[WALLS[I].w2].length)>0.75*WALLS[I].length:
                candidates.append(I)
        
    I = candidates[0]
        
    A = WALLS[I].length
    W = WALLS[I]
    if WALLS[WALLS[I].w1].length < WALLS[WALLS[I].w2].length:
        P = WALLS[I].p
        X = WALLS[WALLS[I].w1]
    else:
        P = WALLS[I].q
        X = WALLS[WALLS[I].w2]
    W.spaceIn = True
    X.spaceIn = True
    B = X.length
    Qbox = {}
    Qs = P + X.n*A + W.n*B
    Qbox["minX"] = min(Qs[0],P[0])
    Qbox["minZ"] = min(Qs[2],P[2])
    Qbox["maxX"] = max(Qs[0],P[0])
    Qbox["maxZ"] = max(Qs[2],P[2])
    Qbox["fixedX"] = P[0]
    Qbox["fixedZ"] = P[2]
    Qbox["signX"] = -1 if P[0]<Qs[0] else 1
    Qbox["signZ"] = -1 if P[2]<Qs[2] else 1
    pass
    
    #P is the starting point of this space
    #Q is an important thing, there are lots of constraints, from the walls, and other spaces
        #Firstly, we think Q is on the other side of P
        #Then we think Q should be inside the walls
        #Other spaces cut the valid spaces of current one.
        #fuck!
    #for s in SPCES:
        #Qbox = adjacent(s,Qbox)
    #Are you going to squeeze this box?
    SPCES.append(spce(P,Qs))

    return "fuck"
    pass

def spaceExtracts():
    spaceExtract()

    pass

def main():
    global F
    for n in range(10):        
        init()
        F = n
        a = change()
        spaceExtracts()
        storedDraw(lim=10)
        plt.savefig("./newRooms/" + str(n) + ".png")
        plt.clf()
        clearScene()
        if a:
            print(n)
            break
        #print(n)
        #break
    pass
    


if __name__ == "__main__":
    main()

#what?

#for those not 