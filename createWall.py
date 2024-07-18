import numpy as np
from statistics import mean
import json
import sys
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

def stickWall(w,x):
    wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
    return w.v and x.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

def crossWall(w,x):
    wxp,wxq,xwp,xwq,wwp,xxp = w.n@x.p, w.n@x.q, x.n@w.p, x.n@w.q, w.n@w.p, x.n@x.q
    return w.v and x.v and (min(wxp,wxq)<wwp and wwp<max(wxp,wxq) and min(xwp,xwq)<xxp and xxp<max(xwp,xwq))

def crossCheck():
    for i in range(len(WALLS)):
        for j in range(i):
            if crossWall(WALLS[i], WALLS[j]):
                return True
    return False

F=-1
def postProcess():
    global F
    I = -1
    for i in range(len(WALLS)):
        if WALLS[i].v and WALLS[i].length < 0.7:
            I=i
    while I > 0:#print(F)
        #print(str(F)+"postProcess")
        L1 = WALLS[I].length/2.0 if (WALLS[I].q-WALLS[I].p) @ WALLS[WALLS[I].w1].n > 0 else -WALLS[I].length/2.0
        L2 = WALLS[I].length/2.0 if (WALLS[I].p-WALLS[I].q) @ WALLS[WALLS[I].w2].n > 0 else -WALLS[I].length/2.0
        WALLS[WALLS[I].w1].mWall(L1)
        WALLS[WALLS[I].w2].mWall(L2)
        w = WALLS[WALLS[I].w1]
        w.q = np.copy(WALLS[WALLS[I].w2].q)
        w.lengthh()
        w.w2= WALLS[WALLS[I].w2].w2
        WALLS[w.w2].w1 = w.idx
        if (w.p-w.q)[0]*w.n[2] > (w.p-w.q)[2]*w.n[0]:
            w.n = -w.n
        WALLS[I].v=False
        WALLS[WALLS[I].w2].v=False
        LOGS.append({"id":I,"delete":0})
        
        I = -1
        for i in range(len(WALLS)):
            if WALLS[i].v and WALLS[i].length < 0.7:
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
            wls = sorted([w.idx for w in WALLS if w.v], key=lambda x:-WALLS[x].length)
            wid = 0
            while wid < len(wls)-1 and WALLS[wls[wid]].length > 1.5 and np.random.rand()<0.5:
                wid += 1
            wid = wls[wid]
            L = WALLS[wid].length
            r = np.random.uniform(0.9 / L, 1.0 - (0.9 / L)) #0.5+(t-0.5)*0.5
            breakWall(wid,r)
            LOGS.append({"id":wid,"rate":r})
            # if False and (WALLS[WALLS[wid].w1].p - WALLS[wid].p) @ WALLS[wid].n < 0:
            #     assert -WALLS[WALLS[wid].w1].length+1.0 < -1.0
            #     t = np.random.uniform(-WALLS[WALLS[wid].w1].length+1.0, -1.0)
            # else:
            #     t = np.random.uniform(-4.0, -1.0)
            lower = min(maxDepth(WALLS[wid])+1.0,0.0) # <0     #最好是maxHeight(WALLS[wid])+？？？ ？越大越好
            upper = 0.0#maxHeight(WALLS[wid])-1.0# >0      #最好是 maxDepth(WALLS[wid])-？？？ ？越大越好
            t = np.random.uniform(lower, upper)
            WALLS[wid].mWall(t)
            LOGS.append({"id":wid,"leng":t,"lower":lower,"upper":upper})
        else: 
            wid = np.random.randint(len(WALLS))
            lower = min(maxDepth(WALLS[wid])+1.0,0.0) # <0
            upper = 0.0#maxHeight(WALLS[wid])-1.0# >0
            t = np.random.uniform(lower, upper)
            LOGS.append({"id":wid,"leng":t,"lower":lower,"upper":upper})
            WALLS[wid].mWall(t)
            # if WALLS[WALLS[wid].w1].n @ WALLS[WALLS[wid].w2].n > 0:
            #     if (WALLS[WALLS[wid].w1].p - WALLS[wid].p) @ WALLS[wid].n > 0:
            #         t = np.random.uniform(-WALLS[WALLS[wid].w2].length+1.0, WALLS[WALLS[wid].w1].length-1.0)
            #         #assert -WALLS[WALLS[wid].w2].length+1.0 < WALLS[WALLS[wid].w1].length-1.0
            #         LOGS.append({"id":wid,"leng":t, "lower":-WALLS[WALLS[wid].w2].length+1.0, "upper":WALLS[WALLS[wid].w1].length-1.0})
            #     else:
            #         t = np.random.uniform(-WALLS[WALLS[wid].w1].length+1.0, WALLS[WALLS[wid].w2].length-1.0)
            #         #assert -WALLS[WALLS[wid].w1].length+1.0 < WALLS[WALLS[wid].w2].length-1.0
            #         LOGS.append({"id":wid,"leng":t, "lower":-WALLS[WALLS[wid].w1].length+1.0, "upper":WALLS[WALLS[wid].w2].length-1.0})
            #     WALLS[wid].mWall(t)
            # else:
            #     t = np.random.uniform(-4.0, -1.0)
            #     WALLS[wid].mWall(t)
            #     LOGS.append({"id":wid,"leng":t, "lower":0, "upper":0})
        
        if crossCheck():
            J = min([w.idx for w in WALLS if w.v])#WALLS[0].w2
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

def maxHeight(x, bd=4.0, lst=WALLS):
    ls = [(w.p-x.p)@x.n for w in lst if (w.v and ((w.p-x.p)@x.n > 0.01))]
    #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值
    return bd if len(ls)==0 else min(min(ls),bd) 

def maxDepth(x, bd = -4.0, lst=WALLS):
    ls = [(w.p-x.p)@x.n for w in lst if (w.v and ((w.p-x.p)@x.n <-0.01))]
    #扫描所有墙面，如果两侧小于id的两侧并且方向和它不垂直，那么他到id的垂向上的距离的较小值就作为height。统计这些height中的最小值
    return bd if len(ls)==0 else max(max(ls),bd) 

Falls=[]

def spaceExtract():
    #我觉得就是找个U型的东西
    #找一下
    global Falls
    global F
    if len([_ for _ in Falls if _.v]) == 0:
        return "fuck"
    candidates = []
    
    J = min([w.idx for w in Falls if w.v])#WALLS[0].w2
    I = Falls[J].w2
    while I != J:
        I = Falls[I].w2
        if (not Falls[I].spaceIn)and (Falls[Falls[I].w1].n @ Falls[Falls[I].w2].n < 0) and (Falls[I].n @ (Falls[Falls[I].w1].p-Falls[Falls[I].w1].q)>0):
            candidates.append(I)
            #if max(Falls[Falls[I].w1].length, Falls[Falls[I].w2].length)>0.5*Falls[I].length:
            #    if min(Falls[Falls[I].w1].length, Falls[Falls[I].w2].length)>0.66*Falls[I].length:
            #        if min(Falls[Falls[I].w1].length, Falls[Falls[I].w2].length)<1.5*Falls[I].length:
            #            candidates.append(I)
    
    if len(candidates)==0:
        return
    candidates.sort(key=lambda x:-Falls[x].length) 
    idx = 0
    I = candidates[idx]
    maxH = maxHeight(Falls[I],-4.0,Falls)
    minL = min(Falls[Falls[I].w1].length,Falls[Falls[I].w2].length)
    if maxH < minL and False:
        print(F)
        print(I)
        print(maxH)
        print(minL)
        print(str(F)+" hey")


    while maxH < minL and abs(minL/Falls[I].length - 1.0) > 0.4 and idx < len(candidates)-1:
        idx += 1
        I = candidates[idx]
        maxH = maxHeight(Falls[I],-4.0,Falls)
        minL = min(Falls[Falls[I].w1].length,Falls[Falls[I].w2].length)
    # print(Falls[I].length)
    # print(Falls[Falls[I].w1].length)
    # print(Falls[Falls[I].w2].length)
    # print(minL)
    # print(abs(minL/Falls[I].length - 1.0))
    if idx >= len(candidates):
        return
        
    
    if Falls[Falls[I].w1].length < Falls[Falls[I].w2].length:
        X = Falls[Falls[I].w1]
        W = Falls[I]
    else:
        W = Falls[Falls[I].w2]
        X = Falls[I]
    P = W.p
    W.spaceIn = True
    # Falls[Falls[I].w1].spaceIn = True
    # Falls[Falls[I].w2].spaceIn = True
    X.spaceIn = True
    B = X.length
    A = W.length
    Qs = P + X.n*A + W.n*B
    #X:[P+W.n*B,P] -> W:[P,P+X.n*A]
    X.w2 = len(Falls)
    Falls.append(wall(P,P+W.n*B,-X.n,X.idx,len(Falls)+1,len(Falls)))
    Falls.append(wall(P+W.n*B,Qs,W.n,len(Falls)-1,len(Falls)+1,len(Falls)))
    Falls.append(wall(Qs,P+X.n*A,X.n,len(Falls)-1,len(Falls)+1,len(Falls)))
    W.w1 = len(Falls)
    Falls.append(wall(P+X.n*A,P,-W.n,len(Falls)-1,W.idx,len(Falls)))


    # print("global")
    # for I in range(len(Falls)):
    #     print(Falls[I])
    Falls = minusWall(X.idx,Falls)
    # print("global")
    # for I in range(len(Falls)):
    #     print(Falls[I])
    Falls = minusWall(W.idx,Falls)

    SPCES.append(spce(P,Qs,len(SPCES)))

    return "fuck"


def spaceExtracts():
    global F
    for w in WALLS:
        Falls.append(wall(w.p, w.q, w.n, w.w1, w.w2, w.idx, w.v, w.spaceIn, F))
    #print(F)
    #return
    f = open("./newRooms/"+str(F)+".txt","w")
    for w in WALLS:
        f.write(str(w)+"\n")
    f.close()
    f = open("./newRooms/log"+str(F)+".txt","w")
    for l in LOGS:
        if "rate" in l:
            f.write(str(l["id"])+"::"+str(l["rate"])+"\n")
        elif "delete" in l:
            f.write(str(l["id"])+"delete\n")
        else:
            f.write(str(l["id"])+"=="+str(l["lower"])+"<<"+str(l["leng"])+">>"+str(l["upper"])+"\n")
    f.close()
    return
    spaceExtract()
    if np.random.rand()>-0.5:
        spaceExtract()
    
    for s in SPCES:
        s.adjustSize()

def main():
    global F
    global Falls
    for n in range(100):        
        init()
        F = n
        Falls = []
        a = change()
        spaceExtracts()
        storedDraw(lim=10)

        # if len([_ for _ in Falls if _.v]) > 0:
        #     J = min([w.idx for w in Falls if w.v])#WALLS[0].w2
        #     contour,w =[[Falls[J].p[0],Falls[J].p[2]]], Falls[J].w2
        #     while w != J:
        #         contour.append([Falls[w].p[0],Falls[w].p[2]])
        #         w = Falls[w].w2
        #     contour = np.array(contour)
        #     plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o")


        plt.savefig("./newRooms/" + str(n) + ".png")
        plt.clf()
        clearScene()
        if a:
            print(n)
            break
    pass
    


if __name__ == "__main__":
    main()

#what?

#for those not 