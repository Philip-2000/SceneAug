import numpy as np
from statistics import mean
import json
import os
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Grup import *
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


D=2
DBGPRT=False
def dis(a,b):
    global D
    return (a[0]-b[0])**2+(a[2]-b[2])**2 if D==2 else ((a-b)**2).sum()

#
def formGroup(name):
    storeScene(name,True,False)
    if name.find("living") >= 0:
        chairs=[ o.idx for o in OBJES if o.class_name()=='dining_chair']
        cchair=[ o.idx for o in OBJES if o.class_name()=="lounge_chair"]
        if len(chairs)>=2 and len(cchair)>=2:
            print("fuck")
        elif len(cchair)>=2:
            chairs=cchair
        elif len(chairs)<2 and len(cchair)<2:
            #print("no dining")
            GRUPS.append(grup([i for i in range(len(OBJES))]))#return [[i for i in range(len(OBJES))]]
        
        #find the dinning_table/coffee_table surrouded by the chairs
        chairsTr = [OBJES[i].translation for i in chairs]
        chairsTrMid = np.average(np.array(chairsTr),axis=0)
        ONE = chairs 

        tables = [ o.idx for o in OBJES if (o.class_name() in ["dining_table","coffee_table"])]
        if len(tables)>0:
            T = tables[np.argmin(np.array([dis(OBJES[t].translation,chairsTrMid) for t in tables]))]
            LT = np.min(np.array([dis(OBJES[t].translation,chairsTrMid) for t in tables]))
            if LT < 0.6:
                ONE.append(T)

        lamps = [ o.idx for o in OBJES if (o.class_name() in ["ceiling_lamp","pendant_lamp"])]
        if len(lamps)>0:
            L = lamps[np.argmin(np.array([dis(OBJES[t].translation,chairsTrMid) for t in lamps]))]
            LL = np.min(np.array([dis(OBJES[t].translation,chairsTrMid) for t in lamps]))
            if LL < 0.6:
                ONE.append(L)

        TWO = [i for i in range(len(OBJES)) if not(i in ONE)]
        TWOTypes = [OBJES[i].class_name() for i in TWO]
        if not( ("l_shaped_sofa" in TWOTypes) or ("multi_seat_sofa" in TWOTypes) or ("loveseat_sofa" in TWOTypes) ):
            GRUPS.append(grup([i for i in range(len(OBJES))]))#return [[i for i in range(len(OBJES))]]
            
        twoTrMid = np.average(np.array([OBJES[i].translation for i in TWO]),axis=0)
        oneTrMid = np.average(np.array([OBJES[i].translation for i in ONE]),axis=0)

        toOne=[]
        for i in TWO:
            if dis(OBJES[i].translation,oneTrMid) < dis(OBJES[i].translation,twoTrMid):
                if (OBJES[i].translation[0]-oneTrMid[0])*(OBJES[i].translation[0]-twoTrMid[0])+(OBJES[i].translation[2]-oneTrMid[2])*(OBJES[i].translation[2]-twoTrMid[2]) > 0:
                    toOne.append(i)
        
        for i in toOne:
            TWO.remove(i)
            ONE.append(i)

        GRUPS.append(grup([i for i in ONE]))
        GRUPS.append(grup([i for i in TWO]))#return [ONE,TWO]
    elif name.find("bed") >= 0:
        pass

def adjustGroup():
    for g in GRUPS:
        g.adjust([0,0,0],[1,0,0])
    
    pass

def wrapScene():
    #说白了就是当前这么一个GRUPS状态应该怎么包裹我们的墙壁呢？
    
    #啊啊啊
    
    #他们都在哪里呀？

    #哪里需要裹一下墙，哪里最好不要裹。

    pass

def main():
    for n in os.listdir("./")[:20]:
        if (not n.endswith(".png")) or n.endswith("after.png") or n.endswith("before.png"):
            continue
        formGroup(n[:-4])
        adjustGroup()
        #mv = createMovements()
        #adjustScene(mv)
        wrapScene()
        storedDraw()
        clearScene()
        #print(n)
        #break
    pass
    

#this graph can also be broken right?

#select a wall, a length? ||||||||| select a part of a wall, and check a length to move it?


#would door or window break by this movement? #would collision occurs? #would path still exist? 



#this tree is pre-calculated and stored? maybe.


#our augmentation based on trees?
#place the groups and wrap the walls?!!!!!!!!
#No idea anyway, fuck. 

if __name__ == "__main__":
    main()

#what?

#for those not 