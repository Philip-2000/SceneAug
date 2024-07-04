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

def change():
    a = 5
    while np.random.rand() < 0.8 and a > 0:
        a -= 1
        if np.random.rand() < 0.5:
            wls = sorted([w.idx for w in WALLS], key=lambda x:-WALLS[x].length)
            wid = 0
            while wid < len(wls)-1 and WALLS[wls[wid]].length > 1.5 and np.random.rand()<0.5:
                wid += 1
            L = WALLS[wls[wid]].length
            t = np.random.uniform(0.4 / L, 1.0 - (0.4 / L)) #0.5+(t-0.5)*0.5
            breakWall(wls[wid],t)
            t = np.random.rand()+0.5
            WALLS[wls[wid]].mWall(-2.0*t)
        else: 
            wid = np.random.randint(len(WALLS))
            if WALLS[WALLS[wid].w1].n @ WALLS[WALLS[wid].w2].n > 0:
                if (WALLS[WALLS[wid].w1].p - WALLS[WALLS[wid].w1].q) @ WALLS[wid].n > 0:
                    t = np.random.uniform(-WALLS[WALLS[wid].w2].length+0.5, WALLS[WALLS[wid].w1].length-0.5)
                else:
                    t = np.random.uniform(-WALLS[WALLS[wid].w1].length+0.5, WALLS[WALLS[wid].w2].length-0.5)
                WALLS[wid].mWall(t)
            else:
                t = np.random.rand()+0.5
                WALLS[wid].mWall(-2.0*t)

    #for w in WALLS:print(str(w.w1)+"<-"+str(w.idx)+"->"+str(w.w2),w.p,w.n)

def main():
    for n in range(20):        
        init()
        change()
        storedDraw(lim=10)
        plt.savefig("./newRooms/" + str(n) + ".png")
        plt.clf()
        clearScene()
        #print(n)
        #break
    pass
    


if __name__ == "__main__":
    main()

#what?

#for those not 