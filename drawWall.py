import numpy as np
from statistics import mean
import json
import os
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Link import *
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


def formGraph(name):
    storeScene(name)

    #把经典关系标示出来
    for oi in range(len(OBJES)):
        o = OBJES[oi]
        shortest = 3
        RI = -1
        for ri in range(len(OBJES)):
            r = OBJES[ri]
            if (oi == ri) or not(o.class_name() in common_links.keys() and r.class_name() in common_links[o.class_name()]):
                continue #semantic check print(o.class_name()+"   "+r.class_name())
            Tor = r.translation - o.translation
            Tor[1] = 0
            Lor = (Tor**2).sum()**0.5 + 0.0001
            Ior = Tor / Lor
            if Lor < shortest:
                shortest = Lor
                RI = ri
        if RI >= 0:
            LINKS.append(objLink(RI,oi,len(LINKS)))

    #把贴墙关系都标识出来
    for oi in range(len(OBJES)):
        o = OBJES[oi]
        if len(o.destIndex)>0:
            continue
        om = o.matrix()
        for wi in range(len(WALLS)):
            w = WALLS[wi]
            #translate w.p into o's co-ordinate, scaled
            p=(om @ (w.p-o.translation)) / o.size
            #translate w.q into o's co-ordinate, scaled
            q=(om @ (w.q-o.translation)) / o.size
            #get a distance and projection
            n=(om @ w.n)

            pp=np.cross(n,p)[1]#min(np.cross(n,p),np.cross(n,q))
            qq=np.cross(n,q)[1]#max(np.cross(n,p),np.cross(n,q))
            
            if(wi==4 and oi == 5) and False:
                print(w.p)
                print(w.q)
                print(w.n)
                print(o.class_name())
                print(o.translation)
                print(" ")
                print(p)
                print(q)
                print(n)
                print(pp)
                print(qq)
                
                pass

            if abs(abs(p@n)-1.0)<0.1 and min(pp,qq) < 0.9 and max(pp,qq) > -0.9:
                LINKS.append(walLink(wi,oi,len(LINKS),o.translation))#a,b = LINKS[-1].arrow()
            
            elif abs((p*o.size-o.size)@n)<0.1 and min(pp,qq) < 0.0 and max(pp,qq) > -0.0:
                LINKS.append(walLink(wi,oi,len(LINKS),o.translation))#a,b = LINKS[-1].arrow()

    #物体指向它朝向的一个东西
    for oi in range(len(OBJES)):
        o = OBJES[oi] #print("\n"+o.class_name() + "\n")
        if len(o.destIndex)>0:
            continue
        shortest = 3
        RI = -1
        for ri in range(len(OBJES)):
            r = OBJES[ri] #print(r.class_name())
            if ri == oi: #len(r.destIndex)==0:
                continue 
            Tor = r.translation - o.translation
            Tor[1] = 0
            Lor = (Tor**2).sum()**0.5 + 0.0001
            Ior = Tor / Lor
            #find the facing one
            if (Ior @ o.direction()) > 0.5:# or True: #print("     ??? "+r.class_name())
                if Lor < shortest:
                    shortest = Lor
                    RI = ri
        if RI >= 0:
            LINKS.append(objLink(RI,oi,len(LINKS)))#print("    "+OBJES[RI].class_name())

def visualizeGraph(name, dstDir="./"):
    #visualize the OBJES, WALLS, especially the LINKS
    storedDraw()
    #singleDraw(name)
    for li in LINKS:
        src,dst = li.arrow()
        plt.plot([dst[0]], [-dst[2]], marker="x")
        plt.plot([src[0], dst[0]], [-src[2], -dst[2]], marker=".")

    plt.savefig(dstDir + name + ".png")
    plt.clf()
    pass

def recursiveRange(o,wid):
    mis,mas = o.project(wid)
    for l in o.linkIndex:
        mi,ma = recursiveRange(OBJES[LINKS[l].dst],wid)
        mis = min(mis,mi)
        mas = max(mas,ma)
    return mis,mas

def createMovements():
    #move = {id:0,length:1}
    #move = {id:0,rate:0.5}
    #what are we going to do?
    #find a legal break point. How to find the legal point?
    #scan the linked branches on this wall.
    wls = sorted([w.idx for w in WALLS if len(w.linkIndex)>0], key=lambda x:-WALLS[x].length)
    if len(wls)==0:
        return [{"id":0,"rate":0.5}, {"id":-1,"length":-0.5}]
    return [{"id":wls[0],"length":-1.5}]
    wid = 0
    while wid < len(wls)-1 and np.random.rand()<0.8:
        wid += 1
    wid = wls[wid]
    w = WALLS[wid]

    rs = [0,1]
    
    for l in w.linkIndex:
        mi,ma = recursiveRange(OBJES[LINKS[l].dst],wid)
        mi,ma = max(mi,0),min(ma,1)
        #如何将禁区mi和ma加进去
        #搜索mi所在的有效区下界，将有效区上界设置为mi。继续搜索，找到ma所在的有效区，将有效区下界设置为ma
        rss=[0.0]
        idx=1
        while idx+1<len(rs) and rs[idx+1]<mi:      #valid-lower == invalid-upper
            rss.append(rs[idx])  #valid-upper == invalid-lower
            rss.append(rs[idx+1])#valid-lower == invalid-upper
            idx+=2
        rss.append(min(mi,rs[idx]))
        while rs[idx]<ma:        #valid-upper == invalid-lower
            idx+=2
        rss.append(max(ma,rs[idx-1]))
        while idx < len(rs):
            rss.append(rs[idx])
            idx+=1
        rs=rss
    
    r = (0.5+(np.random.rand()-0.5)*0.5)*sum([rs[_+1]-rs[_] for _ in range(0,len(rs),2)])
    idx = 0
    while r > (rs[idx+1]-rs[idx]):
        r -= (rs[idx+1]-rs[idx])
        idx += 2

    return [{"id":wid,"rate":rs[idx]+r}, {"id":-1,"length":-0.5}]

def adjustScene(movements):
    for move in movements:
        if "rate" in move.keys():
            breakWall(move["id"],move["rate"])
        elif "length" in move.keys():
            WALLS[move["id"]].mWall(move["length"])
        # for w in WALLS:
        #     print(str(w.w1) + "<-" + str(w.idx) + "->" + str(w.w2))
        #     print(w.p)
        #     print(w.q)
        #     print(w.n)
        # print("\n")
    pass

def main():
    for n in os.listdir("./"):#[:20]:
        if (not n.endswith(".png")) or n.endswith("after.png") or n.endswith("before.png"):
            continue
        formGraph(n[:-4])
        #visualizeGraph(n[:-4])
        mv = createMovements()
        adjustScene(mv)
        visualizeGraph("after/"+n[:-4])
        clearScene()
        #print(n)
        #break
    

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