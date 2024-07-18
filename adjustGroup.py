import numpy as np
from statistics import mean
import json
import os
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Grup import *
from classes.Wall import *
from classes import *
from util import two23,fullLoadScene,storeScene,storedDraw,clearScene,draftRoomMask

#dir = "C:/Users/win/Downloads/3d_front_processed/livingrooms_objfeats_32_64/"
dir = "../novel3DFront/"
#construct the scene graph based on wall,

# wall0                  wall1                 wall2               wall3
# obj1-   obj4-          ...                    obj2-               obj3-

D=2
DBGPRT=False
def dis(a,b):
    global D
    return (a[0]-b[0])**2+(a[2]-b[2])**2 if D==2 else ((a-b)**2).sum()

#

def storeGroup(n):
    if not (os.path.exists("../novel3DFront_grp1/"+n)):
        os.mkdir("../novel3DFront_grp1/"+n)
    np.savez_compressed("../novel3DFront_grp1/"+n+"/group.npz", group=np.array([o.gid for o in OBJES],dtype=int))

def adjustGroups(src,dst,cn):
    cnt = cn
    if cn==-1:
        for o in OBJES:
            if o.gid == src:
                o.gid = dst
        return

    while cnt > 0:
        cnt -= 1
        srcId = [o.idx for o in OBJES if o.gid == src]
        dstId = [o.idx for o in OBJES if o.gid == dst]
        if src != -1:
            #从src里找到距离其他东西最远的一个？
            srcTrMid = np.average(np.array([OBJES[i].translation for i in srcId]),axis=0)#print(twoTrMid)
            srcDis = [dis(OBJES[i].translation,srcTrMid) for i in srcId]
            I = np.argmax(np.array(srcDis))
        else:
            #从找src里到距离dst的最近距离最近的一个？
            mat = [min([dis(OBJES[i].translation,OBJES[j].translation) for i in dstId]) for j in srcId]
            I = np.argmin(np.array(mat))
        
        OBJES[srcId[I]].gid = dst

def adjustGroup(src,dst,cn):
    cnt = cn
    if cn==-1:
        for o in OBJES:
            if o.gid == src:
                o.gid = dst
        return

    while cnt > 0:
        cnt -= 1
        srcId = [o.idx for o in OBJES if o.gid == src]
        dstId = [o.idx for o in OBJES if o.gid == dst]
        if dst == -1:
            #从src里找到距离其他东西最远的一个？
            srcTrMid = np.average(np.array([OBJES[i].translation for i in srcId]),axis=0)#print(twoTrMid)
            srcDis = [dis(OBJES[i].translation,srcTrMid) for i in srcId]
            I = np.argmax(np.array(srcDis))
        else:
            #从src里找到距离dst的最近距离最近的一个？
            mat = [min([dis(OBJES[i].translation,OBJES[j].translation) for i in dstId]) for j in srcId]
            I = np.argmin(np.array(mat))
        
        OBJES[srcId[I]].gid = dst

sigs = []
def constructGroup(dir,cnt,idx,fil=0):
    global sigs
    if dir == '.':
        s = sorted([[o.translation[2],o.idx] for o in OBJES if o.idx in sigs],key=lambda x:-x[0])
    elif dir == ';':
        s = sorted([[o.translation[2],o.idx] for o in OBJES if o.idx in sigs],key=lambda x:x[0])
    elif dir == ',':
        s = sorted([[o.translation[0],o.idx] for o in OBJES if o.idx in sigs],key=lambda x:x[0])
    elif dir == '/':
        s = sorted([[o.translation[0],o.idx] for o in OBJES if o.idx in sigs],key=lambda x:-x[0])
    else:
        return
    for i in range(cnt if cnt >= 0 else len(s)):
        sigs.remove(s[i][1])
        if i >= fil:
            OBJES[s[i][1]].gid = idx

fl=True
def clearGroup():
    global GRUPS
    global fl
    if fl:
        GRUPS = []
        for o in OBJES:
            o.gid = -1
        fl = False

def main():
    global sigs
    global fl
    f = open("./segment4/hintt.txt")
    lines = f.readlines()
    f.close()
    for lin in lines:
        k = lin.split("\t")
        print(k[0])
        fl=True

        storeScene(k[0],False,True,True)
        sigs = [o.idx for o in OBJES]
        c = 0
        for i in range(1,len(k)):
            if len(k[i]) == 0:
                continue
            m = k[i].split(' ')
            if m[0][0] in [',','.','/',';']:
                clearGroup()
                #adjustGroups(int(m[0]),int(m[1]),int(m[2]))
                if len(m[0])>=3 and m[0][2] == '=':
                    constructGroup(m[0][0],int(m[0][1:2]),c if len(m)==1 else int(m[1]),int(m[0][3:]))
                else:
                    constructGroup(m[0][0],int(m[0][1:]),c if len(m)==1 else int(m[1]))
            else:
                adjustGroups(int(m[0]),int(m[1]),int(m[2]))
            c+=1
        storeGroup(k[0])
        storedDraw(drawWall=True,objectGroup=True)
        plt.savefig("./segment5/" + k[0] + ".png")
        plt.clf()
        #draftRoomMask(n)
        clearScene()

def mains():
    for n in os.listdir("../novel3DFront_grp"):
        storeScene(n,False,True,True)
        print(n)
        storedDraw(drawWall=True,objectGroup=True)
        plt.savefig("./segment3/" + n + ".png")
        plt.clf()
        #draftRoomMask(n)
        clearScene()
    

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