# I have to write something to place my feet back on the ground
import numpy as np
from statistics import mean
import json
import os
dir = "C:/Users/win/Downloads/3d_front_processed/livingrooms_objfeats_32_64/"
object_types=["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet"]

D=2
DBGPRT=False

def dis(a,b):
    global D
    if D==2:
        return dis2D(a,b)
    if D==3:
        return dis3D(a,b)

def dis3D(a,b):
    return ((a-b)**2).sum()

def dis2D(a,b):
    return (a[0]-b[0])**2+(a[2]-b[2])**2


PATTERNS = []
def patternMatching(name,p):
    #what is pattern
    #what is pattern
    #
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr = boxes["translations"]
    cl = boxes["class_labels"]
    
    #CHECK pattern p for scene boxes
    #extract the objects types fits pattern p
    #

    #then what
    #then place the co-ordinate to the box. with translation and orientation.
    
    #what would this translation be?

    #for i in range(len(tr)):
    #output what? output a group of objects translated and rotated into the new co-ordinates.
    
    pass

def addPattern(pp):
    global PATTERNS
    p={"object_types":[],"object_type_count":[]}
    idList = pp["idList"]
    boxes = pp["boxes"]
    cl = boxes["class_labels"]
    for i in idList:
        if not(object_types[np.argmax(cl[i])] in p["object_types"]):
            p["object_types"].append(object_types[np.argmax(cl[i])])

    #p["object_type_count"]
    cll = np.zeros_like(cl[0])
    for i in idList:
        cll += cl[i]
    p["object_type_count"].append(cll.tolist())
    PATTERNS.append(p)

def addToPattern(pid,pp):
    global PATTERNS
    p = PATTERNS[pid]
    idList = pp["idList"]
    boxes = pp["boxes"]
    cl = boxes["class_labels"]

    #p["object_types"]
    for i in idList:
        if not(object_types[np.argmax(cl[i])] in p["object_types"]):
            p["object_types"].append(object_types[np.argmax(cl[i])])

    #p["object_type_count"]
    cll = np.zeros_like(cl[0])
    for i in idList:
        cll += cl[i]
    p["object_type_count"].append(cll.tolist())
    
    #p["object_spatial_distribution"]?

    PATTERNS[pid] = p

def patternStorage(pp):
    #what is pp
    # pp is idList and boxes
    idList = pp["idList"]
    boxes = pp["boxes"]
    cl = boxes["class_labels"]
    global PATTERNS

    fitId = -1
    for pid in range(len(PATTERNS)):
        p=PATTERNS[pid]
        #check if pp and p matches
        fitType=0
        for i in idList:
            if object_types[np.argmax(cl[i])] in p["object_types"]:
                fitType+=1
        if len(idList)-fitType<2 and fitId>-1:
            print("fuck", fitId, pid)
            fitId = 10000
        if len(idList)-fitType<2 and fitId==-1:
            fitId = pid

    if fitId < 1000:
        if fitId == -1:
            addPattern(pp)
        else:
            addToPattern(fitId,pp)
    pass

def singleCluster(name):
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr = boxes["translations"]
    cl = boxes["class_labels"]
    patternId = [0 for _ in range(len(tr))]

    #np.savez_compressed(dir + name + ".npz", contour=contour)
    #If there are 2 clusters,
    I,J,L = 0,1,dis(tr[0],tr[1])#((tr[0]-tr[1])**2).sum()
    for i in range(len(tr)):
        for j in range(i):
            if dis(tr[i],tr[j])>L:#((tr[i]-tr[j])**2).sum() > L:
                I,J,L = i,j,dis(tr[i],tr[j])#((tr[i]-tr[j])**2).sum()
    if DBGPRT:
        print(object_types[np.argmax(cl[I])])
        print(object_types[np.argmax(cl[J])])

    patternI=[i for i in range(len(tr)) if dis(tr[i],tr[I])<dis(tr[i],tr[J])]#((tr[i]-tr[I])**2).sum() < ((tr[i]-tr[J])**2).sum()]
    patternJ=[i for i in range(len(tr)) if dis(tr[i],tr[I])>dis(tr[i],tr[J])]#((tr[i]-tr[I])**2).sum() > ((tr[i]-tr[J])**2).sum()]
    #return [patternI,patternJ]

    cenI=patternI[0]
    cenLI=1000
    if len(patternI)>1:
        for pi in range(len(patternI)):
            L = min([dis(tr[patternI[pi]],tr[patternI[pj]]) for pj in range(len(patternI)) if not(pj==pi)]) #((tr[patternI[pi]]-tr[patternI[pj]])**2).sum()
            if L < cenLI:
                cenLI = L
                cenI = patternI[pi]
    
    cenJ=patternJ[0]
    cenLJ=1000
    if len(patternJ)>1:
        for pi in range(len(patternJ)):
            L = min([dis(tr[patternJ[pi]],tr[patternJ[pj]]) for pj in range(len(patternJ)) if not(pj==pi)])#((tr[patternJ[pi]]-tr[patternJ[pj]])**2).sum()
            if L < cenLJ:
                cenLJ = L
                cenJ = patternJ[pi]
    if DBGPRT:
        print(object_types[np.argmax(cl[cenI])])
        print(object_types[np.argmax(cl[cenJ])])

    meanDistance = mean([mean([dis(tr[i],tr[j]) for i in patternI]) for j in patternJ]) #((tr[i]-tr[j])**2).sum()
    minDistance = min([min([dis(tr[i],tr[j]) for i in patternI]) for j in patternJ])
    cenDistance = dis(tr[cenI],tr[cenJ])#((tr[cenI]-tr[cenJ])**2).sum()
    if DBGPRT:
        print([[dis(tr[i],tr[j]) for i in range(len(tr))] for j in range(len(tr))])
        print(meanDistance)
        print(minDistance)
        print(cenDistance)
        print(cenLI)
        print(cenLJ)
    if minDistance > min(cenLI,cenLJ)*1.2:
        return [patternI,patternJ],[[object_types[np.argmax(cl[i])] for i in patternI],[object_types[np.argmax(cl[i])] for i in patternJ]]
    else:
        return [[i for i in range(len(tr))]],[[object_types[np.argmax(cl[i])] for i in range(len(tr))]]

def newCluster(name):
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr = boxes["translations"]
    cl = boxes["class_labels"]
    patternId = [0 for _ in range(len(tr))]

    #np.savez_compressed(dir + name + ".npz", contour=contour)
    #If there are 2 clusters,
    I,J,L = 0,1,((tr[0]-tr[1])**2).sum()
    for i in range(len(tr)):
        for j in range(i):
            if ((tr[i]-tr[j])**2).sum() > L:
                I,J,L = i,j,((tr[i]-tr[j])**2).sum()

    patternI=[i for i in range(len(tr)) if ((tr[i]-tr[I])**2).sum() < ((tr[i]-tr[J])**2).sum()]
    patternJ=[i for i in range(len(tr)) if ((tr[i]-tr[I])**2).sum() > ((tr[i]-tr[J])**2).sum()]
    #return [patternI,patternJ]

    cenI=patternI[0]
    cenLI=1000

    for pi in range(len(patternI)):
        L = min([((tr[patternI[pi]]-tr[patternI[pj]])**2).sum() for pj in range(len(patternI)) if not(pj==pi)])
        if L < cenLI:
            cenLI = L
            cenI = patternI[pi]
    
    cenJ=patternJ[0]
    cenLJ=1000
    for pi in range(len(patternJ)):
        L = min([((tr[patternJ[pi]]-tr[patternJ[pj]])**2).sum() for pj in range(len(patternJ)) if not(pj==pi)])
        if L < cenLJ:
            cenLJ = L
            cenJ = patternJ[pi]

    patternII=[i for i in range(len(tr)) if ((tr[i]-tr[cenI])**2).sum() < ((tr[i]-tr[cenJ])**2).sum()]
    patternJJ=[i for i in range(len(tr)) if ((tr[i]-tr[cenI])**2).sum() > ((tr[i]-tr[cenJ])**2).sum()]

    return [patternII,patternJJ],[[object_types[np.argmax(cl[i])] for i in patternII],[object_types[np.argmax(cl[i])] for i in patternJJ]]
    
def higherCluster(names):
    for name in names:
        _,a=singleCluster(name)
        if DBGPRT or 1:
            print(a)
            #print(_)
        for k in _:
            patternStorage({"idList":k,"boxes":np.load(dir + name + "/boxes.npz", allow_pickle=True)})


    pass

def storePATTERNS(dst):
    f=open(dst,"w")
    f.write(json.dumps(PATTERNS))
    f.close()

testNames=[
    "2b49ac86-df1a-4cf4-b9d7-44b588d1d594_LivingDiningRoom-201",
    "ad0426db-a9e3-4428-989f-c2c38e1157c9_LivingDiningRoom-326",
    "35970ff6-4588-4bcf-8461-16448f1d0650_LivingDiningRoom-233",
    "01ba1742-4fa5-4d1e-8ba4-2f807fe6b283_LivingDiningRoom-4271",
    "d9df2354-d80d-49e7-a9e8-2e8fbf31dea3_LivingDiningRoom-998",
    "3f5df315-6fc7-46c2-84eb-862e27012530_LivingDiningRoom-174",
    "a3b91a1e-f51e-4714-a74b-33940b352efe_LivingDiningRoom-1900",
    "41a6ff52-5596-4560-97c4-14f827d12914_LivingDiningRoom-11510",
    "79b8f927-e1c8-4fc6-b0c2-6b9e2ff6708c_LivingDiningRoom-74474",
    "23a5fa77-0aa5-45f4-8399-3265005b1def_LivingDiningRoom-106046",
]
#testNames=os.listdir(dir)[:10]

higherCluster(testNames)
storePATTERNS("./patterns.json")
