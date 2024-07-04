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
    return (a[0]-b[0])**2+(a[2]-b[2])**2 if D==2 else ((a-b)**2).sum()

def func(name):

    #find the dinning_chairs
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr = boxes["translations"]
    cl = boxes["class_labels"]

    chairs=[ i for i in range(len(cl)) if object_types[np.argmax(cl[i])]=='dining_chair']
    cchair=[ i for i in range(len(cl)) if object_types[np.argmax(cl[i])]=="lounge_chair"]
    if len(chairs)>=2 and len(cchair)>=2:
        print("fuck")
    elif len(cchair)>=2:
        chairs=cchair
    elif len(chairs)<2 and len(cchair)<2:
        print("no dining")
        return [[i for i in range(len(cl))]]
    
    #find the dinning_table/coffee_table surrouded by the chairs
    chairsTr = [tr[i] for i in chairs]
    chairsTrMid = np.average(np.array(chairsTr),axis=0)
    ONE = chairs 

    tables = [ i for i in range(len(cl)) if (object_types[np.argmax(cl[i])] in ["dining_table","coffee_table"])]
    if len(tables)>0:
        T = tables[np.argmin(np.array([dis(tr[t],chairsTrMid) for t in tables]))]
        LT = np.min(np.array([dis(tr[t],chairsTrMid) for t in tables]))
        if LT < 0.6:
            ONE.append(T)


    lamps = [ i for i in range(len(cl)) if (object_types[np.argmax(cl[i])] in ["ceiling_lamp","pendant_lamp"])]
    if len(lamps)>0:
        L = lamps[np.argmin(np.array([dis(tr[t],chairsTrMid) for t in lamps]))]
        LL = np.min(np.array([dis(tr[t],chairsTrMid) for t in lamps]))
        if LL < 0.6:
            ONE.append(L)

    TWO = [i for i in range(len(tr)) if not(i in ONE)]
    TWOTypes = [object_types[np.argmax(cl[i])] for i in TWO]
    if not( ("l_shaped_sofa" in TWOTypes) or ("multi_seat_sofa" in TWOTypes) or ("loveseat_sofa" in TWOTypes) ):
        return [[i for i in range(len(cl))]]
        
    twoTrMid = np.average(np.array([tr[i] for i in TWO]),axis=0)
    oneTrMid = np.average(np.array([tr[i] for i in ONE]),axis=0)

    toOne=[]
    for i in TWO:
        if dis(tr[i],oneTrMid) < dis(tr[i],twoTrMid):
            if (tr[i][0]-oneTrMid[0])*(tr[i][0]-twoTrMid[0])+(tr[i][2]-oneTrMid[2])*(tr[i][2]-twoTrMid[2]) > 0:
                toOne.append(i)
    
    for i in toOne:
        TWO.remove(i)
        ONE.append(i)

    return [ONE,TWO]

def higherCluster(names):
    for name in names:
        _=func(name)
        boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
        cl = boxes["class_labels"]
        
        a = [[object_types[np.argmax(cl[i])] for i in __] for __ in _]
        print(a)

testNames=[
    "65cd43e1-1294-44f7-a560-7a230ff893d2_LivingDiningRoom-72973",
    "73818192-7ca3-4161-8cf1-ca3e5e51b1ab_LivingDiningRoom-62037",
    "07867590-fb15-4cb6-947b-118385d7a9da_LivingDiningRoom-300",
    "44903efe-45ea-42a0-954e-bacb917bd7dc_LivingDiningRoom-648",
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

higherCluster(testNames)