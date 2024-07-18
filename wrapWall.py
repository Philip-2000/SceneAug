from classes.Scne import scne
from util import fullLoadScene

"""
def giveUps(ONE,TWO):
    #for i in 
    if len(ONE) > 0:
        oneTrMid = np.average(np.array([OBJES[i].translation for i in ONE]),axis=0)#print(oneTrMid)
        oneTrDev = np.average(np.array([dis(OBJES[i].translation,oneTrMid) for i in ONE]))
        sortedONE = sorted(ONE,key=lambda i:-dis(OBJES[i].translation,oneTrMid))
        sortedDis = [dis(OBJES[j].translation,oneTrMid) for j in sortedONE]
        i=0
        while i+1 < len(sortedONE):
            if (sortedDis[i]>max(3.0,2.0*sortedDis[i+1]) or sortedDis[i] > 3.0*oneTrDev) and (not OBJES[sortedONE[i]].class_name() in ["L-shaped Sofa","Three-seat / Multi-seat Sofa","Loveseat Sofa","TV Stand","Dining Chair"]):
                ONE.remove(sortedONE[i])
                i+=1
            else:
                break
        GRUPS.append(grup(ONE,len(GRUPS)))

    if len(TWO) > 0:
        twoTrMid = np.average(np.array([OBJES[i].translation for i in TWO]),axis=0)#print(twoTrMid)
        twoTrDev = np.average(np.array([dis(OBJES[i].translation,twoTrMid) for i in TWO]))
        sortedTWO = sorted(TWO,key=lambda i:-dis(OBJES[i].translation,twoTrMid))
        sortedDis = [dis(OBJES[j].translation,twoTrMid) for j in sortedTWO]
        i=0
        while i+1 < len(sortedTWO):
            if (sortedDis[i]>max(3.0,2.0*sortedDis[i+1]) or sortedDis[i] > 3.0*twoTrDev) and (not OBJES[sortedTWO[i]].class_name() in ["L-shaped Sofa","Three-seat / Multi-seat Sofa","Loveseat Sofa","TV Stand","Dining Chair"]):
                TWO.remove(sortedTWO[i])
                i+=1
            else:
                break
        GRUPS.append(grup(TWO,len(GRUPS)))#return [ONE,TWO]

    #delete other things in ONE and TWO

    pass

def formGroup(name):
    storeScene(name,False,True)
    ONE = []
    TWO = []
    if name.find("Living") >= 0:
        chairs=[ o.idx for o in OBJES if o.class_name()=='Dining Chair']
        lchair=[ o.idx for o in OBJES if o.class_name()=="Lounge Chair / Cafe Chair / Office Chair"]
        cchair=[ o.idx for o in OBJES if o.class_name()=="Classic Chinese Chair"]
        tables=[ o.idx for o in OBJES if o.class_name()=="Dining Table"]
        ctable=[ o.idx for o in OBJES if o.class_name()=="Coffee Table"]
        if (len(chairs)<2 and len(cchair)<2 and len(lchair)<2) or (len(tables) < 1 and len(ctable) < 1):
            ONE = [i for i in range(len(OBJES))]#print("no dining")#GRUPS.append(grup([i for i in range(len(OBJES))],len(GRUPS))) #return
            giveUps(ONE,TWO)
            return

        if len(chairs)>=2:
            pass
        elif len(lchair)>=2:
            chairs=lchair #print(name + " fuck")
        elif len(cchair)>=2:
            chairs=cchair


        #find the dinning_table/coffee_table surrouded by the chairs
        chairsTr = [OBJES[i].translation for i in chairs]
        chairsTrMid = np.average(np.array(chairsTr),axis=0)
        ONE = chairs 

        tables = [ o.idx for o in OBJES if (o.class_name() in ["Dining Table","Coffee Table"])]
        if len(tables)>0:
            T = tables[np.argmin(np.array([dis(OBJES[t].translation,chairsTrMid) for t in tables]))]
            LT = np.min(np.array([dis(OBJES[t].translation,chairsTrMid) for t in tables]))
            if LT < 0.6:
                ONE.append(T)

        lamps = [ o.idx for o in OBJES if (o.class_name() in ["Ceiling Lamp","Pendant Lamp"])]
        while len(lamps)>0:
            L = lamps[np.argmin(np.array([dis(OBJES[t].translation,chairsTrMid) for t in lamps]))]
            LL = np.min(np.array([dis(OBJES[t].translation,chairsTrMid) for t in lamps]))
            if LL < 0.6:
                ONE.append(L)
            else:
                break
            lamps.remove(L)

        TWO = [i for i in range(len(OBJES)) if not(i in ONE)]
        TWOTypes = [OBJES[i].class_name() for i in TWO]
        if not( ("L-shaped Sofa" in TWOTypes) or ("Three-seat / Multi-seat Sofa" in TWOTypes) or ("Loveseat Sofa" in TWOTypes) or ("TV Stand" in TWOTypes) ):
            ONE = [i for i in range(len(OBJES))]#GRUPS.append(grup([i for i in range(len(OBJES))],len(GRUPS)))#return
            giveUps(ONE,TWO)
            return

        twoTrMid = np.average(np.array([OBJES[i].translation for i in TWO]),axis=0)#print(twoTrMid)
        oneTrMid = np.average(np.array([OBJES[i].translation for i in ONE]),axis=0)#print(oneTrMid)
        
        toOne=[]
        for i in TWO:
            if dis(OBJES[i].translation,oneTrMid) < dis(OBJES[i].translation,twoTrMid):
                if not (OBJES[i].class_name() in ["L-shaped Sofa","Three-seat / Multi-seat Sofa","Loveseat Sofa","TV Stand"]):#(OBJES[i].translation[0]-oneTrMid[0])*(OBJES[i].translation[0]-twoTrMid[0])+(OBJES[i].translation[2]-oneTrMid[2])*(OBJES[i].translation[2]-twoTrMid[2]) > 0:
                    toOne.append(i)
        for i in toOne:
            TWO.remove(i)
            ONE.append(i)

        twoTrMid = np.average(np.array([OBJES[i].translation for i in TWO]),axis=0)#print(twoTrMid)
        oneTrMid = np.average(np.array([OBJES[i].translation for i in ONE]),axis=0)#print(oneTrMid)
        
        toTwo=[]
        for i in ONE:
            if dis(OBJES[i].translation,twoTrMid) < dis(OBJES[i].translation,oneTrMid):
                if not (OBJES[i].class_name() in ["Dining Table","Dining Chair"]):
                    toTwo.append(i)
        for i in toTwo:
            ONE.remove(i)
            TWO.append(i)

    else:#if name.find("Bedroom") >= 0:
        ONE = [i for i in range(len(OBJES))] #GRUPS.append(grup([i for i in range(len(OBJES))],len(GRUPS)))
    #give ups
    
    giveUps(ONE,TWO)
    return
    
np.savez_compressed("../novel3DFront_grp/"+n+"/group.npz", group=np.array([o.gid for o in OBJES],dtype=int))
    cnt = 0
    for n in os.listdir("../novel3DFront_img"):
        if (not n.endswith(".png")) or n.endswith("2024.png") or n.endswith("Mask.png") or (n.find("Living") == -1):
            continue
        formGroup(n[:-4])
        #adjustGroup()
        storeGroup(n[:-4])
        storedDraw(drawWall=True,objectGroup=True)
        plt.savefig("./segment/" + n[:-4] + ".png")
        plt.clf()
        #draftRoomMask(n)
        if cnt%1000==999:
            print(cnt)
        cnt += 1
        clearScene()
        #break
    pass
""" 
def test():
    for n in ["0d83ef53-4122-4678-93be-69f8b6d32c77_LivingDiningRoom-974.png"]:#os.listdir("./"):#[:20]:
        A = scne(fullLoadScene(n[:-4]),grp=True)#storeScene(n[:-4])
        A.adjustGroup()
        A.draw("./" + n[:-4] + "grp.png",drawWall=False,objectGroup=True,drawUngroups=False)#storedDraw()
        A.draftRoomMask("./" + n[:-4] + "mask.png")
#this graph can also be broken right?

#select a wall, a length? ||||||||| select a part of a wall, and check a length to move it?


#would door or window break by this movement? #would collision occurs? #would path still exist? 



#this tree is pre-calculated and stored? maybe.


#our augmentation based on trees?
#place the groups and wrap the walls?!!!!!!!!
#No idea anyway, fuck. 

if __name__ == "__main__":
    test()#main()

#what?

#for those not 