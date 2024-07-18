
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Link import *
from classes.Wall import *
from classes.Grup import *
from classes.Spce import *
from classes import *

dir = "../novel3DFront/"

def two23(a):
    return np.array([a[0],0,a[1]])

def fullLoadScene(name):
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr, si, oi, cl, ce = boxes["translations"],boxes["sizes"],boxes["angles"],boxes["class_labels"],boxes["floor_plan_centroid"]
    walls = np.load(dir + name + "/contours.npz", allow_pickle=True)["contour"]
    widos = np.load(dir + name + "/conts.npz", allow_pickle=True)["cont"]
    grops = np.load("../novel3DFront_grp/" + name + "/group.npz", allow_pickle=True)["group"]
    return {"tr":tr,"si":si,"oi":oi,"cl":cl,"ce":ce,"walls":walls,"widos":widos,"grops":grops}

def storeScene(name, windoor=True,wl=True,grp=False):
    scene = fullLoadScene(name)
    tr,si,oi,cl,ce,walls,widos = scene["tr"],scene["si"],scene["oi"],scene["cl"],scene["ce"],scene["walls"],scene["widos"]
    #firstly, store those objects and walls into the WALLS and OBJES
    if grp:
        grops=scene["grops"]

    for i in range(len(tr)):
        cli = np.concatenate([cl[i],[0,0]])
        OBJES.append(obje(tr[i]+ce,si[i],oi[i],cli,idx=len(OBJES),gid=-1 if (not grp) else grops[i]))

    if grp:
        A = {-1:[],0:[],1:[]}
        for i in range(len(grops)):
            A[grops[i]].append(i)
        GRUPS.append(grup(A[0],0))
        if len(A[1]):
            GRUPS.append(grup(A[1],1))
    
    for k in range(len(widos)):
        oii = np.math.atan2(widos[k][-1],widos[k][-2])
        sii = np.array([max(widos[k][3],widos[k][5]),widos[k][4],min(widos[k][3],widos[k][5])]) #?
        tri = widos[k][:3]
        c = len(object_types)-1 if tri[1]-sii[1] < 0.1 else len(object_types)-2
        if windoor:
            OBJES.append(obje(tri,sii,oii,i=c,idx=len(OBJES)))

    #obje(t,s,o,c,i)
    #wall(p,q,n,w1,w2)
    for j in range(len(walls)):
        w1 = (j-1)%len(walls)
        w2 = (j+1)%len(walls)
        if wl:
            WALLS.append(wall(two23(walls[j][:2]),two23(walls[w2][:2]),np.array([walls[j][3],0,walls[j][2]]),w1,w2,j))

    # for w in WALLS:
    #     print(w.p)
    #     print(w.n)

    # for o in OBJES:
    #     print(o.class_name())
    #     print(o.translation)
    return

def storedDraw(lim=-1,drawWall=True,objectGroup=False):
    for i in range(len(SPCES)):
        SPCES[i].draw()

    for i in range(len(GRUPS)):
        GRUPS[i].draw()

    for i in range(len(OBJES)):
        OBJES[i].draw(objectGroup)#corners = OBJES[i].corners2()
        #plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-OBJES[i].class_index>2 else "*")

    if drawWall:
        J = min([w.idx for w in WALLS if w.v])#WALLS[0].w2
        contour,w =[[WALLS[J].p[0],WALLS[J].p[2]]], WALLS[J].w2
        while w != 0:
            contour.append([WALLS[w].p[0],WALLS[w].p[2]])
            w = WALLS[w].w2
        contour = np.array(contour)
        plt.plot(np.concatenate([contour[:,0],contour[:1,0]]),np.concatenate([-contour[:,1],-contour[:1,1]]), marker="o")
    plt.axis('equal')
    if lim > 0:
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
    else:
        plt.axis('off')
    pass

from PIL import Image
def draftRoomMask(n):
    #scale the image space to the real world
    #with a center and a scale
    #
    sz=64
    ce=np.array([0.,0.,0.])
    rt=8
    # if len(GRUPS)==1:
    #     ce = GRUPS[0].translation + np.array([np.random.rand()*2.-1.,0.0,np.random.rand()*2.-1.])
    # else:
    #     ce =(GRUPS[0].translation + GRUPS[1].translation)/2.0
    

    N = np.zeros((sz*2,sz*2))
    if len(GRUPS)==1:
        con = ce - GRUPS[0].translation
        lineMin,lineMax = 0.4,0.8
    else:
        con = GRUPS[1].translation - GRUPS[0].translation#print(con)
        lineMin,lineMax = 1.0,2.0
    con_ = con/np.linalg.norm(con)#print(con_)
    nom = np.cross(con_,np.array([0,1,0]))#print(nom)
    cons = con_/np.linalg.norm(con)
    areaMin,areaMax = 0.8,1.2
    for i in range(sz*2):
        for j in range(sz*2):
            t = np.array([(i-sz)/rt,0.0,(j-sz)/rt])+ce
            #check distance toward 
            #print(t)
            norm0t = np.clip( np.max(np.abs((t - GRUPS[0].translation)/GRUPS[0].size)),areaMin,areaMax)
            c0 = np.math.floor(255*(areaMax-norm0t)/(areaMax-areaMin))
            N[i,j]=c0
            if len(GRUPS)>1:
                norm1t = np.clip( np.max(np.abs((t - GRUPS[1].translation)/GRUPS[1].size)),areaMin,areaMax)
                c1 = np.math.floor(255*(areaMax-norm1t)/(areaMax-areaMin))
                c0 = max(c0,c1)
                N[i,j]=c0
            
            
            rat = (t-GRUPS[0].translation)@cons
            dis = np.abs((t-GRUPS[0].translation)@nom)
            #GRUPS[0].translation -> GRUPS[1].translation, GRUPS[0]
            if rat < 0.0 or rat > 1.0:
                continue
            clipdis = np.clip(dis,lineMin,lineMax)
            c2 = np.math.floor(255*(lineMax-clipdis)/(lineMax-lineMin))
            c0 = max(c0,c2)
            N[i,j]=c0

    img = Image.new("RGB", (sz*2,sz*2), "#000000")  
    pixels = img.load()
    K = 2
    for i in range(sz*2):
        for j in range(sz*2):
            v = 0
            for k in range(max(i-K,0),min(i+K+1,sz*2)):
                for l in range(max(j-K,0),min(j+K+1,sz*2)):
                    v += N[k,l]
            v = int(v/25)
            pixels[i,j] = (v,v,v,255)
    img.save("./"+n[:-4]+"Mask.png")
    pass

def clearScene():
    global WALLS
    global OBJES
    global LINKS
    global GRUPS
    global SPCES
    WALLS.clear()
    OBJES.clear()
    LINKS.clear()
    GRUPS.clear()
    SPCES.clear()
