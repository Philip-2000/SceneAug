
from matplotlib import pyplot as plt

from classes.Obje import *
from classes.Link import *
from classes.Wall import *
from classes.Grup import *
from classes import *

dir = "../novel3DFront/"

def two23(a):
    return np.array([a[0],0,a[1]])

def fullLoadScene(name):
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr, si, oi, cl, ce = boxes["translations"],boxes["sizes"],boxes["angles"],boxes["class_labels"],boxes["floor_plan_centroid"]
    walls = np.load(dir + name + "/contours.npz", allow_pickle=True)["contour"]
    widos = np.load(dir + name + "/conts.npz", allow_pickle=True)["cont"]
    return {"tr":tr,"si":si,"oi":oi,"cl":cl,"ce":ce,"walls":walls,"widos":widos}

def storeScene(name, windoor=True,wl=True):
    scene = fullLoadScene(name)
    tr,si,oi,cl,ce,walls,widos = scene["tr"],scene["si"],scene["oi"],scene["cl"],scene["ce"],scene["walls"],scene["widos"]
    #firstly, store those objects and walls into the WALLS and OBJES
    
    for i in range(len(tr)):
        cli = np.concatenate([cl[i],[0,0]])
        OBJES.append(obje(tr[i]+ce,si[i],oi[i],cli,idx=len(OBJES)))
    
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

def storedDraw(lim=-1):
    for i in range(len(OBJES)):
        corners = OBJES[i].corners2()
        plt.plot( np.concatenate([corners[:,0],corners[:1,0]]), np.concatenate([-corners[:,1],-corners[:1,1]]), marker="." if len(object_types)-OBJES[i].class_index>2 else "*")

    contour,w =[[WALLS[0].p[0],WALLS[0].p[2]]], WALLS[0].w2
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

def clearScene():
    global WALLS
    global OBJES
    global LINKS
    global GRUPS
    WALLS.clear()
    OBJES.clear()
    LINKS.clear()
    GRUPS.clear()
