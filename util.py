import os
import numpy as np

dir = "../novel3DFront/"

def fullLoadScene(name):
    boxes = np.load(dir + name + "/boxes.npz", allow_pickle=True)
    tr, si, oi, cl, ce = boxes["translations"],boxes["sizes"],boxes["angles"],boxes["class_labels"],boxes["floor_plan_centroid"]
    walls = np.load(dir + name + "/contours.npz", allow_pickle=True)["contour"]
    widos = np.load(dir + name + "/conts.npz", allow_pickle=True)["cont"]
    grops = np.load(dir + name + "/group.npz", allow_pickle=True)["group"] if os.path.exists(dir + name + "/group.npz") else np.zeros(tr.shape[0])
    return {"translations":tr,"sizes":si,"angles":oi,"class_labels":cl,"floor_plan_centroid":ce,"walls":walls,"widos":widos,"grops":grops}
