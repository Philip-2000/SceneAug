import os
import numpy as np
def fullLoadScene(name,dir="../novel3DFront/"):
    boxes,dn = np.load(dir+name+"/boxes.npz"),dir+name
    return {"room_layout":np.zeros((1,64,64)).astype(np.uint8),"translations":boxes["translations"],"sizes":boxes["sizes"],"angles":boxes["angles"],"class_labels":boxes["class_labels"],"floor_plan_centroid":boxes["floor_plan_centroid"],"scene_uid":boxes["scene_uid"],"walls":np.load(dn+"/contours.npz")["contour"],"widos":np.load(dn+"/conts.npz")["cont"],"grops":np.load(dn+"/group.npz")["group"] if os.path.exists(dn+"/group.npz") else None}
