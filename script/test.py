from SceneClasses.Operation.Patn import patternManager as PM #this file is only for running or testing something. 
from SceneClasses.Basic.Scne import scne           #Only for developers. So we didn't set argument parser
from SceneClasses.Operation.Plan import plans 

cans = ["0b105b2a-e368-40ef-90a3-a4c422b915b4_LivingDiningRoom-69892024", "0bae68bc-b465-4f11-9d52-ebf5b44b0100_LivingDiningRoom-188642024", "0d530e8a-e0d5-4514-9e53-fd6c67893f43_LivingDiningRoom-80907",
"0b105b2a-e368-40ef-90a3-a4c422b915b4_MasterBedroom-3532", "0b812f6e-0769-4b37-b872-ee066433207e_MasterBedroom-11619"]


T = PM("losy")
res = [[0 for __ in cans] for _ in cans]
for i in range(len(cans)):
        for j in range(len(cans)):
                PLS = plans(scne.fromNpzs(name=cans[i]),T)
                fit,ass,_ = PLS.recognize(use=False,draw=False,show=False)
                res[i][j] = PLS.currentPlas.diff(scne.fromNpzs(name=cans[j]),fit)
                
import numpy as np
print(np.array(res))