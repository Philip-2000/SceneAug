from matplotlib import pyplot as plt
import sys,numpy as np
def f1(s):
        return 0.2+(0.9-0.2)*np.exp(-0.4*s)
def f2(s):
        return 0.001 + (1.0-0.001)*(1-np.exp(-1.5*s)) 
s = np.array([i for i in range(36)])
plt.plot(s,f1(s))
plt.plot(s,f2(s)+f1(s))
plt.show()
sys.exit(0)

from SceneClasses.Operation import patternManager as PM #this file is only for running or testing something. 
from SceneClasses.Basic import scne           #Only for developers. So we didn't set argument parser
from SceneClasses.Operation import rgnz 

cans = ["0b105b2a-e368-40ef-90a3-a4c422b915b4_LivingDiningRoom-69892024", "0bae68bc-b465-4f11-9d52-ebf5b44b0100_LivingDiningRoom-188642024", "0d530e8a-e0d5-4514-9e53-fd6c67893f43_LivingDiningRoom-80907",
"0b105b2a-e368-40ef-90a3-a4c422b915b4_MasterBedroom-3532", "0b812f6e-0769-4b37-b872-ee066433207e_MasterBedroom-11619"]


T = PM("losy")
res = [[0 for __ in cans] for _ in cans]
for i in range(len(cans)):
        for j in range(len(cans)):
                PLS = rgnz(scne.fromNpzs(name=cans[i]),T)
                fit,ass,_ = PLS.recognize(use=False,draw=False,show=False)
                res[i][j] = PLS.currentPlan.diff(scne.fromNpzs(name=cans[j]),fit)
                
import numpy as np
print(np.array(res))