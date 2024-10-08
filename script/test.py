from SceneClasses.Patn import patternManager #this file is only for running or testing something. 
from SceneClasses.Scne import scne           #Only for developers. So we didn't set argument parser
from SceneClasses.Plan import plans 

cans = ["0b105b2a-e368-40ef-90a3-a4c422b915b4_LivingDiningRoom-69892024",
        "0bae68bc-b465-4f11-9d52-ebf5b44b0100_LivingDiningRoom-188642024",
        "0d530e8a-e0d5-4514-9e53-fd6c67893f43_LivingDiningRoom-80907",
        "0b105b2a-e368-40ef-90a3-a4c422b915b4_MasterBedroom-3532",
        "0b812f6e-0769-4b37-b872-ee066433207e_MasterBedroom-11619",
        ]


T = patternManager("losy")
scene0 = scne.fromNpzs(name=cans[3])
scene1 = scne.fromNpzs(name=cans[4])


PLS = plans(scene0,T)
fit,ass,_ = PLS.recognize(use=False,opt=False,draw=False,show=False)
dif = PLS.currentPlas.diff(scene1,fit)

print(dif)


PLZ = plans(scene1,T)
fit,ass,_ = PLZ.recognize(use=False,opt=False,draw=False,show=False)
dif = PLZ.currentPlas.diff(scene0,fit)

print(dif)

