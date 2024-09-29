# Everything Operating Process about the Pattern
#   - Construction
#   - Recognization
#   - Modification
#                       Unconditional   Room-Conditional    Text-Conditional
#   - Generation        
#   - Completion
#   - Rearrangment

# In this file, we only present some examples of calling these methods in SceneClasses.Patn
# The operation in this file have no real meanings.
# All the operations in Designed Experiments for results is not here.


from SceneClasses.Patn import *

##################
def parse(argv):
    import argparse
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-v','--verbose', default=1)
    parser.add_argument('-d','--maxDepth', default=10)
    parser.add_argument('-n','--name', default="")#
    parser.add_argument('-l','--load', default="merg")#deat
    parser.add_argument('-s','--scaled', default=True, action="store_true")
    parser.add_argument('-w','--wid', default="rand2")
    parser.add_argument('-o','--oid', default="")
    parser.add_argument('-u','--uid', default="1")
    parser.add_argument('-g','--gen', default="")
    return parser.parse_args(argv)

if __name__ == "__main__": #load="testings",
    import sys
    args=parse(sys.argv[1:])
    #assert (len(args.name)>0 or len(args.load)>0) and (len(args.gen)>0 or len(args.uid)>0 or len(args.oid)>0)
    T = patternManager(verb=int(args.verbose),maxDepth=int(args.maxDepth),s=args.scaled,loadDataset=(len(args.load)==0))
    T.treeConstruction(load=args.load,name=args.name,draw=len(args.name)>0 or (len(args.uid)==0 and len(args.gen)==0))#
    
    # DIR = "./newRoom/"
    # W = walls.fromLog(f=DIR+args.wid+".txt",name=args.wid+"_") #wlz.draw(DIR)
    # #print(W)
    # #raise NotImplementedError
    # S = scne.empty(args.wid+"_")
    # S.registerWalls(W)
    # T.generate(nm="testing",theScene=S,useWalls=True,debug=True)

    if len(args.name)>0:#only for constructing
        sys.exit(0)

    if len(args.gen)>0:
        [T.generate(args.gen+str(i)) for i in range(16)]
    elif len(args.uid)>0:#
        UIDS = os.listdir(T.sceneDir)[:300] #["81c47424-f98c-418d-b810-ad23e586b3b2_LivingDiningRoom-876"]#
        scneDs(T.sceneDir,lst=UIDS,grp=False,cen=True,wl=False,keepEmptyWL=True,imgDir="./pattern/rcgs/").recognize(T) #[scne(fullLoadScene(uid),grp=False,cen=True,wl=True,imgDir="./pattern/rcgs/").tra(T) for uid in UIDS]
    elif len(args.oid)>0:
        UIDS = os.listdir(T.sceneDir)[:300]
        scneDs(T.sceneDir,lst=UIDS,grp=False,cen=True,wl=True,imgDir="./pattern/opts/").optimize(T) #[scne(fullLoadScene(uid),grp=False,cen=True,wl=True,imgDir="./pattern/opts/").opt(T) for uid in UIDS]