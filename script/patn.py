from SceneClasses.Patn import patternManager
from SceneClasses.Scne import scneDs
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument("-v","--version", required=True)#,   default="brot"
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-u","--usage",     default="rcgs", choices=["rcgs","evas","opts"])
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-i','--id',        default="")#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    args = parse()
    import os
    UIDS = os.listdir(args.dataset)[:300] if len(args.id)==0 else [args.id]
    T = patternManager(args.version,verb=args.verbose)
    S = scneDs(args.dataset,lst=UIDS,grp=False,cen=True,wl=False,keepEmptyWL=True,imgDir=os.path.join(T.workDir,args.usage,T.version))
    if args.usage == "rcgs":
        S.recognize(T)
    elif args.usage == "evas":
        S.recognize(T,draw=False)
    elif args.usage == "opts":
        raise NotImplementedError
        S.optimize(T)
    #synthesis is in spce.py; constructing is in patc.py

# In this file, we only present some examples of calling recognize/evaluate/optimize in patternManager
# The operation in this file have no real meanings.
# All the operations in Designed Experiments for results is not here.