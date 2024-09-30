from SceneClasses.Patn import patternManager
from SceneClasses.Scne import scneDs
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument("-v","--version",   default="broe")
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-u","--usage",     default="rec", choices=["rec","eva","opt"])
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-i','--id',        default="")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    args = parse()
    import os
    UIDS = os.listdir(args.dataset)[:300] #["81c47424-f98c-418d-b810-ad23e586b3b2_LivingDiningRoom-876"]#
    S,T = scneDs(args.dataset,lst=UIDS,grp=False,cen=True,wl=False,keepEmptyWL=True,imgDir="./pattern/rcgs/"),patternManager(args.version,verb=args.verbose)

    if args.usage == "rec":
        S.recognize(T)
    elif args.usage == "eva":
        S.recognize(T,draw=False)
    elif args.usage == "opt":
        raise NotImplementedError
        S.optimize(T)
    #synthesis is in spce.py; constructing is in patc.py

# In this file, we only present some examples of calling recognize/evaluate/optimize in patternManager
# The operation in this file have no real meanings.
# All the operations in Designed Experiments for results is not here.