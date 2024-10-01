from SceneClasses.Patn import patternManager
from SceneClasses.Scne import scneDs
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument("-v","--version",   default="broe")
    parser.add_argument('-e','--verbose',   default=1, type=int)
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-s','--scaled',    default=True, action="store_true")
    parser.add_argument('-d','--maxDepth',  default=-1, type=int)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    args = parse()
    T,S = patternManager(args.version,verb=args.verbose,new = True), scneDs(args.dataset, grp=False,wl=False,keepEmptyWL=True,cen=True,rmm=False)
    T.construct(args.maxDepth,args.scaled,S)