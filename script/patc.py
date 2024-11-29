def parse(): #construting pattern
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Pattern Manager Construction')
    parser.add_argument("-v","--version", required=True)#,   default="brot"
    parser.add_argument('-e','--verbose',   default=1, type=int)
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-s','--scaled',    default=True, action="store_true")
    parser.add_argument('-d','--maxDepth',  default=-1, type=int)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    from SceneClasses.Operation.Patn import patternManager as PM
    PM("losy").draw()
else:
    args = parse()
    import os
    from SceneClasses.Operation.Patn import patternManager as PM
    from SceneClasses.Basic.Scne import scneDs as SDS
    T,S = PM(args.version,verb=args.verbose,new = True), SDS(args.dataset,lst=os.listdir(args.dataset),grp=False,wl=False,keepEmptyWL=True,cen=True,rmm=False)
    T.construct(args.maxDepth,args.scaled,S)