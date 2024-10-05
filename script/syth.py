from SceneClasses.Syth import * #using pattern to synthesis scene

def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Pattern Manager Application')
    parser.add_argument("-v","--version", required=True)#,   default="brot"
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-a","--application",default="gnrt",  choices=["gnrt","copl","rarg","agmt"])
    parser.add_argument("-c","--condition", default="uncond", choices=["uncond","textcond","roomcond"])
    parser.add_argument("-d","--dataset",   default="../novel3DFront/")
    parser.add_argument('-i','--id',        default="")#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    #import os
    #UIDS = os.listdir(args.dataset)[:1000] if len(args.id)==0 else [args.id]
    #T = patternManager(args.version,verb=args.verbose)
    #S = scneDs(args.dataset,lst=UIDS,grp=False,cen=True,wl=False,keepEmptyWL=True,imgDir=os.path.join(T.workDir,args.usage,T.version))

    args,S = parse(),None
    if args.application == "gnrt":
        S = gnrt(args.version,args.task)
    elif args.application == "copl":
        S = copl(args.version,args.task)
    elif args.application == "rarg":
        S = rarg(args.version)
    elif args.application == "agmt":
        S = agmt(args.version)

    if args.application == "agmt":
        S.augment()
    elif args.condition == "uncond":
        S.uncond()
    elif args.condition == "textcond":
        S.textcond()
    elif args.condition == "roomcond":
        S.roomcond(cnt=args.cnt)
