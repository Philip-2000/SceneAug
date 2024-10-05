from SceneClasses.Scne import scneDs #using pattern to synthesis scene

def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Pattern Manager Application')
    parser.add_argument("-v","--version", required=True)#,   default="brot"
    parser.add_argument('-n','--name',      default="test")
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-q","--n_sequence",default=16, type=int)
    parser.add_argument("-d","--dataset",   default="",         choices=["","../novel3DFront/"])
    parser.add_argument("-a","--application",default="gnrt",    choices=["gnrt","copl","rarg","agmt"])
    parser.add_argument("-c","--condition", default="uncond",   choices=["uncond","textcond","roomcond"])
    parser.add_argument('-i','--id',        default="")#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": 
    args = parse()
    assert (args.application != "agmt") or len(args.id) > 0
    scneDs(name=args.dataset,lst=([args.id]*args.n_sequence if len(args.id) else []),prepare=args.condition,num=args.n_sequence).synthesis(args.application,args.condition,args.version)