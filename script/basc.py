def parse(): #using pattern to understand scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Pattern Manager Application')
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-u","--usage",     default="draw", choices=["draw"])
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-n','--num',       default=20, type=int)#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024
    parser.add_argument('-i','--id',        default="")#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    args = parse()
    import os
    UIDS = os.listdir(args.dataset)[:args.num] if len(args.id)==0 else [args.id]
    from SceneClasses.Basic.Scne import scneDs as SDS
    S = SDS(args.dataset,lst=UIDS,grp=False,cen=True,wl=True,windoor=True)
    S.draw(d=True)