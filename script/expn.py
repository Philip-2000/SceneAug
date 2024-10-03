from SceneClasses.ExPn import RecExpn,OptExpn,GenExpn  #Execute the experiments
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Pattern Manager\'s Application')
    parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-e","--expr",  default="rcg",choices=["rcg","opt","gen"])
    parser.add_argument("-u","--usage", default="run",choices=["run","show"])
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument("-c","--cnt",   default=8,type=int)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    import os
    args,E = parse(),None
    if args.expr == "rcg":
        E = RecExpn(args.version,args.dataset,os.listdir(args.dataset)[:500])
    elif args.expr == "opt":
        E = OptExpn(args.version,args.dataset,os.listdir(args.dataset)[:1000])
    elif args.expr == "gen":
        E = GenExpn(args.version)

    if args.usage == "run":
        E.run()
    elif args.usage == "show":
        E.show(cnt=args.cnt)
