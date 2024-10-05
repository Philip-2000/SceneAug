from SceneClasses.ExSy import *  #Execute the pattern synthesis the experiments
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Pattern Manager\'s Application')
    parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  default="task")
    parser.add_argument("-e","--expr",  default="rcg",choices=["rcg","opt","gen"])
    parser.add_argument("-u","--usage", default="run",choices=["run","show","vis"])
    parser.add_argument("-a","--dataset",default="../novel3DFront/")
    parser.add_argument("-c","--cnt",   default=8,type=int)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    import os
    args,E = parse(),None
    if args.expr == "rcg":
        pass#E = RecExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500],args.task)
    elif args.expr == "opt":
        pass#E = OptExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500],args.task)
    elif args.expr == "gen":
        pass#E = GenExpn(args.version)

    if args.usage == "run":
        pass#E.run()
    elif args.usage == "vis":
        pass#E.visualize()
    elif args.usage == "show":
        pass#E.show(cnt=args.cnt)
