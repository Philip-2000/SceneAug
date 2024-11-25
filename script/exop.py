def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Pattern Manager\'s Application')
    parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  default="task")
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    import os
    args,E = parse(),None
    from SceneClasses.Experiment.ExPn import RecExpn,OptExpn,GenExpn
    if args.expr == "rcg":
        E = RecExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500] if args.usage!="shw" else args.list, args.task, args.cnt)
    elif args.expr == "opt":
        E = OptExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500] if args.usage!="shw" else args.list, args.task, args.cnt)
    elif args.expr == "gen":
        E = GenExpn(args.version)

    if args.usage == "run":
        E.run()
    elif args.usage == "vis":
        E.visualize()
    elif args.usage == "shw":
        E.show()
