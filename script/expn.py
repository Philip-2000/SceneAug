def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Pattern Manager\'s Application')
    showList = ["2b3662b2-7d0c-40eb-aa31-640ed8c7e7d4_ElderlyRoom-908","0bae68bc-b465-4f11-9d52-ebf5b44b0100_LivingDiningRoom-18864",
                "1dcb6909-4e3c-4e34-bb38-207cc78e9263_Bedroom-20397","01ba1742-4fa5-4d1e-8ba4-2f807fe6b283_LivingDiningRoom-4271"]
    parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  default="task")
    parser.add_argument("-e","--expr",  default="rcg",choices=["rcg","opt","gen"])
    parser.add_argument("-u","--usage", default="run",choices=["run","shw","vis"])
    parser.add_argument("-a","--dataset",default="../novel3DFront/")
    parser.add_argument("-c","--cnt",   default=8,type=int)
    parser.add_argument("-l","--list",  default=showList) #list is prior than cnt
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    import os
    args,E = parse(),None
    from SceneClasses.Experiment import RecExpn#,OptExpn,GenExpn
    if args.expr == "rcg":
        E = RecExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500] if args.usage!="shw" else args.list, args.task, args.cnt)
    # elif args.expr == "opt":
    #     E = OptExpn(args.version,args.dataset if args.usage!="vis" else None,os.listdir(args.dataset)[:500] if args.usage!="shw" else args.list, args.task, args.cnt)
    # elif args.expr == "gen":
    #     E = GenExpn(args.version)

    if args.usage == "run":
        E.run()
    elif args.usage == "vis":
        E.visualize()
    elif args.usage == "shw":
        E.show()
