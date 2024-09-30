from SceneClasses.ExPn import RecExpn,OptExpn,GenExpn  #Execute the experiments
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument("-v","--version",default="merg")
    parser.add_argument("-e","--expr",  default="rec",choices=["rec","opt","gen"])
    parser.add_argument("-u","--usage", default="run",choices=["run","show"])
    parser.add_argument("-c","--cnt",   default=8,type=int)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    args,E = parse(),None
    if args.expr == "rec":
        E = RecExpn(args.version)
    elif args.expr == "opt":
        E = OptExpn(args.version)
    elif args.expr == "gen":
        E = GenExpn(args.version)

    if args.usage == "run":
        E.run()
    elif args.usage == "show":
        E.show(cnt=args.cnt)
