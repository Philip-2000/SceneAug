def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Optimizations\'s Application')
    #parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  type=int, default=2)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    from SceneClasses.Experiment.ExOp import exops
    args = parse()
    E = exops(task=args.task)
    E()