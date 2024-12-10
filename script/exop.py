def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Optimizations\'s Application')
    #parser.add_argument("-v","--version",required=True)#,default="merg")
    #parser.add_argument("-t","--task",  default="task")
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    import os
    from SceneClasses.Experiment.ExOp import exops
    args = parse()
    E = exops()
    E()

    #我的意见是，直接把exops实例化出来，把它的第一个exop实例化出来，跑起来行吗，跑动

    #


    

    # if args.usage == "run":
    #     E.run()
    # elif args.usage == "vis":
    #     E.visualize()
    # elif args.usage == "shw":
    #     E.show()
