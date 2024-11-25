def parse(): #using pattern to understand scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Scene Optimizing Process')
    parser.add_argument('-d','--sceneID', default="losy",type=str)
    
    parser.add_argument('-s','--pattern', default=False, type=bool, action="store_true")
    parser.add_argument('-p','--physics', default=True,  type=bool, action="store_true")
    parser.add_argument('-i','--iRate',   default=0.01,  type=float)
    parser.add_argument('-j','--jRate',   default=0.01,  type=float)
    parser.add_argument('-n','--steps',   default=100,   type=int)


    parser.add_argument("-v","--version", default="losy",type=str)
    

    #在这里配置基础选项，包括可视化内容，优化超参数等等。
    config = {
        "pat":{
            
        },
        "phy":{
            "ss": [[1,0,1],[1,0,0],[1,0,-1],[0,0,-1],[-1,0,-1],[-1,0,0],[-1,0,1],[0,0,1]],
            "grid":{
                "L":8,
                "x":0.01,
            },
            "vis":{
                "res":{
                    "res":(0.2,0.2,0.2),
                },
                "syn":{
                    "t":(0,0.5,0.5),
                    "s":(0.5,0,0.5),
                    "r":(0.5,0.5,0),
                    "res":(0.2,0.2,0.2),
                },
                "pnt": {
                    "al":(0,0,0),
                },
                "pns":{
                    "wo":(1.0,0,0),
                    "wi":(0,0,1.0),
                    "dr":(0,1.0,0),
                    "ob":(0.33,0.33,0.33),
                },
                "filv":{
                    "wo":(1.0,0,0),
                    "wi":(0,0,1.0),
                    "dr":(0,1.0,0),
                    "ob":(0.33,0.33,0.33),
                },
                "filh":{
                    "wo":(1.0,0,0),
                    "wi":(0,0,1.0),
                    "dr":(0,1.0,0),
                    "ob":(0.33,0.33,0.33),
                },
                "pot":{

                },
            }
        }
    }
    return parser.parse_args(sys.argv[1:]), config

if __name__ == "__main__": #load="testings",
    args, config = parse()
    import os
    from SceneClasses.Operation.Patn import patternManager as PM 
    from SceneClasses.Basic.Scne import scneDs as SDS
    SDS(args.dataset,lst=[args.id],grp=False,cen=True,wl=True,keepEmptyWL=True,imgDir=os.path.join("./pattern/","optm",args.version)).optimize(PM(args.version,verb=args.verbose),args.pattern,args.physics,config["pat"]["steps"],config["phy"]["iRate"],config["pat"]["iRate"],config)
    #O = optm(args.version,S[0],PatFlag=args.pattern,PhyFlag=args.physics,rand=True)
    
    #O(10)
    # if args.usage == "rcgs":
    #     S.recognize(T,show=args.show)
    # elif args.usage == "evas":
    #     S.recognize(T,draw=False)
    #在optm.json中配置基础选项，包括可视化内容，优化超参数等等。
    