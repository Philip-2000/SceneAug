def parse(): #using pattern to understand scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Scene Optimizing Process')
    parser.add_argument('-d','--dataset',type=str,           default="../novel3DFront/")
    parser.add_argument('-s','--sceneID',type=str,           default="0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487")
    parser.add_argument("-v","--version",type=str,           default="losy")
    
    parser.add_argument('-t','--pattern',action="store_true",default=False)
    parser.add_argument('-p','--physics',action="store_true",default=True)
    parser.add_argument('-i','--iRate',  type=float,         default=0.01)
    parser.add_argument('-j','--jRate',  type=float,         default=0.01)
    parser.add_argument('-n','--steps',  type=int,           default=100)


    #在这里配置基础选项，包括可视化内容，优化超参数等等。
    config = {
        "pat":{
            "steps":1,
            "rate":0.5,
        },
        "phy":{
            "steps":1,
            "rate":0.5,
            "s4": 2,#[[1,0,1],[1,0,0],[1,0,-1],[0,0,-1],[-1,0,-1],[-1,0,0],[-1,0,1],[0,0,1]],
            "door":{"expand":1.1,"out":0.2,"in":1.0,},
            "wall":{"bound":0.5,},
            "object":{
                "Pendant Lamp":[.0,.01,.01],#
                "Ceiling Lamp":[.0,.01,.01],#
                "Bookcase / jewelry Armoire":[.2,1., .9],#
                "Round End Table":[.0,.5, .5],#
                "Dining Table":[.0,.5, .5],#
                "Sideboard / Side Cabinet / Console table":[.0,.9, .9],#
                "Corner/Side Table":[.0,.9, .9],#
                "Desk":[.0,.9, .9],#
                "Coffee Table":[.0,1.,1.1],#
                "Dressing Table":[.0,.9, .9],#
                "Children Cabinet":[.2,1., .9],#
                "Drawer Chest / Corner cabinet":[.2,1., .9],#
                "Shelf":[.2,1., .9],#
                "Wine Cabinet":[.2,1., .9],#
                "Lounge Chair / Cafe Chair / Office Chair":[.0,.5, .5],#
                "Classic Chinese Chair":[.0,.5, .5],#
                "Dressing Chair":[.0,.5, .5],#
                "Dining Chair":[.0,.5, .5],#
                "armchair":[.0,.5, .5],#
                "Barstool":[.0,.5, .5],#
                "Footstool / Sofastool / Bed End Stool / Stool":[.0,.5, .5],#
                "Three-seat / Multi-seat Sofa":[.2,1., .9],#
                "Loveseat Sofa":[.2,1., .9],#
                "L-shaped Sofa":[.0,.6, .9],#
                "Lazy Sofa":[.2,1., .9],#
                "Chaise Longue Sofa":[.2,1., .9],#
                "Wardrobe":[.2,1., .9],#
                "TV Stand":[.2,1., .9],#
                "Nightstand":[.0,.5, .5],#
                "King-size Bed":[.2,1.,1.2],#
                "Kids Bed":[.2,1.,1.2],#
                "Bunk Bed":[.2,1.,1.2],#
                "Single bed":[.2,1.,1.2],#
                "Bed Frame":[.2,1.,1.2],#
            },
            "syn":{"T":1.0,"S":0.01,"R":1.0,},
            "grid":{"L":5.5,"d":0.1,"b":10,},
            "vis":{
                "res":{"res":(.5,.5,.5),},
                "syn":{"t":(.0,.5,.5),"s":(.5,.0,.5),"r":(.5,.5,.0),"res":(.5,.5,.5),},
                "pnt":{"al":(.0,.0,.0),},
                "pns":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
                "fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
                "fih":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
                #"fip":{"res":(0.33,0.33,0.33)},
                "fiq":{"wo":(1.0,0,0),"wi":(0,0,1.0),"dr":(.33,.33,.33),"ob":(0,1.0,0),},
            }
        }
    }
    return parser.parse_args(sys.argv[1:]), config

if __name__ == "__main__": #load="testings",
    args, config = parse()
    import os
    #from SceneClasses.Operation.Patn import patternManager as PM 
    from SceneClasses.Basic.Scne import scneDs as SDS
    SDS(args.dataset, lst=[args.sceneID],
        grp=False, cen=False, wl=True, windoor=True,
        imgDir="./",#os.path.join("./pattern/","optm",args.version),
    ).optimize(
        None,#PM(args.version),
        args.pattern, args.physics,
        config["phy"]["steps"], config["phy"]["rate"], config["pat"]["rate"],
        config
    )
    #O = optm(args.version,S[0],PatFlag=args.pattern,PhyFlag=args.physics,rand=True)
    
    #O(10)
    # if args.usage == "rcgs":
    #     S.recognize(T,show=args.show)
    # elif args.usage == "evas":
    #     S.recognize(T,draw=False)
    #在optm.json中配置基础选项，包括可视化内容，优化超参数等等。
    