def parse(): #using pattern to understand scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Scene Optimizing Process')
    parser.add_argument('-d','--dataset',type=str,           default="../novel3DFront/")
    parser.add_argument("-v","--version",type=str,           default="losy")
    
    parser.add_argument('-t','--pattern',action="store_true",default=True)
    parser.add_argument('-p','--physics',action="store_true",default=True)

    sceneLst = [
        "5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487",
        "1a5bd12f-4877-405c-bb58-9c6bfcc0fb62_LivingRoom-53927",
        "1befc228-9a81-4936-b6a1-7e1b67cee2d7_Bedroom-352",
        "0de89e0a-723c-4297-8d99-3f9c2781ff3b_LivingDiningRoom-18932",
        "34f5f040-eb63-482b-82cb-9a3914c92c79_LivingDiningRoom-8678",
        "328ada87-9de8-4283-879d-58bffe5eb37a_Bedroom-5280",
        "39629e24-b405-420b-8fb0-72cef0238f70_SecondBedroom-1255",
        "4efedd5d-31d9-46c2-8c26-94ebdd7c0187_MasterBedroom-39695",
    ]

    from SceneClasses.Operation.Optm import default_optm_config as config_template
    #edit the config_template
    config_template["phy"]["vis"], config_template["pat"]["vis"] = {"res":{"res":(.5,.5,.5),},}, {}#{"pat":True}
    config_template["adjs"] = {"decay":200.0,'inertia':0.0}
    config_template["pat"]["prerec"] = False
    return parser.parse_args(sys.argv[1:]), config_template, sceneLst

if __name__ == "__main__": #load="testings",
    args, config, sceneLst = parse()
    from SceneClasses.Basic.Scne import scneDs as SDS
    SDS(args.dataset, lst=sceneLst,
        grp=False, cen=False, wl=True, windoor=True,
        #imgDir="./",os.path.join("./pattern/","optm",args.version),
    ).optimize(
        args.version,
        args.pattern, args.physics,
        -1,#config["phy"]["steps"],
        config, 1.5
    )
    #O = optm(args.version,S[0],PatFlag=args.pattern,PhyFlag=args.physics,rand=True)
    
    #O(10)
    # if args.usage == "rcgs":
    #     S.recognize(T,show=args.show)
    # elif args.usage == "evas":
    #     S.recognize(T,draw=False)
    #在optm.json中配置基础选项，包括可视化内容，优化超参数等等。
    