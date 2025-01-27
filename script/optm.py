def parse(): #using pattern to understand scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Scene Optimizing Process')
    parser.add_argument('-d','--dataset',type=str,           default="../novel3DFront/")
    parser.add_argument("-v","--version",type=str,           default="losy")
    parser.add_argument("-r","--rand",   type=float,         default=1.0)
    
    parser.add_argument('-t','--pattern',action="store_true",default=True)
    parser.add_argument('-p','--physics',action="store_true",default=True)

    sceneLst = [
        #"ef0613b7-4461-461b-96c6-cb140f7a2f2a_LivingDiningRoom-16342",
        #"1d43e076-fc80-4d55-a07d-1b7a8c6dc6e7_LivingDiningRoom-3814",
        #"4a102195-8b2f-444d-b95d-3b332822bc9a_LivingDiningRoom-152",
        #"5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088",
        #"0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        #"0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487",
        #"1a5bd12f-4877-405c-bb58-9c6bfcc0fb62_LivingRoom-53927",
        #"1befc228-9a81-4936-b6a1-7e1b67cee2d7_Bedroom-352",
        #"0de89e0a-723c-4297-8d99-3f9c2781ff3b_LivingDiningRoom-18932",
        #"34f5f040-eb63-482b-82cb-9a3914c92c79_LivingDiningRoom-8678",
        #"328ada87-9de8-4283-879d-58bffe5eb37a_Bedroom-5280",
        #"39629e24-b405-420b-8fb0-72cef0238f70_SecondBedroom-1255",
        #"4efedd5d-31d9-46c2-8c26-94ebdd7c0187_MasterBedroom-39695",
        "0abea0c8-3398-4d26-b03d-9fb1fc9708a4_LivingDiningRoom-212696",
    ]

    from SceneClasses.Operation.Optm import default_optm_config as config_template
    #edit the config_template #"res":{"res":(.5,.5,.5),},  
    config_template["phy"]["vis"], config_template["pat"]["vis"] = {"fiv":{"wo":(1.0,0,0),"wi":(0,0,1.0), "dr":(0,1.0,0),}, "ob":(0.33,0.33,0.33) }, {}#{"pat":True}
    #config_template["phy"]["vis"]["fiv"] = { "dr":(0,1.0,0), "ob":(0.33,0.33,0.33) } #"wo":(1.0,0,0),"wi":(0,0,1.0),
    #config_template["phy"]["grid"] = {"L":4.0,"d":0.1,"b":10}
    #config_template["adjs"] = {"decay":200.0,'inertia':0.0}
    #config_template["pat"]["prerec"] = False
    config_template["phy"]["s4"] = 4
    return parser.parse_args(sys.argv[1:]), config_template, sceneLst

if __name__ == "__main__": #load="testings",
    args, config, sceneLst = parse()
    from SceneClasses.Operation.Patn import patternManager as PM
    from SceneClasses.Operation.Optm import optm
    from SceneClasses.Basic.Scne import scne
    import os
    P = PM(args.version)
    for scene_uid in sceneLst:
        scene = scne.fromNpzs(name=scene_uid, grp=False, cen=False, wl=True, windoor=True, imgDir=os.path.join(".", "pattern", "opts", scene_uid))
        #print(scene)
        O = optm(P,scene,rand=args.rand,config=config)
        O.qualitative()

    # from SceneClasses.Basic.Scne import scneDs as SDS
    # SDS(args.dataset, lst=sceneLst,
    #     grp=False, cen=False, wl=True, windoor=True,
    #     #imgDir="./",os.path.join("./pattern/","optm",args.version),
    # ).optimize(
    #     PM(args.version),
    #     args.pattern, args.physics,
    #     1,#config["phy"]["steps"],
    #     config, args.rand
    # )
    