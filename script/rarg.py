def parse(): #using pattern to synthesis scene
    import argparse,sys
    parser = argparse.ArgumentParser(prog='a temporary sricpt')
    parser.add_argument("-v","--version",   default="losy")#,   default="brot"
    parser.add_argument('-n','--name',      default="test")
    parser.add_argument('-e','--verbose',   default=0, type=int)
    parser.add_argument("-q","--n_sequence",default=16, type=int)
    parser.add_argument("-d","--dataset",   default="../novel3DFront/", choices=["","../novel3DFront/"])
    parser.add_argument("-a","--application",default="gnrt",    choices=["gnrt","copl","rarg","agmt"])
    parser.add_argument("-c","--condition", default="uncond",   choices=["uncond","textcond","roomcond"])
    parser.add_argument('-i','--id',        default="")#ede1bcab-2298-4756-b03a-690d5cf8dfe5_LivingDiningRoom-199172024

    sceneLst = [
        #"5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
        # "0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487",
        # "1a5bd12f-4877-405c-bb58-9c6bfcc0fb62_LivingRoom-53927",
        # "1befc228-9a81-4936-b6a1-7e1b67cee2d7_Bedroom-352",
        # "0de89e0a-723c-4297-8d99-3f9c2781ff3b_LivingDiningRoom-18932",
        # "34f5f040-eb63-482b-82cb-9a3914c92c79_LivingDiningRoom-8678",
        # "328ada87-9de8-4283-879d-58bffe5eb37a_Bedroom-5280",
        # "39629e24-b405-420b-8fb0-72cef0238f70_SecondBedroom-1255",
        # "4efedd5d-31d9-46c2-8c26-94ebdd7c0187_MasterBedroom-39695",
    ]
    return parser.parse_args(sys.argv[1:]), sceneLst

if __name__ == "__main__": 
    args, lst = parse()
    from SceneClasses.Basic import scneDs as SDS
    from SceneClasses.Operation import patternManager as PM
    pm = PM(args.version)
    SDS(name=args.dataset,lst=(lst),prepare=args.condition).synthesis("rarg",args.condition,pm, use=True, draw=True)