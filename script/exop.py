def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Optimizations\'s Application')
    #parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  type=int, default=2)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
#     from SceneClasses.Basic.Scne import scneDs as SDS
#     import sys
#     sds = SDS("../novel3DFront/", num=32)
#     T = []
#     f = open("./txt.txt","w")
#     a = sys.stdout
#     sys.stdout = f
#     for s in sds:
#         if s.scene_uid.find("LivingDiningRoom") != -1:
#             print(s)
#             T.append(s.scene_uid)
#         if len(T)>4:
#             break
#     print("[\n\"" + "\",\n\"".join(T) + "\",\n]")
#     sys.stdout = a
        
# else:
    from SceneClasses.Experiment.ExOp import exops
    args = parse()
    E = exops(task=args.task)
    E()