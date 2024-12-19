'''
Randomly initialize a scene dataset
- create scene from npz or json
    - disrupt existing objects in the scene
    - or sample objects from the tree and randomly initialize
- create scene from explicitly given wall list
    - sample objects from the tree and randomly initialize
'''

def parse(): 
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Pattern Manager Construction')
    parser.add_argument("-v","--version", required=True)#,   default="brot"
    parser.add_argument('-e','--verbose',   default=1, type=int)
    parser.add_argument("-a","--dataset",   default="../novel3DFront/")
    parser.add_argument('-s','--scaled',    default=True, action="store_true")
    parser.add_argument('-d','--maxDepth',  default=-1, type=int)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = parse()
    import os
    from SceneClasses.Operation.Patn import patternManager as PM
    from SceneClasses.Basic.Scne import scneDs as SDS

    # create a tree
    if args.dataset == "":
        PM(args.version).draw()
    T,S = PM(args.version,verb=args.verbose,new = True), SDS(args.dataset,lst=os.listdir(args.dataset),grp=False,wl=False,keepEmptyWL=True,cen=True,rmm=False)
    T.construct(args.maxDepth,args.scaled,S)

    # randomly sampling
    # create an empty scene dataset
    rand_scene = SDS(None, [], prepare='uncond') # empty dataset
    wall_list = None # TODO
    rand_scene.randSynthesisFromWL(T, wall_list, scene_num=10, obj_num=[5,9], max_tr=2.0, max_sc=1.2)

    # create a scene from npz or json
    rand_scene = SDS(args.dataset, os.listdir(args.dataset), prepare='uncond')
    rand_scene.randDisrupt(0.5)

    rand_scene = SDS(args.dataset, os.listdir(args.dataset), prepare='uncond')
    rand_scene.randSynthesis(T, obj_num=[6,8], max_tr=2.0, max_sc=1.2)