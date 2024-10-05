from SceneClasses.Spce import * #construting spaces
def parse():
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Space Extraction')
    parser.add_argument('-i','--identity', default="3")
    parser.add_argument('-n','--new', default=False, action="store_true")
    parser.add_argument('-b','--bound', default=19)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    DIR,args = "./newRoom/",parse()
    for i in (range(9,args.bound) if int(args.identity) == -1 else range(int(args.identity),int(args.identity)+1)):
        if args.new:
            wls = walls(name="rand"+str(i))#print(wls.LOGS)
            wls.randomWalls()
            wls.output()
        
        #print(i)
        wlz = walls.fromLog(f=DIR+"rand"+str(i)+".txt",name="rand"+str(i)+"_",drawFolder=DIR) #wlz.draw(DIR)
        #print(wlz)
        sm = spces(wals=wlz,name=wlz.name,drawFolder=DIR+wlz.name+"/")
        sm.extractingSpces(2)
        #sm.extractingMoreSpces()


    # DIR = "./newRoom/"
    # W = walls.fromLog(f=DIR+args.wid+".txt",name=args.wid+"_") #wlz.draw(DIR)
    # #print(W)
    # #raise NotImplementedError
    # S = scne.empty(args.wid+"_")
    # S.registerWalls(W)
    # T.generate(nm="testing",theScene=S,useWalls=True,debug=True)