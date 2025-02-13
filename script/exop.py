def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Optimizations\'s Application')
    #parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  type=int, default=1)
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__": #load="testings",
    from SceneClasses.Experiment import exops
    args = parse()
    E = exops(task=args.task)
    E()
#     from matplotlib import pyplot as plt
#     import numpy as np
#     z= ["4","5","6-10","11+"]
#     c= ["#69C9B1","#99B999","#E79A78","#BD683A"]
#     #m,n=[[32, 58, 7, 3], [13, 81, 5, 1], [6, 84, 7, 3], [ 5, 85, 8, 2], [ 2, 75, 20, 3]], ["dev\n1.0","dev\n1.5","dev\n2.0","dev\n2.5","dev\n3.0"]
#     m,n=[[11, 74, 11, 4], [11, 77, 10, 2], [12, 79, 8, 1]], ["M=8","M=12","M=16"] 
#     x = np.arange(len(m)) #if len(m)==5 else np.arange(0,5,2)
#     width = 0.5 if len(m)==5 else 0.4
#     malef = np.zeros(len(m))
#     ax = plt.subplot()
#     for i in range(len(z)):
#         I = 3-i if len(m)==3 else i
#         femalef = np.array(m)[:,I]
#         ax.barh(x, femalef, width, left = malef, align='center', color=c[I], \
#             label=z[I],tick_label=n) #, alpha=0.5
#         for j in range(len(m)):
#             ax.text(malef[j]+femalef[j]/2, j+width/1.6, "%d"%(femalef[j]), fontsize = 10, ha='center', va='center')
#         malef = malef + femalef
#     if len(m)==3:
#         ax.set_xticks(np.arange(0,101,20))
#         ax.set_xticklabels(np.arange(100,-1,-20))
#         ax.yaxis.tick_right()
#     else:
#         ax.set_xticks(np.arange(0,101,20))
#         ax.set_xticklabels(np.arange(0,101,20))
#         ax.yaxis.tick_left()
#     plt.savefig("./experiment/opts/tmer/steps s4.png")