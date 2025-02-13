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
# else:
#     from matplotlib import pyplot as plt
#     import numpy as np
#     z= ["4","5","6-10","11+"]
#     c= ["#69C9B1","#99B999","#E79A78","#BD683A"]
#     m,n=[[91,  4, 4, 1], [49, 45, 4, 2], [17, 77, 3, 3], [ 5, 87, 4, 4], [ 4, 84, 9, 3]], ["dev\n0.5","dev\n1.0","dev\n1.5","dev\n2.0","dev\n2.5"]
#     #m,n=[[32, 61, 4, 3], [33, 57, 7, 3], [34, 58, 6, 2]], ["M=8","M=12","M=16"] 
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
#     plt.savefig("./experiment/opts/last/steps dev.png")