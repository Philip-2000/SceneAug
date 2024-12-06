from ..Operation.Patn import *
from ..Operation.Optm import *
from .ExPn import expn

class exop():
    def __init__():
        #总之就是实验的怎么进行的问题
        #（0）拿到场景之后随机打乱，尝试对场景进行“优化”，检测优化的过程以及时间

        #（a）操作就是逐步优化pat，优化phy

        #（b）有哪些有待测试的超参数？采样点采样率s4是不是？学习率？步数不算，门状态和墙壁范围不想改了意义不大
        #（c）优化的速度如何
            #（2.1）收敛步数？
            #（2.2）收敛时间？
        #（c）优化的效果？
            #（2.1）violate变小了吗，变小的速度如何？
            #（2.2）物体adjust变小了吗，变小的速度如何？
            #（2.3）recognize评分变好了吗，变好的速度如何？
            #（2.4）pat操作和phy操作冲突大吗
        pass

    def run():
        pass


    #怎么做？重新写吧，还挺有意思的。



# class expn():
#     def __init__(self,pmVersion,dataset,UIDS,expName,num,mt,task,roomMapping={}):
#         from ..Operation.Patn import patternManager as PM
#         from ..Basic.Scne import scneDs as SDS
#         self.pm = PM(pmVersion)
#         if dataset is not None:
#             self.sDs = SDS(dataset,lst=UIDS,num=num,grp=False,cen=True,wl=False,keepEmptyWL=True)
#         self.devs = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
#         self.ld = len(self.devs)
#         self.roomMapping = ma()#roomMapping #{"Bedroom":"Bedroom", "MasterBedroom":"Bedroom", "SecondBedroom":"Bedroom", "KidsRoom":"Bedroom", "ElderlyRoom":"Bedroom",
#                                         #"LivingRoom":"LivingRoom", "DiningRoom":"DiningRoom", "LivingDiningRoom":"LivingDiningRoom", "Library":"Library"}
#         self.rooms = list(set(self.roomMapping.values())) if self.roomMapping["Bedroom"] != "Room" else ["Room"]

#         self.result = [[ [] for j in self.rooms ] for i in self.devs]
        
#         self.visualDir = os.path.join(EXP_IMG_BASE_DIR,pmVersion,expName,task)
#         self.videoDir = os.path.join(EXP_IMG_BASE_DIR,pmVersion,expName,"video")
#         os.makedirs(self.visualDir,exist_ok=True)
#         os.makedirs(self.videoDir,exist_ok=True)
#         self.mt = mt
#         self.mtl = len(self.mt)

#     def randomize(self, s, dev, hint=None):
#         from copy import deepcopy
#         t = deepcopy(s)
#         diff = []
#         for o in t.OBJES:
#             dT = (np.random.randn((3)) if hint is None else hint[:3] + np.random.randn((3))*0.05) * dev* 0.1
#             o.translation += dT #np.random.randn((3)) * dev* 0.1
#             dS = (np.random.randn((3)) if hint is None else hint[3:-1] + np.random.randn((3))*0.05) * dev * 0.01
#             o.size += dS #np.random.randn((3)) * dev * 0.01
#             dO = (np.random.randn((1)) if hint is None else hint[-1:] + np.random.randn((1))*0.05) * dev* 0.1
#             o.orientation += dO #np.random.randn((1)) * dev*0.1
#             diff.append(np.concatenate([dT,dS,dO]))
#         return t, diff

#     def save(self):#roomSum = sum(roomCnt)
#         from itertools import chain
#         lens = [len(j) for j in self.result[0]]
#         locs = [sum(lens[:r+1]) for r in range(-1,len(self.rooms))]
#         data = np.array([list(chain(*(self.result[l]))) for l in range(self.ld) ])
#         np.savez_compressed(os.path.join(self.visualDir,"result.npz"),data=data,locs=locs,rooms=self.rooms)
#         return locs, data

#     def load(self):#roomSum = sum(roomCnt)
#         result = np.load(os.path.join(self.visualDir,"result.npz"))
#         self.rooms = result["rooms"]
#         return result["locs"], result["data"]

#     def run(self, **kwargs):
#         pbar = tqdm.tqdm(range(len(self.sDs)))
#         for i in pbar:
#             pbar.set_description("%s experiment %s "%(self.__class__.__name__,self.sDs[i].scene_uid[:20]))
#             s=self.sDs[i]
#             res = self.execute(s, None, **kwargs)#[fit,n]
#             self.store(0,s.roomType,res)
#             for dev in self.devs[1:]:
#                 rands,diff = self.randomize(s,dev)
#                 res = self.execute(rands,diff, **kwargs)#[fitr,nr] rands.recognize(self.pm) #plans(rands,self.pm,v=0).recognize(draw=False,**kwargs)
#                 self.store(self.devs.index(dev),s.roomType,res)
#         self.save()
#         self.visualize()

#     def store(self,id,roomType,res):
#         self.result[id][self.rooms.index(self.roomMapping[roomType])].append(res)

#     def visualSingles(self,data,figName=""):
#         xTitles = self.devs
#         metricsTitles = self.mt
        
#         from matplotlib import pyplot as plt
#         import os, numpy as np
#         N = (len(xTitles)-1) + len(xTitles)*len(metricsTitles)
#         fig, (box,bar) = plt.subplots(1,2,figsize=((N/7)*4, 16)) #  2:1 
#         #fig.suptitle(os.path.basename(os.path.splitext(figName)[0]),fontsize=18)


#         fontCap, fontBar, fontCnt = "medium", "xx-large", "xx-large"
#         mO = metricsTitles
#         mT = len(mO)
#         mI = [ metricsTitles.index( mO[mi] ) for mi in range(mT) ]
#         bar.set_position([0.02,0.05,0.95,0.43])
#         box.set_position([0.02,0.55,0.95,0.43])
#         scl,gap,colors,dolors,dNames = 0.18, 0.9, ["#BF9A6D","#578279","#B29D94","#BB7967","#81875A"], ["darkblue","#513E1B","yellow"], ["medians","means"]
#         barsArea = 0.9
#         barArea = 0.8
#         dMeans = dNames.index("means")
#         ds = len(dNames)
#         barL = scl*barsArea/ds
        
#         exps = np.average(data.reshape((-1,mT)),axis=0)
#         devs = (np.average((data.reshape((-1,mT)) - exps.reshape((1,-1)))**2,axis=0))**0.5


#         #-----------initialization----------
#         #-----------box plot----------------

#         bars,Xs = [],[]
#         for i in range(len(xTitles)):
#             for j in range(mT):
#                 X,exp,dev = (i*(mT+gap)+j)*scl, exps[mI[j]], devs[mI[j]]
#                 A = box.boxplot((data[i,:,mI[j]]-exp)/dev, positions=[X], labels = [mO[j]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[j]}, medianprops={"color":dolors[dNames.index("medians")]}, meanprops={'marker':"*","markeredgecolor":dolors[dMeans],"markerfacecolor":dolors[dMeans]})
#                 capsUp, capsDn,x= A["caps"][0].get_data()[1][0], A["caps"][1].get_data()[1][0], A["means"][0].get_data()[0][0]
#                 VcapsUp,VcapsDn = capsUp*dev+exp, capsDn*dev+exp
#                 box.annotate("%.2f"%(VcapsUp), xy=(x-(0.04 if abs(VcapsUp)/10 < 1 else 0.08),capsUp-0.2 ),fontsize=fontCap, bbox={"facecolor":'white', "edgecolor":'white', "boxstyle":'round'})
#                 box.annotate("%.2f"%(VcapsDn), xy=(x-(0.04 if abs(VcapsDn)/10 < 1 else 0.08),capsDn+0.08),fontsize=fontCap, bbox={"facecolor":'white', "edgecolor":'white', "boxstyle":'round'})
#                 Xs = Xs + [X-(ds-1)*(barL/2)+k*barL  for k in range(ds)] #[X-0.2*scl,X+0.2*scl]
#                 bars = bars + [A[dNames[k]][0].get_data()[1][0] for k in range(ds)]     

#         box.set_xticks([(i*(mT+gap)+(mT-1)/2.0)*scl for i in range(len(xTitles))], xTitles, fontsize=25)
#         box.set_ylabel("")
#         box.set_yticklabels("")
#         box.legend(handles=[plt.Line2D([0],[0],color=colors[i],lw=5,label=mO[i]) for i in range(mT)], labels=["%s(%.3f)"%(mO[j], np.var( [ bars[(i*mT+j)*ds+dMeans] for i in range(len(xTitles))]  ) ) for j in range(mT)], ncol=mT) #, labelcolor=colors, loc='lower left'
#         #box.set_title("overall",fontsize=15)


#         #-----------box plot----------------
#         #-----------bar plot----------------

#         y_min, y_max = box.get_ylim()

#         from itertools import chain
#         bar.bar(Xs, np.array(bars)-y_min, bottom=y_min, width=barArea*barL, color=(list(chain(*[[c]*ds for c in colors[:mT]])))*(int(len(Xs)/ (ds*mT) )), edgecolor=dolors[:ds]*(int(len(Xs)/ds)))

#         for i in range(len(xTitles)):
#             for j in range(mT):
#                 I0,exp,dev = (i*mT+j)*ds, exps[mI[j]], devs[mI[j]]
#                 ord = sorted([ (k,bars[I0+k],bars[I0+k]*dev+exp) for k in range(ds)],key= lambda x:-x[1])
#                 for o in ord:
#                     y = 2.5+(len(ord)-1)*0.3-ord.index(o)*0.6
#                     bar.annotate("%.2f"%(o[2]), xy=(Xs[I0]-(0.02 if abs(o[2])/10 < 1 else 0.10),y),fontsize=fontBar , color=dolors[o[0]], bbox={"facecolor":'white', "edgecolor":'white', "boxstyle":'round'})

#         bar.set_xticks([(i*(mT+gap)+(mT-1)/2.0)*scl for i in range(len(xTitles))], xTitles, fontsize=25)
#         bar.set_ylim(y_min,y_max)
#         bar.set_ylabel("")
#         bar.set_yticklabels("")
#         bar.legend(handles=[plt.Rectangle((0,0),width=1.0, height=0.8,facecolor="white",edgecolor=dolors[k],linewidth=0.4,label=dNames[k]) for k in range(ds)], labels=dNames, ncol=ds) #, labelcolor=colors, loc='lower left'
#         bar.set_title( " & ".join(dNames),fontsize=30)


#         #os.makedirs(os.path.dirname(figName),exist_ok=True)
#         plt.savefig(os.path.join(self.visualDir,figName+".png"))
#         plt.clf()
#         return

#     def visualSingle(self,data,figName=""):
#         #data.shape = len(self.dev) = 10? : len(self.roomTypes) = 4 and num = ??? : len(xTitle) = 2/1
#         from matplotlib import pyplot as plt
#         fig, ax1 = plt.subplots()

#         metricsExp = np.average(data.reshape((-1,self.mtl)),axis=0)
#         metricsDev = (np.average((data.reshape((-1,self.mtl)) - metricsExp.reshape((1,-1)))**2,axis=0))**0.5
        
#         metricsStandard = np.array([[metricsExp[j],metricsDev[j]] for j in range(self.mtl)])

#         colors = ["khaki","paleturquoise","plum","tomato", "springgreen"]#print(metricsStandard)
#         scl=1
#         for i in range(len(self.devs)):
#             for j in range(len(self.mt)):
#                 st = metricsStandard[j]
#                 A = ax1.boxplot((data[i,:,j]-st[0].reshape((-1)))/st[1].reshape((-1)), positions=[(i*(self.mtl+2)+j)*scl], labels = [self.mt[j]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[j]})
#                 B = {"medians":A["medians"][0].get_data()[1][0],"means":A["means"][0].get_data()[1][0],"capsUp":A["caps"][0].get_data()[1][0],"capsDown":A["caps"][1].get_data()[1][0],"x":A["means"][0].get_data()[0][0]}
#                 ax1.text(B["x"]+0.1,B["means"],"%.2f"%((B["means"])*st[1]+st[0]),fontsize='xx-small')
#                 ax1.text(B["x"]+0.1,B["capsUp"]-0.05,"%.2f"%((B["capsUp"])*st[1]+st[0]),fontsize='xx-small')
#                 ax1.text(B["x"]+0.1,B["capsDown"]+0.05,"%.2f"%((B["capsDown"])*st[1]+st[0]),fontsize='xx-small')
        
#         ax1.set_xticks([(i*(self.mtl+2)+(self.mtl-1)/2.0)*scl for i in range(len(self.devs))], ["%.3f"%(d) for d in self.devs])
#         ax1.set_ylabel("")
#         ax1.set_yticklabels("")
        
#         if self.mtl>1:
#             ax1.legend(handles=[plt.Line2D([0],[0],color=colors[i],lw=5,label=self.mt[i]) for i in range(self.mtl)], labels=self.mt, ncol=self.mtl) #, labelcolor=colors, loc='lower left'
#         plt.savefig(os.path.join(self.visualDir,figName+".png"))
#         plt.clf()
#         plt.close()

#     def visualize(self):
#         locs, data = self.load()
#         self.visualSingles(data,figName="Overall")
#         for r in range(len(locs)-1):
#             if len(locs)>2:
#                 self.visualSingle(data[:,locs[r]:locs[r+1],:],figName=self.rooms[r])

#     def execute(self,scene,**kwargs):
#         pass




# class OptExpn(expn):
#     def __init__(self,pmVersion,dataset,UIDS,task,num):
#         super(OptExpn,self).__init__(pmVersion,dataset,UIDS,self.__class__.__name__,num,["modify"],task)
    
#     def loss(self,ope,noise):
#         return ope-noise

#     def execute(self, s, diff, **kwargs):
#         from ..Operation.Plan import plans
#         return plans(s,self.pm).recognize(draw=False,opt=True,**kwargs)#rands.recognize(self.pm)

#     def show(self):
#         raise NotImplementedError
#         return