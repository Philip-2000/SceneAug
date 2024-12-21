EXOP_BASE_DIR = "./experiment/opts/"
class exop():
    def __init__(self,pmVersion,dataset,UIDS,num,expName,dirName,mod,dev,config,task):
        from ..Basic.Scne import scneDs as SDS
        self.sDs = SDS(dataset,lst=UIDS,num=num,grp=False, cen=False, wl=True, windoor=True)
        self.pmVersion = pmVersion
        from SceneClasses.Operation.Patn import patternManager as PM 
        self.PM = PM(pmVersion)
        self.dev = dev
        self.config = config
        self.mod = mod
        self.origin = {s.scene_uid[:10]:[] for s in self.sDs}#the original results
        self.plots = {"fit":{},"vio":{},"adjs":{},"cos":{},"diff":{},"time":{},"steps":{}} #"the data for plots"
        self.task = task #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
        import os
        self.dirname = os.path.join(EXOP_BASE_DIR,expName,dirName)
        print(self.dirname)
        os.makedirs(self.dirname,exist_ok=True) #
        
    def __randomize(self,t):
        import numpy as np
        from ..Operation.Adjs import adjs,adj
        for o in t.OBJES:
            o.adjust = adj(T=np.random.randn((3))*self.dev,S=np.random.randn((3))*self.dev * 0.1,R=np.random.randn((1))*self.dev,o=o)#o.adjust()
        return adjs(t.OBJES)
    
    def debugdraw(self,s,step):
        s.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d+.png"%(s.scene_uid[:10],step))#return #

    def __call__(self):
        if self.task==2: #2:experiment, 1:process and visualize, 0:load and visualize
            from ..Operation.Optm import optm
            from ..Operation.Plan import plans
            for s in self.sDs:
                
                if self.mod == "prerec":
                    plans(s,self.PM,v=0).recognize(use=True,draw=False,show=False)
                    adjs0 = self.__randomize(s)
                elif self.mod == "postrec":
                    adjs0 = self.__randomize(s)
                    plans(s,self.PM,v=0).recognize(use=True,draw=False,show=False)

                OP = optm(pmVersion=self.pmVersion,scene=s,PatFlag=True,PhyFlag=True,config=self.config,exp=True)
                s.draw(imageTitle=EXOP_BASE_DIR+"debug/%s.png"%(s.scene_uid[:10]))
                step = 0
                ret = {"over":False}
                while (not ret["over"]): #the over criterion
                    ret = OP(step) #timer, adjs, vio, fit, cos, over
                    ret["diff"] = adjs0 - ret["adjs"]
                    self.store(ret,s,step)
                    self.debugdraw(s,step)
                    step += 1
                    if step > 33:
                        break

                #print(step,time)
            self.save()
        self.visualize()

    def store(self,ret,s,steps):
        assert steps == len(self.origin[s.scene_uid[:10]])
        ret["timer"]= ret["timer"].dct()
        ret["adjs"] = ret["adjs"].Norm()
        self.origin[s.scene_uid[:10]].append(ret)

    def store_plot(self,key,ele,value):
        self.plots[key][ele] = value

    def save(self,n="origin"):
        import json,os
        open(os.path.join(self.dirname,n+".json"),"w").write(json.dumps(self.origin if n == "origin" else self.plots))

    def load(self,n="origin"):
        import json,os
        if n == "origin":
            self.origin= json.load(open(os.path.join(self.dirname,n+".json"),"r"))
        else:
            self.plots = json.load(open(os.path.join(self.dirname,n+".json"),"r"))

    def visualize_box(self,key,content,boxes):
        from matplotlib import pyplot as plt
        import os, numpy as np
        #print(len(content))
        #print(len(content[0]))
        dolors = {"medians":"darkblue","means":"#513E1B"}
        colors = {"vio":"#BF9A6D","fit":"#578279","adjs":"#B29D94","cos":"#BB7967","diff":"#81875A"}
        
        #content: 20*72
        array = np.array(content).T
        means,Xs = [],[]#, positions=[X*5 for X in range(len(content))]
        A = plt.boxplot(array)#, labels = [content[1]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[key]}, medianprops={"color":dolors["medians"]}, meanprops={'marker':"*","markeredgecolor":dolors["means"],"markerfacecolor":dolors["means"]})
        #print(A["means"][0].get_data())
        #raise NotImplementedError
        #means = means + [A["means"][0].get_data()[1][0]] 

        # for j in range(len(content)):
        #     X = j*5
        #     A = plt.boxplot(content[j], positions=[X], labels = [content[1]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[key]}, medianprops={"color":dolors["medians"]}, meanprops={'marker':"*","markeredgecolor":dolors["means"],"markerfacecolor":dolors["means"]})
        #     means = means + [A["means"][0].get_data()[1][0]]     

        #plt.set_xticks([(i*(mT+gap)+(mT-1)/2.0)*scl for i in range(len(xTitles))], xTitles, fontsize=25)
        #plt.set_ylabel("")
        #plt.set_yticklabels("")
        #plt.legend(handles=[plt.Line2D([0],[0],color=colors[i],lw=5,label=mO[i]) for i in range(mT)], labels=["%s(%.3f)"%(mO[j], np.var( [ bars[(i*mT+j)*ds+dMeans] for i in range(len(xTitles))]  ) ) for j in range(mT)], ncol=mT) #, labelcolor=colors, loc='lower left'
        if self.task>=0: #save this plot if it is not visualized by exops
            plt.savefig(os.path.join(self.dirname,key+"-steps.png"))
            plt.clf()
        
    def visualize_line(self,key,ts,vs):
        #return
        from matplotlib import pyplot as plt
        plt.plot(ts,vs)
        if self.task>=0: #save this plot if it is not visualized by exops
            import os
            plt.savefig(os.path.join(self.dirname,key+"-time.png" if key in ["time","steps"] else key+".png"))
            plt.clf()

    def visualize_bar(self,key,hs,gap):
        from matplotlib import pyplot as plt
        import os
        #print(len(hs[0])) print(hs)
        plt.bar([gap*i for i in range(len(hs))],hs,0.4*gap)
        
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
        if self.task>=0: #save this plot if it is not visualized by exops
            plt.savefig(os.path.join(self.dirname,key+"-his.png"))
            plt.clf()

    def organize_metrics(self,key):
        #（1）vio（2）fit（3）adjs（4）cos:pat's trend and phy's trend（5）diff：adjust's trend and noise's trend

        #step-aligned flatten, as box-plot
        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize
            #print([len(self.origin[s.scene_uid[:10]]) for s in self.sDs])
            ms = max([len(self.origin[s.scene_uid[:10]]) for s in self.sDs])
            boxes = 34
            bs = [[] for _ in range(boxes)]
            for j in range(0,ms,ms//boxes):
                for i in range(j,min(j+ms//boxes,ms)):
                    bs[j] = bs[j] + [self.origin[s.scene_uid[:10]][min(i,len(self.origin[s.scene_uid[:10]]))][key] for s in self.sDs]
            self.store_plot(key,"bs",bs)
            self.store_plot(key,"boxes",boxes)
        else:
            bs = self.plots[key]["bs"]
            boxes = self.plots[key]["boxes"]
        if self.task > -1: #only lines are visualized by exops
            self.visualize_box(key,bs,boxes)
        
        #time-aligned flatten:
        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize
            ds = [self.origin[s.scene_uid[:10]][0][key] for s in self.sDs]
            mean = sum(ds)/float(len(self.sDs))
            ts, vs = [],[]
            idxs = [1 for s in self.sDs] #we should evaluate the initial value of these benchmarks as well, its time should always be zero
            T = 0
            while True:
                ts.append([T])
                vs.append([mean])

                ifirst,Tfirst = -1,100000
                for i,s in enumerate(self.sDs):
                    if idxs[i] < len(self.origin[s.scene_uid[:10]]):
                        t = self.origin[s.scene_uid[:10]][idxs[i]]["timer"]["accum"] #to be accumulative
                        if t < Tfirst:
                            Tfirst, ifirst = t,i
                if Tfirst == 100000:
                    break
                T = Tfirst
                
                mean -= ds[ifirst] / float(len(self.sDs))
                s = self.sDs[ifirst]
                ds[ifirst] = self.origin[s.scene_uid[:10]][idxs[ifirst]][key]
                mean += ds[ifirst] / float(len(self.sDs))
                idxs[ifirst] += 1
            self.store_plot(key,"ts",ts)
            self.store_plot(key,"vs",vs)
        else:
            ts = self.plots[key]["ts"]
            vs = self.plots[key]["vs"]
        self.visualize_line(key,ts,vs)

    def organize_steps(self,key):
        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            vs = [self.origin[s.scene_uid[:10]][-1]["timer"]["accum"] for s in self.sDs] if key == "time" else [len(self.origin[s.scene_uid[:10]]) for s in self.sDs]
            vs = sorted(vs)
            ts,ls = [],[]
            T,l = 0,len(vs)
            for v in vs:
                ts.append(v)
                ls.append(l)
                l -= 1
            self.store_plot(key,"ts",ts)
            self.store_plot(key,"ls",ls)
        else:
            ts = self.plots[key]["ts"]
            ls = self.plots[key]["ls"]
        self.visualize_line(key,ts,ls)

        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            vs = [self.origin[s.scene_uid[:10]][-1]["timer"]["accum"] for s in self.sDs] if key == "time" else [len(self.origin[s.scene_uid[:10]]) for s in self.sDs]
            vs = sorted(vs)
            #print(vs)
            boxes = 34
            gap = max(vs)/float(boxes)
            hs =  [0 for b in range(boxes)]
            i = 0
            for b in range(1,boxes+1):
                while vs[i]<b*gap:
                    hs[b-1],i = hs[b-1]+1,i+1
            self.store_plot(key,"hs",hs)
            self.store_plot(key,"boxes",boxes)
            self.store_plot(key,"gap",gap)
        else:
            hs = self.plots[key]["hs"]
            boxes = self.plots[key]["boxes"]
            gap = self.plots[key]["gap"]
        if self.task > -1: #only lines are visualized by exops
            self.visualize_bar(key,hs,gap)
        
    def visualize(self,keys=["time","steps","vio","fit","adjs","cos","diff"]):
        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            self.load("origin")
        elif self.task == 0: #when task is -1, we have already loaded the data in exops
            self.load("plots")
        for key in keys:
            _ = self.organize_steps(key) if key in ["time","steps"] else self.organize_metrics(key)
        if self.task >= 1: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            self.save("plots")

class exops():
    def __init__(self,pmVersion='losy',dataset="../novel3DFront/",task=2,expName="test",UIDS=[]):
        self.UIDS = UIDS
        if len(UIDS)==0:
            self.UIDS = [
                "0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487",
                "0de89e0a-723c-4297-8d99-3f9c2781ff3b_LivingDiningRoom-18932",
                "1a5bd12f-4877-405c-bb58-9c6bfcc0fb62_LivingRoom-53927",
                "1befc228-9a81-4936-b6a1-7e1b67cee2d7_Bedroom-352",
                "34f5f040-eb63-482b-82cb-9a3914c92c79_LivingDiningRoom-8678",
                "328ada87-9de8-4283-879d-58bffe5eb37a_Bedroom-5280",
                "39629e24-b405-420b-8fb0-72cef0238f70_SecondBedroom-1255",
                "4efedd5d-31d9-46c2-8c26-94ebdd7c0187_MasterBedroom-39695",
                "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
                "0ead178d-e4db-4b93-a9d0-0a8ee617d879_LivingRoom-18781",
                "001ef085-8b13-48ec-b4e4-4a0dc1230390_KidsRoom-1704",
                "1c70b531-095e-44aa-9284-6585b89c4d56_DiningRoom-78405",
                "1e6211be-ba2f-4dc0-9206-c5dcd4ae85be_LivingDiningRoom-3184",
                "05a80889-128e-40b0-8375-fea9856931b8_LivingDiningRoom-64430",
                "1e442945-e065-4453-b056-ddbb916e5c7c_SecondBedroom-1995",
                "5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088"
            ]
        self.num=0 #self.num=512, self.UIDS = []
        self.task = task #2:experiment, 1:process and visualize, 0:load and visualize
        self.pmVersion=pmVersion
        self.dataset=dataset
        self.expName = expName
        import os
        os.makedirs(os.path.join(EXOP_BASE_DIR,expName),exist_ok=True)

        self.hypers = []
        self.mods = ["prerec",]#"postrec",
        self.s4s = [2]
        self.rates =[ 0.1*i for i in range(1,4)]
        self.devs = [ 0.1*i for i in range(1,6)]
        self.hypers = [[[[ (mod,rate,s4,dev) for dev in self.devs ] for rate in self.rates] for s4 in self.s4s ] for mod in self.mods] 
        self.exops = [[[[ None for dev in self.devs ] for rate in self.rates]  for s4 in self.s4s ]for mod in self.mods] 
        
    def __call__(self):
        if self.task==2: #2:experiment, 1:process and visualize, 0:load and visualize
            for m,mod in enumerate(self.mods):
                for s,s4 in enumerate(self.s4s):
                    for r,rate in enumerate(self.rates):
                        for d,dev in enumerate(self.devs):
                            config = {
                                "pat":{
                                    "rerec":bool(mod=="rerec"),
                                    "prerec":bool(mod=="prerec"),
                                    "rand":False,
                                    "rate":{"mode":"exp_dn","r0":rate*2,"lda":0.5,"rinf":rate/5.0},#{"mode":"static","v":rate},
                                },
                                "phy":{
                                    "rate":{"mode":"exp_up","rinf":rate*10,"lda":1.5,"r0":rate/100.0},#{"mode":"static","v":rate/500},#
                                    "s4": s4,
                                    "door":{"expand":0.6,"out":0.1,"in":0.2,},
                                    "wall":{"bound":0.5,},
                                    "object":{
                                        "Pendant Lamp":[.0,.01,.01,False],#
                                        "Ceiling Lamp":[.0,.01,.01,False],#
                                        "Bookcase / jewelry Armoire":[.2,1., .9,True],#
                                        "Round End Table":[.0,.5, .5,False],#
                                        "Dining Table":[.0,.5, .5,False],#
                                        "Sideboard / Side Cabinet / Console table":[.0,.9, .9,True],#
                                        "Corner/Side Table":[.0,.9, .9,True],#
                                        "Desk":[.0,.9, .9,True],#
                                        "Coffee Table":[.0,1.,1.1,False],#
                                        "Dressing Table":[.0,.9, .9,True],#
                                        "Children Cabinet":[.2,1., .9,True],#
                                        "Drawer Chest / Corner cabinet":[.2,1., .9,True],#
                                        "Shelf":[.2,1., .9,True],#
                                        "Wine Cabinet":[.2,1., .9,True],#
                                        "Lounge Chair / Cafe Chair / Office Chair":[.0,.5, .5,False],#
                                        "Classic Chinese Chair":[.0,.5, .5,False],#
                                        "Dressing Chair":[.0,.5, .5,False],#
                                        "Dining Chair":[.0,.5, .5,False],#
                                        "armchair":[.0,.5, .5,False],#
                                        "Barstool":[.0,.5, .5,False],#
                                        "Footstool / Sofastool / Bed End Stool / Stool":[.0,.5, .5,False],#
                                        "Three-seat / Multi-seat Sofa":[.2,1., .9,True],#
                                        "Loveseat Sofa":[.2,1., .9,True],#
                                        "L-shaped Sofa":[.0,.6, .9,True],#
                                        "Lazy Sofa":[.2,1., .9,True],#
                                        "Chaise Longue Sofa":[.2,1., .9,True],#
                                        "Wardrobe":[.2,1., .9,True],#
                                        "TV Stand":[.2,1., .9,True],#
                                        "Nightstand":[.0,.5, .5,True],#
                                        "King-size Bed":[.2,1.,1.2,True],#
                                        "Kids Bed":[.2,1.,1.2,True],#
                                        "Bunk Bed":[.2,1.,1.2,True],#
                                        "Single bed":[.2,1.,1.2,True],#
                                        "Bed Frame":[.2,1.,1.2,True],#
                                    },
                                    "syn":{"T":1.1,"S":0.1,"R":1.0,},
                                },
                                "adjs":{
                                    "inertia":0.,"decay":1.0,
                                }
                            }
                            dirName = "%s %.2f %.3f %d"%(mod,dev,rate,s4)
                            Exop = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,num=self.num,expName=self.expName,dirName=dirName,mod=mod,dev=dev,config=config,task=self.task)
                            Exop()

        self.visualize()
            
    def load(self):
        for m,mod in enumerate(self.mods):
            for s,s4 in enumerate(self.s4s):
                for r,rate in enumerate(self.rates):
                    for d,dev in enumerate(self.devs):
                        dirName = "%s %.2f %.3f %d"%(mod,dev,rate,s4)
                        self.exops[m][s][r][d] = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,num=self.num,expName=self.expName,dirName=dirName,mod=mod,dev=dev,config=None,task=-1)
                        self.exops[m][s][r][d].load(n="plots")
    
    def visualize(self):
        self.load()
        assert len(self.mods)==1
        assert len(self.s4s)==1
        from matplotlib import pyplot as plt
        import os
        for m,mod in enumerate(self.mods):
            for s,s4 in enumerate(self.s4s):
                for key in ["time","steps","vio","fit","adjs","cos","diff"]:
                    for r,rate in enumerate(self.rates):
                        for d,dev in enumerate(self.devs):
                            self.exops[m][s][r][d].visualize(keys=[key])
                        plt.legend(["dev=%.3f"%(dev) for dev in self.devs])
                        plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s rate=%.3f.png"%(key,rate)))
                        plt.clf()

                    for d,dev in enumerate(self.devs):
                        for r,rate in enumerate(self.rates):
                            self.exops[m][s][r][d].visualize(keys=[key])
                        plt.legend(["rate=%.3f"%(rate) for rate in self.rates])
                        plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s dev=%.3f.png"%(key,dev)))
                        plt.clf()        