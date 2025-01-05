from matplotlib import pyplot as plt
import json,os,numpy as np
EXOP_BASE_DIR = "./experiment/opts/"
DEBUG_FILTER = []#["","+"]
class exop():
    def __init__(self,pmVersion,dataset,UIDS,num,expName,dirName,dev,config,task):
        from ..Basic.Scne import scneDs as SDS
        self.sDs = SDS(dataset,lst=UIDS,num=num,grp=False, cen=False, wl=True, windoor=True)
        self.pmVersion = pmVersion
        from SceneClasses.Operation.Patn import patternManager as PM 
        self.PM = PM(pmVersion)
        self.dev = dev
        self.config = config
        self.mod = "prerec"
        self.origin = {s.scene_uid[:10]:[] for s in self.sDs}#the original results
        self.plots = {"fit":{},"vio":{},"adjs":{},"cos":{},"diff":{},"time":{},"steps":{}} #"the data for plots"
        self.task = task #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
        self.dirname = os.path.join(EXOP_BASE_DIR,expName,dirName)
        print(self.dirname)
        os.makedirs(self.dirname,exist_ok=True) #
    
    def __call__(self):
        if self.task==2: #2:experiment, 1:process and visualize, 0:load and visualize
            from ..Operation.Optm import optm
            from ..Operation.Plan import plans
            for s in self.sDs:
                
                if self.mod == "prerec":
                    plans(s,self.PM,v=0).recognize(use=True,draw=False,show=False)
                    adjs0 = self.__randomize(s)
                elif self.mod == "postrec":
                    raise Exception("Forbid to use postrec")
                    adjs0 = self.__randomize(s)
                    plans(s,self.PM,v=0).recognize(use=True,draw=False,show=False)

                OP = optm(pmVersion=self.pmVersion,scene=s,PatFlag=True,PhyFlag=True,config=self.config,exp=True)
                self.debugdraw(s,0,"")#s.draw(imageTitle=EXOP_BASE_DIR+"debug/%s.png"%(s.scene_uid[:10]))
                step = 0
                ret = {"over":False}
                [o.adjust.clear() for o in s.OBJES]
                while (not ret["over"]): #the over criterion
                    ret = OP(step,self.debugdraw) #timer, adjs, vio, fit, cos, over
                    ret["diff"] = adjs0 - ret["adjs"]
                    self.store(ret,s,step)
                    self.debugdraw(s,step,"+")#self.debugdraw(s,step)
                    step += 1
                    if step > 33:
                        break

                #print(step,time)
            self.save()
        self.visualize()

    #region: buffer
    def store(self,ret,s,steps):
        if steps != len(self.origin[s.scene_uid[:10]]):
            print(steps,s.scene_uid[:10])
        assert steps == len(self.origin[s.scene_uid[:10]])
        ret["timer"]= ret["timer"].dct()
        ret["adjs"] = ret["adjs"].Norm()
        self.origin[s.scene_uid[:10]].append(ret)

    def store_plot(self,key,ele,value):
        self.plots[key][ele] = value

    def save(self,n="origin"):
        from moviepy.editor import ImageSequenceClip
        open(os.path.join(self.dirname,n+".json"),"w").write(json.dumps(self.origin if n == "origin" else self.plots))
        for s in self.sDs:
            _ = ImageSequenceClip([os.path.join(EXOP_BASE_DIR,"debug","%s-%d%s.png"%(s.scene_uid[:10],i,"+")) for i in range(1,33)],fps=3).write_videofile(os.path.join(EXOP_BASE_DIR,"%s.mp4"%(s.scene_uid[:10])),logger=None) if "+" in DEBUG_FILTER and n=="origin" else None

    def load(self,n="origin"):
        if n == "origin":
            self.origin= json.load(open(os.path.join(self.dirname,n+".json"),"r"))
        else:
            self.plots = json.load(open(os.path.join(self.dirname,n+".json"),"r"))
    #endregion: buffer

    #region: visualize
        #region: visualize plots
    def visualize_box(self,key,content,boxes):
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
        plt.plot(ts,vs)
        if self.task>=0: #save this plot if it is not visualized by exops
            plt.savefig(os.path.join(self.dirname,key+"-time.png" if key in ["time","steps"] else key+".png"))
            plt.clf()

    def visualize_bar(self,key,hs,gap):
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
        #endregion: visualize plots    
    
        #region: organize data to plot
    def organize_metrics(self,key):
        #（1）vio（2）fit（3）adjs（4）cos:pat's trend and phy's trend（5）diff：adjust's trend and noise's trend

        #step-aligned flatten, as box-plot
        if self.task > 0: #2:experiment, 1:process and visualize
            #print([len(self.origin[s.scene_uid[:10]]) for s in self.sDs])
            ms = max([len(self.origin[s.scene_uid[:10]]) for s in self.sDs])
            boxes = 34
            bs = [[] for _ in range(boxes)]
            for j in range(0,ms,ms//boxes):
                for i in range(j,min(j+ms//boxes,ms)):
                    bs[j] = bs[j] + [self.origin[s.scene_uid[:10]][min(i,len(self.origin[s.scene_uid[:10]]))][key] for s in self.sDs]
            self.store_plot(key,"bs",bs)
            self.store_plot(key,"boxes",boxes)
        else: #0:load and visualize, -1:load and visualized by exops
            bs = self.plots[key]["bs"]
            boxes = self.plots[key]["boxes"]
        if self.task > -1: #only lines will be visualized by exops
            self.visualize_box(key,bs,boxes)
        
        #time-aligned flatten:
        if self.task > 0: #2:experiment, 1:process and visualize
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
        else: #0:load and visualize, -1:load and visualized by exops
            ts = self.plots[key]["ts"]
            vs = self.plots[key]["vs"]
        self.visualize_line(key,ts,vs)

    def organize_steps(self,key):
        if self.task > 0: #2:experiment, 1:process and visualize
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
        else: #0:load and visualize, -1:load and visualized by exops
            ts = self.plots[key]["ts"]
            ls = self.plots[key]["ls"]
        self.visualize_line(key,ts,ls)

        if self.task > 0: #2:experiment, 1:process and visualize
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
        else: #0:load and visualize, -1:load and visualized by exops
            hs = self.plots[key]["hs"]
            boxes = self.plots[key]["boxes"]
            gap = self.plots[key]["gap"]
        if self.task > -1: #only lines will be visualized by exops
            self.visualize_bar(key,hs,gap)
        #endregion: organize data to plot

        #region: visualize main
    def visualize(self,keys=["time","steps","vio","fit","adjs","cos","diff"]):
        if self.task > 0: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            self.load("origin")
        elif self.task == 0: #when task is -1, we have already loaded the data in exops
            self.load("plots")
        for key in keys:
            _ = self.organize_steps(key) if key in ["time","steps"] else self.organize_metrics(key)
        if self.task >= 1: #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized by exops
            self.save("plots")
        #endregion: visualize main
    #endregion: visualize

    #region: util
    def debugdraw(self,s,step,suffix):
        s.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d%s.png"%(s.scene_uid[:10],step,suffix)) if suffix in DEBUG_FILTER else None#return #
        
    def __randomize(self,t):
        a,b = t.randomize(dev=self.dev,cen=True)
        return a
    #endregion: util
class exops():
    def __init__(self,pmVersion='losy',dataset="../novel3DFront/",task=2,expName="test",UIDS=[]):
        self.UIDS = UIDS
        if len(UIDS)==0:
            self.UIDS = [
                "0ea43759-83d3-4042-9988-dc86fe75e462_LivingDiningRoom-1933",
                "0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487",
                "0de89e0a-723c-4297-8d99-3f9c2781ff3b_LivingDiningRoom-18932",
                "1a5bd12f-4877-405c-bb58-9c6bfcc0fb62_LivingRoom-53927",
                "1befc228-9a81-4936-b6a1-7e1b67cee2d7_Bedroom-352",
                "34f5f040-eb63-482b-82cb-9a3914c92c79_LivingDiningRoom-8678",
                "328ada87-9de8-4283-879d-58bffe5eb37a_Bedroom-5280",
                "39629e24-b405-420b-8fb0-72cef0238f70_SecondBedroom-1255",
                "4efedd5d-31d9-46c2-8c26-94ebdd7c0187_MasterBedroom-39695",
                "0ead178d-e4db-4b93-a9d0-0a8ee617d879_LivingRoom-18781",
                "001ef085-8b13-48ec-b4e4-4a0dc1230390_KidsRoom-1704",
                "1c70b531-095e-44aa-9284-6585b89c4d56_DiningRoom-78405",
                "1e6211be-ba2f-4dc0-9206-c5dcd4ae85be_LivingDiningRoom-3184",
                "05a80889-128e-40b0-8375-fea9856931b8_LivingDiningRoom-64430",
                "1e442945-e065-4453-b056-ddbb916e5c7c_SecondBedroom-1995",
                "5e6f0a50-b34c-45a8-8e31-55c7d9adad2d_MasterBedroom-92088",
                "14d1fa1d-6421-4457-81ad-46db84d6e3d4_LivingDiningRoom-1140",
                "120b8cb2-aa75-49d8-8928-ee4dbb177966_LivingDiningRoom-617192024",
                "5e647f08-d821-4554-8598-9684ffd28f7c_LivingDiningRoom-3367",
                "f1a605ec-155d-48fd-ac89-d54c3ebdbfb6_LivingDiningRoom-3581",
                "d0615bd5-c5af-474b-a2bf-c434fcfaf74c_MasterBedroom-110",
                "4a102195-8b2f-444d-b95d-3b332822bc9a_LivingDiningRoom-152",
                "0e44eab5-891c-49df-8311-4395a610016d_LivingDiningRoom-574142024",
                "5530455f-bf14-464b-88c0-7eaee2d2d85b_LivingDiningRoom-3064",
                "0685adc2-121b-4ea7-894d-6a81cb779be9_LivingDiningRoom-10913",
                "0b35b16b-f277-427a-8274-09c50f6e2b99_LivingDiningRoom-7940",
                "12219573-1263-4658-a5db-d048f3f0d668_MasterBedroom-569",
                "908633f6-545c-4016-a2e7-f334120ac392_MasterBedroom-7728",
                "196930e3-70ae-4432-bf23-7a706f5feb22_LivingDiningRoom-109282024",
                "43f250b1-17a4-4ac7-af2d-ba722f3a8bcb_LivingDiningRoom-449432024",
                "1b8e0aa0-1feb-4030-b45b-ba797530c57f_LivingDiningRoom-36888",
                "6e4825e3-4ceb-40d4-a7f8-2adf7a3ab6f5_MasterBedroom-536",
                "822213ac-e850-4189-ae92-5a9e0946be04_Bedroom-72334",
                "00d15904-be14-4958-b67b-5a0137a279fa_SecondBedroom-1070",
                "967e40cb-ac87-4ff0-90f0-f775d80ca4ae_LivingDiningRoom-1110",
                "1b6ad955-6e6e-4a87-9a46-ddfd3bd91a50_LivingDiningRoom-616382024",
                "6e3d565d-89ce-4a2d-b39b-aef362e9857e_MasterBedroom-4091",
                "3d048fd0-a931-4343-bd40-0f62860034a1_Bedroom-11202",
                "4f63ffc2-2473-42e9-bec4-0317ab3c70a7_Bedroom-40616",
                "711bd99b-2719-45a9-9836-c4baae874ad7_LivingDiningRoom-5053",
                "1ac5c316-3dc5-4ed2-8348-76715cfe4837_LivingDiningRoom-437092024",
                "320dc0a4-0acf-4880-97c5-33273d37bd93_LivingDiningRoom-241022024",
                "08bd64bf-132b-4659-a302-00d37dd74471_MasterBedroom-12183",
                "82eb76dd-8673-4f2b-b2bb-5e28d92ffb6c_SecondBedroom-48979",
                "67a4135c-c9d3-4c90-9c01-359d9c583e74_LivingDiningRoom-228272024",
                "7cf6e5de-a1ac-4bb4-bcb7-3b8dba3b4a78_LivingDiningRoom-13879",
                "ef0613b7-4461-461b-96c6-cb140f7a2f2a_LivingDiningRoom-16342",
                "af17e492-9278-4023-9c48-c289e819a641_LivingDiningRoom-4522024",
                "bcb18073-a4a7-4ce4-b24b-f67877bde572_LivingDiningRoom-23320",
                "081b8907-7e4d-4764-b931-7709386de7da_LivingDiningRoom-202292024",
                "89630235-284d-49c1-8258-65af4e749633_LivingDiningRoom-825",
                "ae1f318b-a8ac-4eec-974c-10a587ea0a71_LivingDiningRoom-36167",
                "22310ce9-9de0-4bd5-885a-fd54e0c39b46_LivingDiningRoom-5198",
                "2be2628f-bec8-4217-9660-805b1c8a1baa_LivingDiningRoom-142185",
                "33d1a77a-47d4-49ba-b682-61a47eff9c3c_LivingDiningRoom-25022024",
                "18c98959-cef7-4f72-b95e-387eba4e86b8_LivingDiningRoom-8855",
                "09170637-2952-4014-b97b-9e3f7d519c66_LivingDiningRoom-344202024",
                "0abea0c8-3398-4d26-b03d-9fb1fc9708a4_LivingDiningRoom-212696",
                "9e9c7c48-a631-4ded-a271-7dd697faf0dc_LivingDiningRoom-16322024",
                "f105dd34-bc81-432e-83bd-314087015257_LivingDiningRoom-1692024",
                "ad2462d1-ed66-45c6-8739-3ccface000c9_LivingDiningRoom-5751",
                "b73ddfc0-9382-4e47-91d1-0a86a69fcbcc_LivingDiningRoom-245462024",
                "022b94eb-7d48-4312-af06-d05a7cbd5c64_LivingDiningRoom-12319",
                "0bd1cdea-366c-47b6-a65d-ee7ccde0aaa9_LivingDiningRoom-23474",
                "08f96e2f-fbb4-4850-88e6-239e998e6f6a_LivingDiningRoom-91422024",
                "2d728150-ecd3-4648-b584-68de8e9f9a89_Library-118121",
                "388a3f2f-1cf9-40f7-8dbe-5452598359dd_LivingDiningRoom-17211",
                "3a0683d2-399c-47fa-99f2-cbb3bff5e156_LivingDiningRoom-12394",
                "339e4d0f-096a-4453-be12-5a730e71e6aa_LivingDiningRoom-15731",
                "e1b50a25-5422-4307-8b9c-24ad0bb8d8bc_LivingDiningRoom-25562",
                "2f5fc1b6-c9d2-4388-a8ed-b2144f30781f_LivingDiningRoom-57454",
                "1d43e076-fc80-4d55-a07d-1b7a8c6dc6e7_LivingDiningRoom-3814",
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
        self.s4s = [2,3,4]
        self.rates =[ 0.1*i for i in range(1,2)] #4)] #
        self.devs = [ 0.5*i for i in range(1,6)] #(1,6)] #
        self.hypers = [[[ (rate,s4,dev) for dev in self.devs ] for rate in self.rates] for s4 in self.s4s ]
        self.exops = [[[ None for dev in self.devs ] for rate in self.rates]  for s4 in self.s4s ]
        
    def __call__(self):
        if self.task==2: #2:experiment, 1:process and visualize, 0:load and visualize
            for s,s4 in enumerate(self.s4s):
                for r,rate in enumerate(self.rates):
                    for d,dev in enumerate(self.devs):
                        from ..Operation.Optm import default_optm_config as config
                        config["pat"]["rate"] = {"mode":"exp_dn","r0":rate*9,"lda":0.2,"rinf":rate*4}#{"mode":"exp_dn","r0":rate*2,"lda":0.5,"rinf":rate/5.0}
                        config["phy"]["rate"] = {"mode":"exp_up","rinf":rate*10,"lda":0.5,"r0":rate/100.0}#{"mode":"exp_up","rinf":rate*10,"lda":1.5,"r0":rate/100.0}
                        config["phy"]["s4"] = s4
                        dirName = "%.2f %.3f %d"%(dev,rate,s4)
                        Exop = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,num=self.num,expName=self.expName,dirName=dirName,dev=dev,config=config,task=self.task)
                        Exop()

        self.visualize()
            
    def load(self):
        for s,s4 in enumerate(self.s4s):
            for r,rate in enumerate(self.rates):
                for d,dev in enumerate(self.devs):
                    dirName = "%.2f %.3f %d"%(dev,rate,s4)
                    self.exops[s][r][d] = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,num=self.num,expName=self.expName,dirName=dirName,dev=dev,config=None,task=-1)
                    self.exops[s][r][d].load(n="plots")
    
    def visualize(self):
        self.load()
        assert len(self.rates)==1
        from matplotlib import pyplot as plt
        import os
        for r,rate in enumerate(self.rates):
            for key in ["time","steps","vio","fit","adjs","cos","diff"]:
                for s,s4 in enumerate(self.s4s):
                    for d,dev in enumerate(self.devs):
                        self.exops[s][r][d].visualize(keys=[key])
                    plt.legend(["dev=%.3f"%(dev) for dev in self.devs])
                    plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s s4=%d.png"%(key,s4)))
                    plt.clf()

                for d,dev in enumerate(self.devs):
                    for s,s4 in enumerate(self.s4s):
                        self.exops[s][r][d].visualize(keys=[key])
                    plt.legend(["s4=%d"%(s4) for s4 in self.s4s])
                    plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s dev=%.3f.png"%(key,dev)))
                    plt.clf()        