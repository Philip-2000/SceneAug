EXOP_BASE_DIR = "./experiment/opts/"
DEBUG_FILTER = []#"","+"] #
class exop():
    def __init__(self,pmVersion,dataset,UIDS,num,expName,dirName,dev,config,task):
        self.task = task #2:experiment, 1:process and visualize, 0:load and visualize, -1:load and visualized the exops
        if task == 2:
            from ...Basic import scneDs as SDS
            self.sDs = SDS(dataset,lst=UIDS,num=num,grp=False, cen=False, wl=True, windoor=True)
            self.pmVersion = pmVersion
            config["phy"]["vis"], config["pat"]["vis"], config["vis"] = {},{},{}
            self.config = config
        self.dev = dev
        import os
        self.dirname,self.seeds,self.S = os.path.join(EXOP_BASE_DIR,expName,dirName), os.path.join(EXOP_BASE_DIR,expName,dirName,"seeds"),True
        #print(self.dirname)
        os.makedirs(self.dirname,exist_ok=True) #
        os.makedirs(self.seeds,exist_ok=True) #

        from .RsOp import rsops
        self.res = rsops(UIDS,self.dirname)
        #self.origin = {s.scene_uid[:10]:[] for s in self.sDs}#the original results
        #self.plots = {"fit":{},"vio":{},"adjs":{},"diff":{},"time":{},"steps":{}} #"the data for plots"
    
    def __call__(self):
        if self.task==2:            #2:experiment, 
            self.experiment()
            self.res.save("origin")
        if self.task>=1:            #1:process the original data from experiment to data points on plots
            self.res.load("origin")
            self.res.to_plot()
            self.res.save("plot")
        if self.task>=0:            #0:load the plot data and visualize with plots
            self.res.load("plot")
            self.res.visualizes()
        if self.task>=-1:           #-1:load the plot data and wait for exops to visualize is
            self.res.load("plot")

    def experiment(self):
        from ...Operation import optm, rgnz, patternManager as pm
        self.pm = pm(self.pmVersion)
        for i,s in enumerate(self.sDs):
            rgnz(s,self.pm,v=0).recognize(use=True,draw=False,show=False)
            
            adjs0, s_key = self.__randomize(s), self.res.connect(s.scene_uid,self.dirname)
            while True:
                #self.init_eval(s)
                OP = optm(pm=self.pm,scene=s,config=self.config,exp=True,timer=self.res[s_key].timer)
                self.debugdraw(s,0,"")
                ret,step = {"over":False},0
                [o.adjust.clear() for o in s.OBJES]
                self.res[s_key].append(step=0,vio=OP.PhyOpt.eval(), fit=OP.PatOpt.eval())
                self.res[s_key].timer("",0) #start stamp
                while (ret["over"] is False) and step <= OP.max_len: #the over criterion
                    assert ret["over"] is False and ret["over"] is not None
                    ret = OP.exps(step)#,self.debugdraw) #timer, adjs, vio, fit, cos, over
                    self.res[s_key].append(step=step+1,dif=adjs0-ret["adj" ],**ret)
                    self.debugdraw(s,step,"+")
                    step += 1
                if ret["over"] is not None and step <= OP.max_len:
                    break
                else:
                    print("restart",s.scene_uid)
                    print("adj",self.res.rsops[s_key].adj[-1],"vio",self.res.rsops[s_key].vio[-1],"fit",self.res.rsops[s_key].fit[-1])
                    adjs0 = self.__randomize(s,use=False)
                    self.res.rsops[s_key].clear()


            if i%18==17: print(i+1)

    def save(self,n="origin"):
        from moviepy.editor import ImageSequenceClip
        self.res.save(n) #open(os.path.join(self.dirname,n+".json"),"w").write(json.dumps(self.origin if n == "origin" else self.plots))
        for s in self.sDs:
            _ = ImageSequenceClip([os.path.join(EXOP_BASE_DIR,"debug","%s-%d%s.png"%(s.scene_uid[:10],i,"+")) for i in range(1,33)],fps=3).write_videofile(os.path.join(EXOP_BASE_DIR,"%s.mp4"%(s.scene_uid[:10])),logger=None) if "+" in DEBUG_FILTER and n=="origin" else None
    #region: util
    def debugdraw(self,s,step,suffix):
        s.draw(imageTitle=EXOP_BASE_DIR+"debug/%s-%d%s.png"%(s.scene_uid[:10],step,suffix)) if suffix in DEBUG_FILTER else None#return #
        
    def __randomize(self,t,use=True):
        import numpy as np,os
        #a,b = self.scene.randomize(dev=rand,cen=True,hint=np.load(os.path.join(self.scene.imgDir,"rand.npy")))#None)#
        if use and self.S and os.path.exists(os.path.join(self.seeds,t.scene_uid+".npy")):
            a,b = t.randomize(dev=self.dev,cen=True,hint=np.load(os.path.join(self.seeds,t.scene_uid+".npy")))#None)#
        else:
            a,b = t.randomize(dev=self.dev,cen=True)
        np.save(os.path.join(self.seeds,t.scene_uid+".npy"), b)
        return a
    #endregion: util
class exops():
    def __init__(self,pmVersion='losy',dataset="../novel3DFront/",task=2,expName="no",UIDS=[]):
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
                "d03b14a0-31a6-4136-8e7c-d364c7b09955_SecondBedroom-1613",
                "f1a605ec-155d-48fd-ac89-d54c3ebdbfb6_LivingDiningRoom-3581",
                "d0615bd5-c5af-474b-a2bf-c434fcfaf74c_MasterBedroom-110",
                "11410829-e920-4e53-b8ab-688d026c0af1_Bedroom-2297",
                "0b812f6e-0769-4b37-b872-ee066433207e_MasterBedroom-11619",
                "5530455f-bf14-464b-88c0-7eaee2d2d85b_LivingDiningRoom-3064",
                "0685adc2-121b-4ea7-894d-6a81cb779be9_LivingDiningRoom-10913",
                "0b35b16b-f277-427a-8274-09c50f6e2b99_LivingDiningRoom-7940",
                "12219573-1263-4658-a5db-d048f3f0d668_MasterBedroom-569",
                "908633f6-545c-4016-a2e7-f334120ac392_MasterBedroom-7728",
                "196930e3-70ae-4432-bf23-7a706f5feb22_LivingDiningRoom-109282024",
                "0d3a11a7-ea64-4ba0-a78a-85719956d7a7_MasterBedroom-59476",
                "1b8e0aa0-1feb-4030-b45b-ba797530c57f_LivingDiningRoom-36888",
                "6e4825e3-4ceb-40d4-a7f8-2adf7a3ab6f5_MasterBedroom-536",
                "822213ac-e850-4189-ae92-5a9e0946be04_Bedroom-72334",
                "00d15904-be14-4958-b67b-5a0137a279fa_SecondBedroom-1070",
                "967e40cb-ac87-4ff0-90f0-f775d80ca4ae_LivingDiningRoom-1110",
                "1b6ad955-6e6e-4a87-9a46-ddfd3bd91a50_LivingDiningRoom-616382024",
                "6e3d565d-89ce-4a2d-b39b-aef362e9857e_MasterBedroom-4091",
                "3d048fd0-a931-4343-bd40-0f62860034a1_Bedroom-11202",
                "4f63ffc2-2473-42e9-bec4-0317ab3c70a7_Bedroom-40616",
                "ef0613b7-4461-461b-96c6-cb140f7a2f2a_Bedroom-15208",
                "1ac5c316-3dc5-4ed2-8348-76715cfe4837_LivingDiningRoom-437092024",
                "320dc0a4-0acf-4880-97c5-33273d37bd93_LivingDiningRoom-241022024",
                "08bd64bf-132b-4659-a302-00d37dd74471_MasterBedroom-12183",
                "82eb76dd-8673-4f2b-b2bb-5e28d92ffb6c_SecondBedroom-48979",
                "67a4135c-c9d3-4c90-9c01-359d9c583e74_LivingDiningRoom-228272024",
                "7cf6e5de-a1ac-4bb4-bcb7-3b8dba3b4a78_LivingDiningRoom-13879",
                "ef0613b7-4461-461b-96c6-cb140f7a2f2a_LivingDiningRoom-16342",
                "af17e492-9278-4023-9c48-c289e819a641_LivingDiningRoom-4522024",
                "cc5f69b9-c76d-41a9-8a40-094e5664ebbd_MasterBedroom-3715",
                "081b8907-7e4d-4764-b931-7709386de7da_LivingDiningRoom-202292024",
                "89630235-284d-49c1-8258-65af4e749633_LivingDiningRoom-825",
                "ae1f318b-a8ac-4eec-974c-10a587ea0a71_LivingDiningRoom-36167",
                "22310ce9-9de0-4bd5-885a-fd54e0c39b46_LivingDiningRoom-5198",
                "2be2628f-bec8-4217-9660-805b1c8a1baa_LivingDiningRoom-142185",
                "33d1a77a-47d4-49ba-b682-61a47eff9c3c_LivingDiningRoom-25022024",
                "18c98959-cef7-4f72-b95e-387eba4e86b8_LivingDiningRoom-8855",
                "09170637-2952-4014-b97b-9e3f7d519c66_LivingDiningRoom-344202024",
                "0abea0c8-3398-4d26-b03d-9fb1fc9708a4_LivingDiningRoom-212696",
                "2d728150-ecd3-4648-b584-68de8e9f9a89_SecondBedroom-119762",
                "f105dd34-bc81-432e-83bd-314087015257_LivingDiningRoom-1692024",
                "ad2462d1-ed66-45c6-8739-3ccface000c9_LivingDiningRoom-5751",
                "ca05c8ca-dcd6-402c-89f8-27ba4e769841_MasterBedroom-412",
                "022b94eb-7d48-4312-af06-d05a7cbd5c64_LivingDiningRoom-12319",
                "0bd1cdea-366c-47b6-a65d-ee7ccde0aaa9_LivingDiningRoom-23474",
                "08f96e2f-fbb4-4850-88e6-239e998e6f6a_LivingDiningRoom-91422024",
                "2d728150-ecd3-4648-b584-68de8e9f9a89_Library-118121",
                "3a16f523-4dc8-4a6e-a57b-497fe0befa0e_KidsRoom-5204",
                "3a0683d2-399c-47fa-99f2-cbb3bff5e156_LivingDiningRoom-12394",
                "339e4d0f-096a-4453-be12-5a730e71e6aa_LivingDiningRoom-15731",
                "3d506c20-7b7e-4757-8884-1fd5525eb63a_Bedroom-1298",
                "1dcb6909-4e3c-4e34-bb38-207cc78e9263_Bedroom-20397",
                "1d43e076-fc80-4d55-a07d-1b7a8c6dc6e7_LivingDiningRoom-3814",
            ]
        self.num=0 #self.num=512, self.UIDS = []
        self.task = task #2:experiment, 1:process and visualize, 0:load and visualize，-1:load and visualized the exops
        self.pmVersion=pmVersion
        self.dataset=dataset
        self.expName = expName
        import os
        os.makedirs(os.path.join(EXOP_BASE_DIR,expName),exist_ok=True)

        self.hypers = []
        self.mods = ["prerec",]#"postrec",
        self.s4s = [2,3,4]#]# 
        self.devs = [ 0.5*i for i in range(2,7)] #(2,3)] #
        self.hypers = [[ (s4,dev) for dev in self.devs ]  for s4 in self.s4s ]
        self.EXOPS = [[ None for dev in self.devs ]  for s4 in self.s4s ]
        self.RSOPS_dev = [None for _ in self.devs]
        self.RSOPS_s4s = [None for _ in self.s4s ]
        
    def __call__(self):
        for s,s4 in enumerate(self.s4s):
            for d,dev in enumerate(self.devs):
                from ...Operation import default_optm_config as config
                config["phy"]["s4"] = s4
                dirName = "%.2f %d"%(dev,s4)
                self.EXOPS[s][d] = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,num=self.num,expName=self.expName,dirName=dirName,dev=dev,config=config,task=self.task)
                self.EXOPS[s][d]()
        
        from .RsOp import rsops
        import os
        for d,dev in enumerate(self.devs):
            os.makedirs(os.path.join(EXOP_BASE_DIR,self.expName,"%.2f"%(dev)),exist_ok=True)
            self.RSOPS_dev[d] = rsops(self.UIDS,os.path.join(EXOP_BASE_DIR,self.expName,"%.2f"%(dev)), [ os.path.join(EXOP_BASE_DIR,self.expName,"%.2f %d"%(dev,s)) for s in self.s4s ])
            if self.task>=1:
                self.RSOPS_dev[d].load("origin")
                self.RSOPS_dev[d].to_plot()
                self.RSOPS_dev[d].save("plot")
            elif self.task>=0:
                self.RSOPS_dev[d].load("plot")
                self.RSOPS_dev[d].visualizes()
            elif self.task>=-1:
                self.RSOPS_dev[d].load("plot")

        for s,s4 in enumerate(self.s4s):
            os.makedirs(os.path.join(EXOP_BASE_DIR,self.expName,"%d"%(s4)),exist_ok=True)
            self.RSOPS_s4s[s] = rsops(self.UIDS,os.path.join(EXOP_BASE_DIR,self.expName,"%d"%(s4)), [ os.path.join(EXOP_BASE_DIR,self.expName,"%.2f %d"%(dev,s4)) for dev in self.devs ])
            if self.task>=1:
                self.RSOPS_s4s[s].load("origin")
                self.RSOPS_s4s[s].to_plot()
                self.RSOPS_s4s[s].save("plot")
            elif self.task>=0:
                self.RSOPS_s4s[s].load("plot")
                self.RSOPS_s4s[s].visualizes()
            elif self.task>=-1:
                self.RSOPS_s4s[s].load("plot")

        self.visualize()
    
    def visualize(self):
        import os, matplotlib.pyplot as plt
        for key in ["times","steps","vio","fit","adj","dif"]:
            kwargs = {"key":key,"subkey":"line" if key in ["times","steps"] else "time"}
            for s,s4 in enumerate(self.s4s):
                for d,dev in enumerate(self.devs):
                    self.EXOPS[s][d].res.line(clear=False,**kwargs)
                plt.legend(["dev=%.3f"%(dev) for dev in self.devs])
                plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s s4=%d.png"%(key,s4)))
                plt.clf()
            
            for s,s4 in enumerate(self.s4s):
                self.RSOPS_s4s[s].line(clear=False,**kwargs)
            plt.legend(["M=%d"%(s4*4) for s4 in self.s4s])
            plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s s4.png"%(key)))
            plt.clf()

            for d,dev in enumerate(self.devs):
                for s,s4 in enumerate(self.s4s):
                    self.EXOPS[s][d].res.line(clear=False,**kwargs)
                plt.legend(["M=%d"%(s4*4) for s4 in self.s4s])
                plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s dev=%.3f.png"%(key,dev)))
                plt.clf()

            for d,dev in enumerate(self.devs):
                self.RSOPS_dev[d].line(clear=False,**kwargs)
            plt.legend(["dev=%.3f"%(dev) for dev in self.devs])
            plt.savefig(os.path.join(EXOP_BASE_DIR,self.expName,"%s dev.png"%(key)))
            plt.clf()
        