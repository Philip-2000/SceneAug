#Design all the experiments about scene recognization, modification and generation
#The process is defined by methods in the object (representing the experiment), and the data structures are variables

#but seting up the experiment by instantiating the object and calling its functions is the ../script/expn.py 's work.
from .Patn import *
from itertools import chain
from moviepy.editor import ImageSequenceClip
EXP_IMG_BASE_DIR = "../experiment/"
class ma():
    def __getitem__(self,a):
        return "Room"
    
class expn():
    def __init__(self,pmVersion,expName,mt,roomMapping={},loadDataset=True):
        self.pm = patternManager(loadDataset=loadDataset)
        self.pm.treeConstruction(load=pmVersion)
        self.sDs = self.pm.sDs
        self.devs = [0.0]
        self.ld = len(self.devs)
        self.roomMapping = { #roomMapping
            "Bedroom":"Bedroom", "MasterBedroom":"Bedroom", "SecondBedroom":"Bedroom", "KidsRoom":"Bedroom", "ElderlyRoom":"Bedroom",
            "LivingRoom":"LivingRoom", "DiningRoom":"DiningRoom", "LivingDiningRoom":"LivingDiningRoom", "Library":"Library"}
        self.rooms = list(set(self.roomMapping.values())) if self.roomMapping["Bedroom"] != "Room" else ["Room"]

        self.result = [[ [] for j in self.rooms ] for i in self.ld]
        
        self.imgDir = os.join(EXP_IMG_BASE_DIR,pmVersion,expName)
        self.mt = mt
        self.mtl = len(self.mt)

    def randomize(self, s, dev):
        t = deepcopy(s)
        for o in t.OBJES:
            o.translation += np.randn((3)) * dev
            o.size += np.randn((3)) * dev * 0.1
            o.orientation += np.randn((1)) * dev
        return t

    def run(self):
        pass

    def store(self,id,roomType,res):
        self.result[id][self.rooms.index(self.roomMapping[roomType])].append(res)

    def visualSingle(self,data,figName=""):
        #data.shape = len(self.dev) = 10? : len(self.roomTypes) = 4 and num = ??? : len(xTitle) = 2/1
        fig, ax1 = plt.subplots()

        metricsExp = np.average(data.reshape((-1,self.mtl)),axis=0)
        metricsDev = (np.average((data.reshape((-1,self.mtl)) - metricsExp.reshape((1,-1)))**2,axis=0))**0.5
        metricsStandard = [[metricsExp[j],metricsDev[j]] for j in range(self.mtl)]

        colors = ["khaki","paleturquoise","plum","tomato", "springgreen"]
        scl=1
        for i in range(len(self.devs)):
            for j in range(len(self.mt)):
                st = metricsStandard[j]
                A = ax1.boxplot((data[i:i+1,:,j]-st[0])/st[1], positions=[(i*(self.mtl+2)+j)*scl], labels = [metricsTitles[j]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[j]})
                B = {"medians":A["medians"][0].get_data()[1][0],"means":A["means"][0].get_data()[1][0],"capsUp":A["caps"][0].get_data()[1][0],"capsDown":A["caps"][1].get_data()[1][0],"x":A["means"][0].get_data()[0][0]}
                ax1.text(B["x"]+0.1,B["means"],"%.2f"%((B["means"]-st[0])*st[1]+st[0]),fontsize='xx-small')
                ax1.text(B["x"]+0.1,B["capsUp"]-0.05,"%.2f"%((B["capsUp"]-st[0])*st[1]+st[0]),fontsize='xx-small')
                ax1.text(B["x"]+0.1,B["capsDown"]+0.05,"%.2f"%((B["capsDown"]-st[0])*st[1]+st[0]),fontsize='xx-small')
        
        ax1.set_xticks([(i*(self.mtl+2)+(self.mtl-1)/2.0)*scl for i in range(len(self.devs))], ["%.3f"%(d) for d in self.devs])
        ax1.set_ylabel("")
        ax1.set_yticklabels("")
        
        if self.mtl>1:
            ax1.legend(handles=[plt.Line2D([0],[0],color=colors[i],lw=5,label=self.mt[i]) for i in range(self.mtl)], labels=self.mt, ncol=self.mtl) #, labelcolor=colors, loc='lower left'
        plt.savefig(os.join(self.imgDir,figName+".png"))
        plt.clf()

    def visualize(self):
        #roomSum = sum(roomCnt)
        lens = [len(j) for j in self.result[0]]
        locs = [sum(lens[:r+1]) for r in range(-1,len(self.rooms))]

        data = np.array([chain(*(self.result[l])) for l in range(self.ld) ])
        self.visualSingle(data,figName="Overall")
        for r in range(len(locs)-1):
            if len(locs)>2:
                self.visualSingle(data[:,locs[r]:locs[r+1],:],figName=self.rooms[r])

    def proc(self, procs):
        pass

    def show(self, cnt):
        while cnt:
            Id = np.random.randint(len(self.sDs))
            s = self.sDs[Id]
            thisDir = os.join(self.imgDir,"show",s.scene_uid)
            rands = deepcopy(s)
            d = 0
            for dev in ([0]+self.devs):
                rands = self.randomize(rands,dev-d)
                rands.draw(figName = os.join(thisDir,"%.3f rand.png"%(dev))) #where???
                procs = deepcopy(rands)
                self.proc(procs)
                procs.draw(figName = os.join(thisDir,"%.3f proc.png"%(dev))) #where???
                d = dev
            ImageSequenceClip([os.path.join(thisDir,f) for f in os.listdir(thisDir) if f.endswith('rand.png')]).write_videofile(os.path.join(thisDir,self.__class__.__name__+" rand.mp4"), fps=5)
            ImageSequenceClip([os.path.join(thisDir,f) for f in os.listdir(thisDir) if f.endswith('proc.png')]).write_videofile(os.path.join(thisDir,self.__class__.__name__+" proc.mp4"), fps=5)
            
            cnt-=1
                
        return

class RecExpn(expn):
    def __init__(self,pmVersion):
        super(RecExpn,self).__init__(pmVersion,self.__class__.__name__,["fitness","assigned"])
    
    def run(self):
        for s in self.sDs:
            fit,n = s.recognize(self.pm)
            for dev in self.devs:
                rands,_ = self.randomize(s,dev)
                fitr,nr = rands.recognize(self.pm)
                self.store(self.devs.index(dev),s.roomType,[fitr/fit,nr/n])
        self.visualize()

    def proc(self, procs):
        procs.recognize(self.pm)


class OptExpn(expn):
    def __init__(self,pmVersion):
        super(OptExpn,self).__init__(pmVersion,self.__class__.__name__,["modify"])
    
    def loss(self,ope,noise):
        return ope-noise

    def run(self):
        for s in self.sDs:
            operate = s.optimize(self.pm)
            for dev in self.devs:
                rands,noise = self.randomize(s,dev)
                operater = rands.optimize(self.pm)
                
                loss = self.loss((operater - operate),noise)
                
                self.store(self.devs.index(dev),s.roomType,[loss])
        self.visualize()    
        pass

    def proc(self, procs):
        procs.optimize(self.pm)

class GenExpn(expn):
    def __init__(self,pmVersion):
        super(GenExpn,self).__init__(pmVersion,loadDataset=False)
        pass
    
    def run(self):
        #generating?
        #but the validation can not be used here?
        #we can design some experiments for
        pass