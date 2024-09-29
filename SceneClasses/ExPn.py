from .Patn import *

class expn():
    def __init__(self,pmVersion,loadDataset,imgDir):
        self.pm = patternManager(loadDataset=loadDataset)
        self.pm.treeConstruction(load=pmVersion)
        self.sDs = self.pm.sDs
        self.devs = [0.0]
        self.ld = len(self.devs)
        self.roomMapping = {}
        self.rooms = ["Bedroom","LivingRoom","DiningRoom","LivingDiningRoom"]
        self.result = {}
        self.imgDir = imgDir
        self.mt = []
        pass

    def randomize(self, s, dev):
        return s

    def run(self):
        pass

    def visualSingle(self,data,metricsTitles,figName=""):
        #data.shape = len(self.dev) = 10? : len(self.roomTypes) = 4 and num = ??? : len(xTitle) = 2/1
        fig, ax1 = plt.subplots()

        metricsExp = np.average(data.reshape((-1,len(metricsTitles))),axis=0)
        metricsDev = (np.average((data.reshape((-1,len(metricsTitles))) - metricsExp.reshape((1,-1)))**2,axis=0))**0.5
        metricsStandard = [[metricsExp[j],metricsDev[j]] for j in range(len(metricsTitles))]

        colors = ["khaki","paleturquoise","plum","tomato", "springgreen"]
        scl=1
        for i in range(len(self.devs)):
            for j in range(len(metricsTitles)):
                st = metricsStandard[j]
                A = ax1.boxplot((data[i:i+1,:,j]-st[0])/st[1], positions=[(i*(len(metricsTitles)+2)+j)*scl], labels = [metricsTitles[j]], patch_artist=True, showmeans=True, boxprops={'facecolor': colors[j]})
                B = {"medians":A["medians"][0].get_data()[1][0],"means":A["means"][0].get_data()[1][0],"capsUp":A["caps"][0].get_data()[1][0],"capsDown":A["caps"][1].get_data()[1][0],"x":A["means"][0].get_data()[0][0]}
                ax1.text(B["x"]+0.1,B["means"],"%.2f"%((B["means"]-st[0])*st[1]+st[0]),fontsize='xx-small')
                ax1.text(B["x"]+0.1,B["capsUp"]-0.05,"%.2f"%((B["capsUp"]-st[0])*st[1]+st[0]),fontsize='xx-small')
                ax1.text(B["x"]+0.1,B["capsDown"]+0.05,"%.2f"%((B["capsDown"]-st[0])*st[1]+st[0]),fontsize='xx-small')
        
        ax1.set_xticks([(i*(len(metricsTitles)+2)+(len(metricsTitles)-1)/2.0)*scl for i in range(len(self.devs))], ["%.3f"%(d) for d in self.devs])
        ax1.set_ylabel("")
        ax1.set_yticklabels("")
        
        ax1.legend(handles=[plt.Line2D([0],[0],color=colors[i],lw=5,label=metricsTitles[i]) for i in range(len(metricsTitles))], labels=metricsTitles, ncol=len(metricsTitles)) #, labelcolor=colors, loc='lower left'
        plt.savefig(os.join(self.imgDir,figName+".png"))
        plt.clf()

    def visualBase(self,data,metricsTitles,locs):
        #roomSum = sum(roomCnt)
        self.visualSingle(data,metricsTitles,figName="Overall")
        for r in range(len(locs)-1):
            self.visualSingle(data[:,locs[r]:locs[r+1],:],metricsTitles,figName=self.roomTypes[r])

class RecExpn(expn):
    def __init__(self,pmVersion):
        super(RecExpn,self).__init__(pmVersion)
    
    def run(self):
        for s in self.sDs:
            fit = s.recognize(self.pm)
            for dev in self.devs:
                rands = self.randomize(s,dev)
                fitr = rands.recognize(self.pm)
            import sys
            sys.exit(0)
        pass

    def visualize(self):
        #something new to be parameters?
        #what?
        #type of room? 

        #organize data and fill to the visualbase()

        self.visualBase()




        
        
        #
        pass


class OptExpn(expn):
    def __init__(self,pmVersion):
        super(OptExpn,self).__init__(pmVersion)
        pass
    
    def run(self):
        for s in self.sDs:
            operate = s.optimize(self.pm)
            for dev in self.devs:
                rands = self.randomize(s,dev)
                operater = rands.optimize(self.pm)
            import sys
            sys.exit(0)    
        pass

    def visualize(self):
        #something new to be parameters?
        #what?
        #type of room? 

        #organize data and fill to the visualbase()

        self.visualBase()

class GenExpn(expn):
    def __init__(self,pmVersion):
        super(GenExpn,self).__init__(pmVersion,loadDataset=False)
        pass
    
    def run(self):
        #generating?
        #but the validation can not be used here?
        #we can design some experiments for
        pass