EXOP_BASE_DIR = "./experiment/opts/"
class exop():
    def __init__(self,pmVersion,dataset,UIDS,expName,mod,dev,config,run):
        
        from ..Basic.Scne import scneDs as SDS
        
        self.sDs = SDS(dataset,lst=UIDS,grp=False, cen=False, wl=True, windoor=True)
        
        self.pmVersion = pmVersion
        self.dev = dev
        self.config = config
        self.mod = mod
        self.res = {s.scene_uid[:10]:[] for s in self.sDs}#"lots of items here, but only the original results, (which scene) : (which step) : [(1)time]"
        self.run = run #if we implement this experiment or only visualize it
        self.name= expName
        import os
        os.makedirs(os.path.join(EXOP_BASE_DIR,expName),exist_ok=True)
        self.filename = os.path.join(EXOP_BASE_DIR,expName,"%s %.2f %.3f %d.json"%(mod,dev,config["phy"]["rate"],config["phy"]["s4"]))
        
    def __randomize(self,t):
        import numpy as np
        for o in t.OBJES:
            from ..Operation.Adjs import adj
            dT = np.random.randn((3)) * self.dev
            dS = np.random.randn((3)) * self.dev * 0.1
            dR = np.random.randn((1)) * self.dev
            o.adjust = adj(T=dT,S=dS,R=dR,o=o)
            o.adjust()
        from ..Operation.Adjs import adjs
        return adjs(t.OBJES)

    def __call__(self):
        if self.run:
            from ..Operation.Optm import optm
            for s in self.sDs:

                adjs0 = self.__randomize(s)

                OP = optm(pmVersion=self.pmVersion,scene=s,PatFlag=True,PhyFlag=True,config=self.config,exp=True)
            
                
                ret,step,time = {"over":False},0,0
                while (not ret["over"]): #the over criterion
                    ret = OP(step) #timer, adjs, vio, fit, cos, over
                    ret["diff"] = adjs0 - ret["adjs"]
                    self.store(ret,s,step)
                    step += 1
                    time += ret["timer"]["all"]
                    break

                print(step,time)
            self.save()
        self.visualize()

    def store(self,ret,s,steps):
        assert steps == len(self.res[s.scene_uid[:10]])
        ret["timer"]= ret["timer"].dct()
        ret["adjs"] = ret["adjs"].dct()
        self.res[s.scene_uid[:10]].append(ret)

    def save(self):
        import json
        open(self.filename,"w").write(json.dumps(self.res))

    def load(self):
        import json
        self.res = json.load(open(self.filename,"r"))
    
    def visualize(self):
        self.load()
        print(self.res)
        raise NotImplementedError
        #综合各次实验的收敛过程：可以获得多种统计结果：
        #（1）violate箱线图（这些箱线图，理论上说是需要每一步都有的，但是步数搞不好会特别的多，所以需要多步并作一步）
        #（2）fit评分箱线图
        #（3）adjust幅度箱线图？
        #（4）adjust趋势和noise的趋势是否一致，这个是否一致可以计算一个cos值我觉得，然后将在1到-1之间画一下这些cos值，发现基本上都在-1附近，代表着我们的调整是逆着噪声进行的
        #（5）pat操作的adjust趋势和phy操作的adjust趋势是否一致，我觉得其实可以类似出图？

class exops():
    def __init__(self,pmVersion='losy',dataset="../novel3DFront/",run=True,UIDS=[]):

        #还有一个十分关键的问题，就是不同的超参数的实验过程需要分别进行的问题
        
        #那么我们的exops类就需要感知到各个超参数的值，对每一组超参数值分别组织实验，对实验的结果分别整合和对齐
        #甚至是不同超参数之间的各个实验结果数值计算比值，
        #这个过程为什么在ExPn中没有出现呢？这是因为，ExPn中其实并没有什么待测试的超参数
        #所以纵观全局会觉得，exop的代码会比expn的代码工程要复杂得多

        #那么有哪些超参数有待测试呢？
        #第一，识别模式，先识别、后识别、全识别
            #最傻逼的是，识别算法需要优化哈哈哈哈哈哈哈死了得了，但是这个事情其实不影响调试

        #第二，学习率

        #第三，采样率？

        #第零，场景打乱的方差

        #暂时，这个超参数矩阵的尺度是（1，1，1，1）
        #矩阵的数值必须放在外面吗？也不一定吧
        
        self.UIDS = UIDS
        if len(UIDS)==0:
            self.UIDS = ["0acdfc7d-6f8f-4f27-a1dd-e4180759caf5_LivingDiningRoom-41487"]
        self.run=run
        self.pmVersion=pmVersion
        self.dataset=dataset

        self.hypers = []
        for mod in ["prerec"]:#,"postrec","rerec"]:#
            for rate in [0.5*i for i in range(1,2)]:
                for s4 in range(2,3):
                    for dev in [0.05*i for i in range(1,2)]:
                        self.hypers.append((mod,rate,s4,dev))
                        

    def __call__(self):
        for h in self.hypers:
            mod,rate,s4,dev = h[0],h[1],h[2],h[3]
            config = {
                "pat":{
                    "rerec":bool(mod=="rerec"),
                    "prerec":bool(mod=="prerec"),
                    "rand":False,
                    "rate":rate
                },
                "phy":{
                    "rate":rate,
                    "s4": s4,
                    "door":{"expand":1.1,"out":0.2,"in":1.0,},
                    "wall":{"bound":0.5,},
                    "object":{
                        "Pendant Lamp":[.0,.01,.01],#
                        "Ceiling Lamp":[.0,.01,.01],#
                        "Bookcase / jewelry Armoire":[.2,1., .9],#
                        "Round End Table":[.0,.5, .5],#
                        "Dining Table":[.0,.5, .5],#
                        "Sideboard / Side Cabinet / Console table":[.0,.9, .9],#
                        "Corner/Side Table":[.0,.9, .9],#
                        "Desk":[.0,.9, .9],#
                        "Coffee Table":[.0,1.,1.1],#
                        "Dressing Table":[.0,.9, .9],#
                        "Children Cabinet":[.2,1., .9],#
                        "Drawer Chest / Corner cabinet":[.2,1., .9],#
                        "Shelf":[.2,1., .9],#
                        "Wine Cabinet":[.2,1., .9],#
                        "Lounge Chair / Cafe Chair / Office Chair":[.0,.5, .5],#
                        "Classic Chinese Chair":[.0,.5, .5],#
                        "Dressing Chair":[.0,.5, .5],#
                        "Dining Chair":[.0,.5, .5],#
                        "armchair":[.0,.5, .5],#
                        "Barstool":[.0,.5, .5],#
                        "Footstool / Sofastool / Bed End Stool / Stool":[.0,.5, .5],#
                        "Three-seat / Multi-seat Sofa":[.2,1., .9],#
                        "Loveseat Sofa":[.2,1., .9],#
                        "L-shaped Sofa":[.0,.6, .9],#
                        "Lazy Sofa":[.2,1., .9],#
                        "Chaise Longue Sofa":[.2,1., .9],#
                        "Wardrobe":[.2,1., .9],#
                        "TV Stand":[.2,1., .9],#
                        "Nightstand":[.0,.5, .5],#
                        "King-size Bed":[.2,1.,1.2],#
                        "Kids Bed":[.2,1.,1.2],#
                        "Bunk Bed":[.2,1.,1.2],#
                        "Single bed":[.2,1.,1.2],#
                        "Bed Frame":[.2,1.,1.2],#
                    },
                    "syn":{"T":1.0,"S":0.01,"R":1.0,},
                }
            }
            Exop = exop(pmVersion=self.pmVersion,dataset=self.dataset,UIDS=self.UIDS,expName="test",mod=mod,dev=dev,config=config,run=self.run)
            Exop()
            
        pass

    def load(self):
        pass
    
    def visualize(self):
        self.load()
        #这里的visualize函数就实际上更加有趣了，因为他们展现的是超参数不同数值之间比较的内容，
        #那么exop中的箱线图就已经暂时很难再进行展示了，因为他们的我们没有那么多空间来展示了
        #我们考虑将箱线改成均值，然后展示一个x轴为时间或步数，z轴为初始dev的曲面

        #具体地获得：
        #（1）violate曲面图
        #（2）fit曲面图
        #（3）adjust曲面图
        #（4）adjust趋势和noise的趋势是否一致，我觉得还是，这个东西可以搞一个曲面图
        #（5）pat操作的adjust趋势和phy操作的adjust趋势是否一致，这个东西可以搞一个曲面图，虽然意义不大，但我觉得代码上应该区别不大，所以搞一下吧
        #（6）实验进行肘方图+收敛步数直方图（高斯波包），这个（6）特色内容，在exop中没有的


# class expn():
#     def __init__(self,pmVersion,dataset,UIDS,expName,num,mt,task,roomMapping={}):
#     def randomize(self, s, dev, hint=None):
#     def save(self):
#     def load(self):
#     def run(self, **kwargs):
#     def store(self,id,roomType,res):
#     def visualSingle(self,data,figName=""):
#     def visualize(self):
#     def execute(self,scene,**kwargs):
