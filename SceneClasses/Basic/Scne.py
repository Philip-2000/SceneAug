import numpy as np

class scne():
    def __init__(self, scene, grp=False, windoor=False, wl=False, keepEmptyWL=False, cen=False, rmm=True, irt=16, imgDir="./"):#print(wl)print(keepEmptyWL)
        from .Obje import objes
        from .Wall import walls
        from ..Semantic.Grup import grup
        from ..Semantic.Spce import spces
        from copy import copy
        
        self.LINKS=[]
        self.copy = copy(scene)
        self.scene_uid = str(scene["scene_uid"]) if "scene_uid" in scene else ""
        rs = [r for r in ["Bedroom","KidsRoom", "ElderlyRoom", "LivingDiningRoom", "LivingRoom", "DiningRoom", "Library"] if self.scene_uid.find(r) > -1]
        self.roomType = "Room" if len(rs)==0 else rs[0]
        ce = scene["floor_plan_centroid"] if cen else np.array([0,0,0])
        c_e= np.array([0,0,0]) if (cen) or (not (wl or windoor)) else scene["floor_plan_centroid"]
        self.grp = grp
        self.imgDir = imgDir
        import os
        os.makedirs(self.imgDir,exist_ok=True)
        self.OBJES=objes(scene,ce,self)
        
        self.roomMask = scene["room_layout"][0] if rmm else np.zeros((64,64))
        self.rmm=rmm
        self.GRUPS=[]
        if grp:
            grops = np.ones(len(self.OBJES)) if scene["grops"] is None else scene["grops"]
            self.GRUPS = [grup([o.idx for o in self.OBJES if grops[o.idx]==j+1],{"sz":self.roomMask.shape[-1],"rt":irt},j+1,scne=self) for j in range(int(max(grops)))]
        
        self.WALLS = walls(scene["walls"] if wl else [], c_e, self, keepEmptyWL=keepEmptyWL, cont=scene["widos"] if windoor else [])
        self.text = scene["text"] if "text" in scene else ""
        self.SPCES = spces(self)#[]
        self.fild = None
        self.plan = None # a object: self.plan = ..Operation.Plan.plan(scene=self,pm=???)

    #region：in/outputs----------#

        #region: inputs----------#
    @classmethod
    def empty(cls,nm="",keepEmptyWL=False):
        return cls({"translations":[],"sizes":[],"angles":[],"class_labels":[],"scene_uid":nm},rmm=False,keepEmptyWL=keepEmptyWL)

    @classmethod
    def fromSceneJson(cls,sj):
        from .Obje import objes
        from .Wall import walls
        scene = cls.empty(keepEmptyWL=True)
        scene.scene_uid = sj["id"]
        rsj = sj["rooms"][0]
        scene.OBJES = objes.fromSceneJson(rsj,scene)
        scene.WALLS = walls.fromWallsJson(rsj["roomShape"],rsj["roomNorm"],scene, rsj["blockList"])
        return scene

    @classmethod
    def fromNpzs(cls,boxes=None,contours=None,conts=None,grops=None,load=True,name=None,dir="../novel3DFront/",**kwargs):
        import os
        if load:
            dn = dir+name
            boxes = np.load(dn+"/boxes.npz")
            contours = np.load(dn+"/contours.npz")["contour"] if os.path.exists(dn+"/contours.npz") else None
            conts = np.load(dn+"/conts.npz")["cont"] if os.path.exists(dn+"/conts.npz") else None
            grops = np.load(dn+"/group.npz")["group"] if os.path.exists(dn+"/group.npz") else None
        sceneDict = {"room_layout":np.zeros((1,64,64)).astype(np.uint8),"walls":contours,"widos":conts,"grops":grops,
            "translations":boxes["translations"],"sizes":boxes["sizes"],"angles":boxes["angles"],"class_labels":boxes["class_labels"],
            "floor_plan_centroid":boxes["floor_plan_centroid"],"scene_uid":boxes["scene_uid"]}
        kwargs["grp"] = (grops is not None) and ("grp" in kwargs and kwargs["grp"])
        kwargs["wl"] = (contours is not None) and ("wl" in kwargs and kwargs["wl"])
        kwargs["windoor"] = (conts is not None) and ("windoor" in kwargs and kwargs["windoor"])#print(kwargs)

        return cls(sceneDict,**kwargs)

    def addObject(self,objec):
        return self.OBJES.addObject(objec)

    def registerWalls(self,wls):
        for w in wls.WALLS:
            w.scne=self
        wls.scne=self
        self.WALLS = wls

    def registerObjes(self,obs):
        for o in obs.OBJES:
            o.scne=self
        obs.scne=self
        self.OBJES = obs

    def registerSpces(self,sps):
        for s in sps.SPCES:
            s.scne=self
        sps.scne=self
        self.SPCES = sps
        #endregion: inputs-------#

        #region: presentation----#
    def __str__(self):
        return str(self.OBJES)+'\n'+str(self.WALLS)

    def drao(self,suffix,config,ts,Js=None):#,lim=-1
        from matplotlib import pyplot as plt
        import os
        plt.figure(figsize=(50, 40))
        _ = self.fild.draw(suffix,config,ts) if suffix[:2]=="fi" else self.OBJES.drao(suffix, config)
        if suffix=="pat":
            from ..Semantic.Link import objLink
            self.LINKS = []
            for jss in Js:
                for oid,J in jss.items():
                    if oid != J: #print(oid,J)
                        self.LINKS.append(objLink(oid,J,len(self.LINKS),self,"black"))
            [li.draw() for li in self.LINKS]
        
        if suffix not in ["fih","fip","fiq"]: #these three images don't go through the matplotlib-2D pipeline, "fih" and "fiq" is from PIL, and "fip" is from pyplot-3d
            self.WALLS.draw()
            plt.axis('equal')
            plt.xlim(-7,7), plt.ylim(-7,7)
            plt.axis('off')
            plt.savefig(os.path.join(self.imgDir,suffix+"_"+str(ts)+".png"))
            plt.clf()
            plt.close()
        return os.path.join(self.imgDir,suffix+"_"+str(ts)+".png")

    def draw(self,imageTitle="",d=False,lim=-1,drawWall=True,drawUngroups=False,drawRoomMask=False,classText=True,suffix="_Layout"):
        from matplotlib import pyplot as plt
        import os
        plt.figure(figsize=(10, 8))
        self.SPCES.draw(dr=False)

        [self.GRUPS[i].draw() for i in range(len(self.GRUPS)) if self.grp]
                
        self.OBJES.draw(self.grp, drawUngroups,d,classText)        
        
        _ = self.WALLS.draw() if drawWall else None
        plt.axis('equal')

        [li.draw() for li in self.LINKS]

        _ = plt.axis('off') if lim < 0 else (plt.xlim(-lim,lim), plt.ylim(-lim,lim))
        
        plt.savefig(os.path.join(self.imgDir,self.scene_uid+suffix+".png") if imageTitle=="" else imageTitle)
        plt.clf()
        plt.close()

        if drawRoomMask:
            self.drawRoomMask(os.path.join(self.imgDir,self.scene_uid+"_Mask.png") if imageTitle=="" else imageTitle[:-4]+"_Mask.png")

    def drawRoomMask(self,maskTitle=""):
        from PIL import Image, ImageDraw
        Image.fromarray(self.roomMask.astype(np.uint8)).save(self.imgDir+self.scene_uid + "_Mask.png" if maskTitle=="" else maskTitle)

    def renderables(self,objects_dataset,scene_render,no_texture=True,depth=0,height=0,sz=192, rt=25.):     #class top2down():
        self.OBJES.renderables(scene_render,objects_dataset,no_texture,depth)
        scene_render.add(self.WALLS.renderable_floor(depth=depth,sz=sz,rt=rt)) #depth is positive
        [scene_render.add(w.renderable(height=height)) for w in self.WALLS]
        return scene_render.renderables

    def exportAsSampleParams(self):
        from copy import copy
        c = copy(self.copy)
        c = self.OBJES.exportAsSampleParams(c)
        c["room_layout"] = self.roomMask[None,:]
        c["walls"] = self.WALLS.exportAsSampleParams()
        return c

    def bpt(self):
        return self.OBJES.bpt()
    
    def toSceneJson(self, roomType=None):
        sj = {"origin": self.scene_uid, "id": self.scene_uid,"bbox": {"min": [0,0,0], "max": [0,0,0]},"up": [0,1,0], "front": [0,0,1], "rooms":[]}
        #load the room

        rsj = {"id": self.scene_uid+"_0" if self.scene_uid else "0",
            "modelId": self.scene_uid,"roomTypes": [roomType],"origin": self.scene_uid,"roomId": 0,
            "bbox": {"min":[1000,1000,1000],"max":[-1000,-1000,-1000]},
            "objList":[],"blockList":[],"roomShape":[],"roomNorm":[],"roomOrient":[]
            }
        
        rsj = self.OBJES.toSceneJson(rsj)
        rsj = self.WALLS.toWallsJson(rsj)
        if len([w for w in self.WALLS if w.v]) > 0:
            bb = self.WALLS.bbox()
            rsj["bbox"] = {"min":[float(bb[0][0]),float(bb[0][1]),float(bb[0][2])],"max":[float(bb[1][0]),float(bb[1][1]),float(bb[1][2])]}

        sj["bbox"] = rsj["bbox"]
        sj["rooms"].append(rsj)
        return sj

        #endregion: presentation-#

    #endregion: in/outputs-------#

    #region: interfaces----------#
    def __getitem__(self,cl):
        return self.OBJES[cl]
       
    def __call__(self,nid):
        return [o for o in self.OBJES if o.nid==nid][0]
    #endregion: interfaces-------#

    #region: properties----------#
    def outOfBoundaries(self): #intersect area,  object occupied area in total,  room area
        contour = self.WALLS.shape()
        return sum([o.shape().area-contour.intersection(o.shape()).area for o in self.OBJES]), sum([o.shape().area for o in self.OBJES]), contour.area
    #endregion: properties-------#

    #region: operations----------#
    def draftRoomMask(self):
        from PIL import Image, ImageDraw
        L = self.roomMask.shape[-1]
        img = Image.new("L",(L,L))  
        img1 = ImageDraw.Draw(img)  
        for g in self.GRUPS:
            img1.rectangle(g.imgSpaceBbox(), fill ="white",outline="gray",width=2)
        img1.line([((L>>1,L>>1) if len(self.GRUPS)==1 else self.GRUPS[1].imgSpaceCe()),self.GRUPS[0].imgSpaceCe()],fill ="white",width=15)
        self.roomMask = np.array(img).astype(np.float32)

    def randomize(self, **kwargs):
        return self.OBJES.randomize(**kwargs)
    #endregion: operations-------#

import tqdm
class scneDs():
    def __init__(self,name="../novel3DFront/",lst=[],prepare="uncond",num=8,_dataset=[],**kwargs):#print(kwargs)
        from numpy.random import choice
        import os
        self.name,self._dataset = name, []
        if prepare=="uncond":
            LST = os.listdir(name) if (os.path.exists(name) and len(lst)==0 and num==0) else ([choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst)
            self.load(LST,name,num,**kwargs)
        elif prepare=="textcond":
            LST = os.listdir(name) if (os.path.exists(name) and len(lst)==0 and num==0) else ([choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst)
            self.prepareTextCond(LST,name,num,**kwargs)
        elif prepare=="roomcond":
            LST = os.listdir(name) if (os.path.exists(name) and len(lst)==0 and num==0) else ([choice(os.listdir(name)) for i in range(num) ] if os.path.exists(name) and len(lst)==0 else lst)
            self.prepareRoomCond(LST,name,num,**kwargs)
        elif prepare=="assigned":
            self._dataset = _dataset
        else:
            raise NotImplementedError
    
    #region: magics---------------#
    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
    #endregion: magics------------#
    
    #region: in/outputs-----------#
    
        #region: inputs-----------#
    def load(self,LST,name="",num=8,**kwargs):
        import json,os
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s loading %s "%(name,LST[i][:20]))
                try:
                    scene = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    scene = scne.fromSceneJson(json.load(open(os.path.join(name,LST[i],"scene.json"))))
            else:
                pbar.set_description("empty scene %s "%(i))
                scene = scne.empty(str(i))
            self._dataset.append(scene)
        #endregion: inputs--------#
        
        #region: outputs----------#
    def save(self,dir):
        import json,os
        for s in self._dataset:
            os.makedirs(os.path.join(dir,s.scene_uid),exist_ok=True)
            open(os.path.join(dir,s.scene_uid,'scene.json'),"w").write(json.dumps(s.toSceneJson())) #raise NotImplementedError#break
        #endregion: outputs-------#
        
    #endregion: in/outputs--------#
    
    #region: preparing------------#
    def prepareTextCond(self,LST,name="",num=8,**kwargs):
        import json,os
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s texting %s "%(name,LST[i][:20]))
                try:
                    template = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    template = scne.fromSceneJson(json.load(open(os.path.join(name,LST[i],"scene.json"))))
                scene = scne.empty(template.scene_uid)
                scene.text = template.text
            else:
                pbar.set_description("random texting %s "%(i))
                scene = scne.empty(str(i))
                #FIXME: generate a text for scene, although it hasn't been done yet, 
            self._dataset.append(scene)

    def prepareRoomCond(self,LST,name="",num=8,**kwargs):
        import json,os
        pbar = tqdm.tqdm(range(len(LST))) if os.path.exists(name) and len(LST) else tqdm.tqdm(range(num))
        for i in pbar:
            if os.path.exists(name) and len(LST):
                pbar.set_description("%s rooming %s "%(name,LST[i][:20]))
                try:
                    template = scne.fromNpzs(dir=name,name=LST[i],**kwargs)
                except:
                    template = scne.fromSceneJson(json.load(open(os.path.join(name,LST[i],"scene.json"))))
                scene = scne.empty(template.scene_uid)
                scene.registerWalls = template.WALLS
            else:
                pbar.set_description("random rooming %s "%(i))
                scene = scne.empty(str(i),keepEmptyWL=False)
                scene.WALLS.randomWalls()
            self._dataset.append(scene)
    #endregion: preparing---------#

    #region: operation------------#

    def synthesis(self,syth,cond,T):
        from ..Operation.Syth import agmt,gnrt,copl,rarg
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("%s-%s, %s "%(syth,cond,self._dataset[i].scene_uid[:20]))
            V=3 if len(self._dataset)==1 else 0
            if syth == "gnrt":
                S = gnrt(T,self._dataset[i],v=V)
            elif syth == "copl":
                S = copl(T,self._dataset[i],v=V)
            elif syth == "rarg":
                S = rarg(T,self._dataset[i],v=V)
            elif syth == "agmt":
                S = agmt(T,self._dataset[i],v=V)

            if syth == "agmt":
                S.scene.scene_uid += str(i)
                S.augment(cnt=1,draw=True)
            elif cond == "uncond":
                S.uncond(draw=True)
            elif cond == "textcond":
                S.textcond(draw=True)
            elif cond == "roomcond":
                S.roomcond(draw=True)

    def draw(self,**kwargs):
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("drawing %s "%(self._dataset[i].scene_uid[:20]))
            self._dataset[i].draw(**kwargs)

    def recognize(self,T,show=False,**kwargs):
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            pbar.set_description("recognizing %s "%(self._dataset[i].scene_uid[:20]))
            from ..Operation.Plan import plans
            plans(self._dataset[i],T,v=3 if (len(self._dataset)==1 and not show) else 0).recognize(show=show,**kwargs)

    def optimize(self,T,PatFlag,PhyFlag,steps,config,rand=-1):
        import os
        BASE_DIR = os.path.join(".","pattern","opts")
        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar: #range(len(self)):#
            pbar.set_description("optimizing %s"%(self._dataset[i].scene_uid[:20]))
            from ..Operation.Optm import optm #print(self._dataset[i])
            self._dataset[i].imgDir = os.path.join(BASE_DIR, self._dataset[i].scene_uid)
            os.makedirs(self._dataset[i].imgDir,exist_ok=True)
            O = optm(T,self._dataset[i],PatFlag=PatFlag,PhyFlag=PhyFlag,rand=rand,config=config,exp=False)
            O.loop(steps,pbar=pbar)
        
    def evaluate(self, metrics=[], cons=[], pmVersion="losy"):        
        from ..Operation.Patn import patternManager as PM
        import numpy as np
        T = PM(pmVersion)
        from SceneClasses.Operation.Plan import plans
        from evaClasses.Titles import titles, indexes
        
        #----------------------------------------original data
        OSA = [-1 for _ in range(len(self))]
        FIT = [-1 for _ in range(len(self))]
        DIS = [ [-1 for _ in range(len(self))] for __ in range(len(self)) ]
        varLevel = min(1,len(cons))
        #----------------------------------------original data
        
        #----------------------------------------statistics over condition
        osa = [[] for _ in range(varLevel+1)]
        fit = [[] for _ in range(varLevel+1)]
        if varLevel == 0:
            raise NotImplementedError("Currently, the unconditional variety calculation is not supported.")
        var = [[] for _ in range(varLevel+1)]
        nms = [[] for _ in range(varLevel+1)]
        #----------------------------------------statistics over condition
        
        #----------------------------------------condition level enumerate
        for vl in range(varLevel+1):
            childCons = cons.childs(vl) if len(cons) else []
            for ci in range(max(1,len(childCons))): #--combinatorial enumerating the conditions of this level
                #childcon = conditions(childCons[ci])
                
                Titles = titles("condition",*(childCons[ci])) if vl > 0 else titles("condition",*([{"fake":"c"}]) )
                indx = indexes.next(None,Titles)
                while indx:
                    lst = cons( **(indx.dict) ) if vl > 0 else [i for i in range(len(self))] #--enumerating the scenes of this condition
                    if "OSA" in metrics:
                        for i in lst:
                            OSA[i] = OSA[i] if OSA[i]>0 else self[i].outOfBoundaries()[0]
                        osa[vl].append( sum([OSA[i] for i in lst])/len(lst) )
                            
                    if "FIT" in metrics:
                        for i in lst:
                            FIT[i] = FIT[i] if FIT[i]>0 else plans(self[i],T,v=0).recognize(use=True,draw=False)[0]
                        fit[vl].append( sum([FIT[i] for i in lst])/len(lst) )

                    if "VAR" in metrics and vl > 0:
                        for i in lst:
                            for j in lst:
                                DIS[i][j] = DIS[i][j] if DIS[i][j] > -0.001 else ((self[i].plan.diff(self[j]) if self[i].plan else 0 ))
                        
                        var[vl].append( sum([sum([DIS[i][j] for j in lst]) for i in lst])/(len(lst)*len(lst)) )#[(indx.dict)] = 
                    nms[vl].append(indx.dict)
                    indx = indexes.next(indx)

        res,full = [],[]
        for t in metrics:
            if t == "CKL":
                res.append(None)
                full.append(None)
            elif t == "OSA":
                res.append(osa[0][0]) #??????
                full.append(osa)
            elif t == "FIT":
                res.append(fit[0][0]) #?????
                full.append(fit)
            elif t == "VAR":
                res.append(sum(var[1])/len(var[1])) #should we output the variety of two level conditions? or more level? and how?
                full.append(var)                                    #full is the answer of this question
        return res, full, nms

        if "OA" in metrics:
            outside_area = 0
            for s in self:
                outArea,objArea,area = s.outOfBoundaries()
                outside_area += outArea/len(self)

        if "FIT" in metrics or "CONDVAR" in metrics:            #1, semantic recognization
            fitness=0
            for s in self:
                P = plans(s,T,v=0)
                fit,ass,_ = P.recognize(use=True,draw=False)
                fitness += fit/len(self)

        if "CONDVAR" in metrics:                                 #2, difference synthesizing and calculation (lazy version)        
            disMatrix,setFlag,varLevel = np.zeros((len(self),len(self))),np.zeros((len(self),len(self))),min(1,len(cons)) #memorized
            from evaClasses.Titles import titles, indexes
            assert cons is not None and len(cons) < 3 
            condVars = [0 for _ in range(varLevel)]
            for vl in range(1,varLevel+1):# which level
                childCons = cons.childs(vl)
                condVarses = [0 for cc in childCons]
                for ci in range(len(childCons)):
                    #childcon = conditions(childCons[ci])
                    Titles = titles("condition",*(childCons[ci]))#childcon.cons))
            
                    indx = indexes.next(None,Titles)
                    while indx:        
                        lst = cons( **(indx.dict) )
                        #print(lst)
                        for i in lst:
                            for j in lst:
                                #print(i)
                                #print(j)
                                # if setFlag[i][j] < 0.5:
                                #     disMatrix[i][j] = self[i].plan.diff(self[j])
                                #     setFlag[i][j] = 1
                                disMatrix[i][j],setFlag[i][j] = ((self[i].plan.diff(self[j]) if self[i].plan else 0 )if setFlag[i][j]<0.5 else disMatrix[i][j]),1
                                condVarses[ci] = condVarses[ci]+disMatrix[i][j]
                        condVarses[ci] /= len(lst)*len(lst)
                        indx = indexes.next(indx)

                condVars[vl-1] = sum(condVarses)/len(condVarses) #is there any better ways to synthesis the conditional variety of different condition elements? rather than just the average value?

        if "CKL" in metrics:
            category = None#self.__compute_category_kl_divergence(raw_dataset_fake, raw_dataset_real, None)
        
        res = []
        for t in metrics:
            if t == "CKL":
                res.append(category)
            elif t == "OA":
                res.append(outside_area)
            elif t == "FIT":
                res.append(fitness)
            elif t == "CONDVAR":
                res.append(condVars[0]) #should we output the variety of two level conditions? or more level? and how?
        return res
    
    #endregion: operation---------#