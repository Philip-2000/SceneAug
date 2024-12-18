#from .Scne import scne
#from ..Basic.Obje import object_types, obje
import os
SYN_IMG_BASE_DIR = "./pattern/syth/"

class agmt():
    def __init__(self,pmVersion,scene,nm="test",v=0):
        from .Patn import patternManager as PM
        self.verb=v
        self.scene = scene
        self.scene.imgDir = os.path.join(SYN_IMG_BASE_DIR,pmVersion,"agmt")
        os.makedirs(self.scene.imgDir,exist_ok=True)
        self.pm = PM(pmVersion)
        self.name = nm

    def augment(self,sdev=0.2,cdev=2.,cnt=8,draw=False):
        if not self.scene.grp:
            from .Plan import plans
            plans(self.scene,self.pm,v=0).recognize(use=True,draw=False,show=False)
        from numpy.random import rand as R
        from numpy.random import randint as Ri 
        import numpy as np
        from math import pi as PI 
        from copy import copy
    
        result,logs = [],[]
        for _ in range(cnt):
            scene = copy(self.scene)
            scene.scene_uid = scene.scene_uid+str(_)
            os.makedirs(scene.imgDir,exist_ok=True)
            if len(scene.GRUPS) == 1:
                l  = (R(3,)-0.5)*cdev
                R0 = (R(3,)-0.5)*sdev+1.0
                Ri0= (Ri(4)/2.0-1)*PI
                scene.GRUPS[0].adjust(l,R0,Ri0)
                logs.append({"t":0,"l":l,"Rs":[R0],"Ris":[Ri0]})
            elif len(scene.GRUPS) == 2:
                t,l = (R()*2-1)*PI,max([scene.GRUPS[0].size[0],scene.GRUPS[0].size[2],scene.GRUPS[1].size[0],scene.GRUPS[1].size[2]]) - R()*0.1
                d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
                Ri0, Ri1 = (Ri(4)/2.0-1)*PI,(Ri(4)/2.0-1)*PI
                R0, R1 = (R(3,)-0.5)*sdev+1.0,(R(3,)-0.5)*sdev+1.0
                scene.GRUPS[0].adjust( d*l,R0,Ri0)
                scene.GRUPS[1].adjust(-d*l,R1,Ri1)
                logs.append({"t":t,"l":l,"Rs":[R0,R1],"Ris":[Ri0,Ri1]})
            elif len(scene.GRUPS) == 3:
                raise NotImplementedError
                t,l = (R()*2-1)*PI,max([scene.GRUPS[0].size[0],scene.GRUPS[0].size[2],self.scene.GRUPS[1].size[0],self.scene.GRUPS[1].size[2]]) - R()*0.1
                d = np.array([np.math.cos(t),0.0,np.math.sin(t)])
                scene.GRUPS[0].adjust( d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
                scene.GRUPS[1].adjust(-d*l,(R(3,)-0.5)*sdev+1.0,(Ri(4)/2.0-1)*PI)
                logs.append({"t":t,"l":l,"Rs":[R0,R1,R2],"Ris":[Ri0,Ri1,Ri2]})
            else:
                raise NotImplementedError
        
            scene.draftRoomMask()
            result.append(scene)
            if draw:
                scene.draw()
        return result, self.scene, logs
    
    def show(self):
        return #not sure yet

class syth():
    def __init__(self,pmVersion,scene,appli,nm,v):
        from .Patn import patternManager as PM
        self.verb=v
        self.scene = scene
        self.scene.imgDir = os.path.join(SYN_IMG_BASE_DIR,pmVersion,appli)
        os.makedirs(self.scene.imgDir,exist_ok=True)
        self.pm = PM(pmVersion)
        self.name = nm
     
    def uncond(self):
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene

    def show():
        return

class gnrt(syth):
    def __init__(self,pmVersion,scene=None,nm="test",v=0):
        super(gnrt,self).__init__(pmVersion,scene,self.__class__.__name__,nm,v)

    def __randomchoose(self,node,version = "confidence"):
        import numpy as np
        if version == "confidence":
            cs = 0
            for ed in node.edges:
                cs += ed.confidence
                if np.random.rand() < ed.confidenceIn:
                    return ed
                if np.random.rand() < cs:
                    return None           
        if version == "random":
            return node.edges[np.random.randint(0,len(node.edges))]
        if version == "random_confidence":
            while 1:
                ran = np.random.randint(0,len(node.edges))
                if np.random.rand() < node.edges[ran].confidence:
                    return node.edges[ran]
        if version == "random_confidence_times":
            value = 0
            for i in node.edges: value += i.confidence*i.times
            while 1:
                ran = np.random.randint(0,len(node.edges))
                if np.random.rand() < node.edges[ran].confidence and np.random.rand() < node.edges[ran].confidence*node.edges[ran].confidence/(0.1+value):
                    node.edges[ran].times += 1
                    return node.edges[ran]

    def uncond(self,draw=False,uid=""):
        import numpy as np
        from ..Semantic.Link import objLink
        from ..Basic.Obje import obje,object_types
        N = self.pm.nods[0]
        while len(N.edges)>0:
            ed = self.__randomchoose(N,"random_confidence_times")
            if ed :
                N,m = ed.endNode,ed.startNode
                while not (N.idx in m.bunches):
                    m = m.source.startNode
                r = m.bunches[N.idx].sample()
                a = [o for o in self.scene.OBJES if o.nid == m.idx] if m.idx > 0 else [obje(np.array([0,0,0]),np.array([1,1,1]),np.array([0]))]
                o = a[0] + obje.fromFlat(r,j=object_types.index(N.type))
                self.scene.addObject(o)
                o.nid = N.idx
                if m.idx > 0:
                    self.scene.LINKS.append(objLink(a[0].idx,o.idx,len(self.scene.LINKS),self.scene))
        if draw:
            self.scene.draw()
        return self.scene

    def textcond(self):
        raise NotImplementedError
        return self.scene

    def tempDebugger(self,theScene,cObjectList,o,spc,imgName):
        from ..Basic.Scne import scne
        from copy import copy
        tmpScene = scne.empty()
        wa = copy(theScene.WALLS)
        cObjectLis = copy(cObjectList)
        sp = copy(theScene.SPCES)
        sp.SPCES.append(spc)
        tmpScene.registerWalls(wa)
        tmpScene.registerSpces(sp)
        for oo in cObjectLis:
            pp = spc.transformOutward([oo])
            tmpScene.addObject(pp[0])
        if o is not None:
            p = spc.transformOutward([o])
            tmpScene.addObject(copy(p[0]))
        #tmpScene.imgDir="./spce_generate/"
        tmpScene.draw(imageTitle="./spce_generate/"+imgName,d=True,lim=5)

    def roomcond(self):
        raise NotImplementedError
        from ..Basic.Obje import obje, object_types
        if useWalls:
            assert theScene.WALLS is not None
            #f = open("./spce_generate/log.txt",'w')#sys.stdout#
            if len(rots)==0:
                rots = ["King-size Bed","Dressing Table"] if np.random.rand()<-0.5 else ["Coffee Table","Dining Table"]

            Sps = spces(wals=theScene.WALLS,drawFolder="./spce_generate/"+theScene.WALLS.name+"/")
            theScene.registerSpces(Sps)
            for r in rots:#[:1]
                if rots.index(r)>-1:
                    spc = Sps.extractingSpce(hint=[5,5])
                else:
                    PRO = prob(np.array([-0.2,0,-0.04]))
                    PRO.res=[(3.1785,True),(2.6265,True),(3.1785,True),(1.2910,True)]
                    PRO.areaFunctionDetection(theScene.WALLS,Sps.delta)
                    spc = spce.fromPro(PRO,scne=theScene,delta=Sps.delta)

                cObjectList = []
                rf = self.nods[0].bunches[self.rootNames.index(r)+1].exp
                N = self.nods[self.rootNames.index(r)+1]
                ro= obje.fromFlat(rf,j=object_types.index(N.type))
                ro.translation = np.copy(ro.size)
                ro.samplesBound()
                ro.nid = N.idx
                cObjectList.append(ro)

                self.tempDebugger(theScene,cObjectList,None,spc,"%d os=%d"%(rots.index(r),len(cObjectList)))

                adding=True
                while adding:
                    cs,m = 0,None
                    if len(N.edges)==0:
                        break
                    for ed in N.edges:
                        cs += ed.confidence
                        if ed in N.edges and N.edges.index(ed)==0:#np.random.rand() < ed.confidenceIn:
                            N,m = ed.endNode,ed.startNode
                            BD = np.array([o.samplesBound()[1] for o in cObjectList]).max(axis=0)
                            area = 2*spc.relA-BD #np.array([BD[1]-2*spc.a,[0,0,0]]).max(axis=0)
                            # print("---------------------start------------------"+str(len(cObjectList)))
                            # print("spc.relA",spc.relA,"BD",BD,"area",area)
                            Ntype = N.type if N.type.find('/') == -1 else N.type[:N.type.find('/')]

                            while not (N.idx in m.bunches):
                                m = m.source.startNode
                            ro = m.bunches[N.idx].exp
                            a = [o for o in cObjectList if o.nid == m.idx]
                            o = a[0] + obje.fromFlat(ro,j=object_types.index(N.type))
                            o.nid = N.idx

                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s before viola"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))

                            delta = 0.04
                            bd = o.samplesBound()
                            viola = [np.array([bd[0],[0,0,0]]).min(axis=0),np.array([bd[1]-2*spc.relA,[0,0,0]]).max(axis=0)]
                            # print(viola) viola[0] is zero or negative, viola[1] is zero or positive
                            if min(viola[0][0],viola[0][2]) < -delta*2 and max(viola[1][0],viola[1][2]) > delta*2:
                                # print("viola[0].min() < -delta*5 and viola[1].max() > delta*5 no space")
                                adding=False
                                break
                            if min(viola[0][0],viola[0][2]) > -delta*2 and max(viola[1][0],viola[1][2]) < delta*2:
                                # print("no viola")
                                self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s with no voila"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))
                                cObjectList.append(o)
                                
                                continue

                            moving = (min(viola[0][0],viola[0][2]) < -delta*2)
                            #vio = viola[0] if min(viola[0][0],viola[0][2]) < -delta*2 else viola[1]
                            if moving:
                                # print("moving "+str(len(cObjectList)))
                                if min(area[0]+viola[0][0],area[2]+viola[0][2]) <-delta*0:
                                    # print("too large viola, please find another node")
                                    continue
                                else:
                                    vio0 = [viola[0][0],viola[0][1],viola[0][2]]
                                    viola[0] = [max(viola[0][0],-area[0]),viola[0][1],max(viola[0][2],-area[2])]
                                    vio = [(viola[0][0]+vio0[0])/2.0,viola[0][1],(viola[0][2]+vio0[2])/2.0]
                            else:
                                if max(viola[1][0],viola[1][2]) > delta*0:
                                    # print("too large viola, please find another node")
                                    continue
                                else:
                                    vio = viola[1]/2.0
                            
                            o.translation -= vio

                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s after fixing vio"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))
                            if not moving:
                                collide = False
                                for oo in cObjectList:
                                    if oo.class_name().find("Lamp")==-1:
                                        a,b=oo.distance(o)
                                        collide = b
                                        if collide:
                                            break
                                if collide:
                                    # print("collision occurs while fixing viola")
                                    continue

                            if moving:
                                for oo in cObjectList:
                                    oo.translation -= vio
                            
                            self.tempDebugger(theScene,cObjectList,o,spc,"%d os=%d %d's child = %s after moving everything"%(rots.index(r),len(cObjectList),ed.startNode.edges.index(ed),Ntype))

                            cObjectList.append(o)
                        else:
                            break

                for oo in cObjectList:
                    theScene.addObject(oo)
                
                BD = np.array([o.samplesBound()[1] for o in cObjectList]).max(axis=0)
                spc.recycle(BD,Sps.WALLS)
                Sps.SPCES.append(spc)
                Sps.draw()
                spc.scne=theScene

                self.tempDebugger(theScene,cObjectList,None,spc,"%d os=%d after recycling"%(rots.index(r),len(cObjectList)))
                            
                a = Sps.eliminatingSpace(spc)
                if not a:
                    break
    
            return

        return self.scene

class copl(syth):
    def __init__(self,pmVersion,scene,nm="test",v=0):
        super(copl,self).__init__(pmVersion,scene,self.__class__.__name__,nm,v)
        raise NotImplementedError

    def uncond(self):#可以类似于rearrange
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene

class rarg(syth):
    def __init__(self,pmVersion,scene,nm="test",v=0):
        super(rarg,self).__init__(pmVersion,scene,self.__class__.__name__,nm,v)

    def __clean_find(self,node,level):
        if level == -1:
            node.chosen_level = 0
            for i in node.edges:
                self.__clean_find(i.endNode,level)
        if node.chosen_level >= 2**level:
            node.chosen_level -= 2**level
        for i in node.edges:
            if i.endNode.chosen_level >= 2**level:
                i.endNode.chosen_level -= 2**level
                self.__clean_find(i.endNode,level)

    def uncond(self,draw = True):
        import numpy as np
        from ..Semantic.Link import objLink
        from ..Basic.Obje import obje,object_types,objes
        from ..Basic.Scne import scne
        indexs = [f.class_index for f in self.scene.OBJES]
        choose_edges = None
        most_long_edge = []
        most_len = 0
        level = 0
        N = self.pm.nods[0]
        #print([object_types.index(ed.endNode.type) for ed in N.edges])
        #print(indexs)
        begin_len = len(indexs)
        most_level = len([i for i in indexs if i in [4,8,29,7,9]])
        choose_edges = [[] for i in range(0,most_level)]
        chosen_index = [[] for i in range(0,most_level)]
        while(True):
            from ..Basic.Obje import obje,object_types,obje
            N = self.pm.nods[0]
            if most_level > 1 and level >= most_level:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[-1]
                indexs += chosen_index[-2]
                chosen_index[-1] = []
                chosen_index[-2] = []
                choose_edges[-1] = []
                choose_edges[-2] = []
                level -= 2
                continue
            if most_level == 1 and level >= most_level:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[-1]
                chosen_index[-1] = []
                choose_edges[-1] = []
                level -= 1
                continue
            if level ==  -1:
                break
            while N.edges and indexs:
                if_continue = False
                for ed in N.edges:
                    if ed.endNode.chosen_level >= 2**level: continue
                    number = object_types.index(ed.endNode.type)
                    if number in indexs:
                        chosen_index[level].append(number)
                        indexs.remove(number)
                        choose_edges[level].append(ed)
                        N = ed.endNode
                        if_continue = True
                        break
                    else:
                        ed.endNode.chosen_level += 2**level
                if if_continue: continue
                if N.chosen_level< 2**level: N.chosen_level += 2**level
                break
            if not N.edges:
                if N.chosen_level< 2**level: N.chosen_level += 2**level
                level += 1
                continue
            if not indexs:
                most_long_edge = choose_edges.copy()
                break
            if self.pm.nods[0].chosen_level < 2**level:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                indexs += chosen_index[level]
                chosen_index[level] = []
                choose_edges[level] = []
                continue
            else:
                if most_len < begin_len - len(indexs):
                    most_long_edge = choose_edges.copy()
                    most_len = begin_len - len(indexs)
                if level == 0:
                    break
                indexs += chosen_index[level-1]
                chosen_index[level-1] = []
                choose_edges[level-1] = []
                self.__clean_find(self.pm.nods[0],level)
                level -= 1
                continue  
        self.__clean_find(self.pm.nods[0],-1)
        choose_edges = most_long_edge
        #print("most_level",most_level)
        #print("edge_len",most_len)
        set_idx = []
        for i in range(0,len(choose_edges)):
            offset = [[0,0,0],[4,0,0],[-4,0,0],[0,4,0],[0,-4,0]]
            idxes = []
            for ed in choose_edges[i]:
                N,m = ed.endNode,ed.startNode
                while not (N.idx in m.bunches):
                    m = m.source.startNode
                r = m.bunches[N.idx].sample()
                a = [o for o in self.scene.OBJES if o.nid == m.idx and not o.idx in set_idx] if m.idx > 0 else [obje(np.array(offset[i]),np.array([1,1,1]),np.array([0]))]
                o = a[0] + obje.fromFlat(r,j=object_types.index(N.type))
                o.nid = N.idx
                for obj in self.scene.OBJES.OBJES:
                    if obj.class_index == o.class_index:
                        idxes.append(obj.idx)
                        obj.translation = o.translation
                        obj.size = o.size
                        obj.orientation = o.orientation
                        obj.v, obj.modelId, obj.gid, obj.nid,obj.linkIndex,obj.destIndex = o.v, o.modelId, o.gid, o.nid,o.linkIndex,o.destIndex
                        if m.idx > 0:
                            self.scene.LINKS.append(objLink(a[0].idx,obj.idx,len(self.scene.LINKS),self.scene))
                        break
            set_idx += idxes
        if draw:
            self.scene.draw()
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene