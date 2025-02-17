from .Syth import syth
class gnrt(syth):
    def __init__(self,pm,scene=None,nm="test",v=0):
        super(gnrt,self).__init__(pm,scene,self.__class__.__name__,nm,v)

    def __randomchoose(self,node,version = "confidence"):
        raise AssertionError("This function is outdated")
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

    def uncond(self,draw=False):
        self.scene = self.pm.random_scene()
        if draw: self.scene.draw(),self.scene.save()
        return self.scene

    def textcond(self, draw=True):
        return self.textconds(draw)
        from ..Patn import pathses
        self.paths = pathses(self.pm)
        matchs = self.paths.matching(self.scene.TEXTS)
        match = matchs[0]
        match.utilize(self.scene,[[-2,0,0],[2,0,0]],[[0],[0]])
        if draw: self.scene.draw(),self.scene.save()
        return self.scene

    def textconds(self, draw=True):
        from ..Patn import pathses
        self.paths = pathses(self.pm)
        self.scene.imgDir = "./pattern/syth/"+self.pm.version+"/gnrt_syth/test/"
        import os
        os.makedirs(self.scene.imgDir,exist_ok=True)
        matchs = self.paths.matching(self.scene.TEXTS)
        for cnt, match in enumerate(matchs[:5]):
            from ...Basic import scne
            self.scene = scne.empty(keepEmptyWL=True)
            self.scene.imgDir = "./pattern/syth/"+self.pm.version+"/gnrt_syth/test/"
            match.utilize(self.scene,[[-2,0,0],[2,0,0]],[[0],[0]])
            if draw:
                self.scene.scene_uid = str(cnt+1)
                self.scene.draw(),self.scene.save()
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

    def roomcond(self,use=True,draw=True):
        self.scene.imgDir = "./pattern/syth/"+self.pm.version+"/gnrt/room/"
        #print(self.scene.imgDir)
        self.scene.SPCES.drop(self.pm)#,["Coffee Table", "Dining Table"])
        self.scene.draw()
        return self.scene
        """
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
                                    if oo.class_name.find("Lamp")==-1:
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
        """

    def txrmcond(self):
        #（1）基于语句获得所有的路径
        #（2）利用路径检测空间
        #（3）逐物体向空间中添加并整体移动
        #（4）如果所有物体完全放置了就回收空间，形成一个新的walls
        #（4.5）如果空间快要不够了，而且我还是第一条路径，就不放置非关键物体了，
        #（5）如果还有空间，就采样下一条路径，再重复
        #空间提取与占用
        return self.scene