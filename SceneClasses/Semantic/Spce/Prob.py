import numpy as np

class pont():
    def __init__(self, p, source):
        self.source = source
        self.p = p
        self.wid = None
        self.r = None
        
        self.s_l = None
        self.s_r = None
        pass

class sgmt():
    def __init__(self,H,source):
        self.source = source
        self.p_l = None
        self.p_r = None

        self.H = H
        pass


from ...Basic.Wall import EPS
class aref():
    def __init__(self, pro, WALLS):
        self.pro = pro
        self.WALLS = WALLS
        self.delta = pro.delta
        self.areaF = []
        self.Us=[]
        self.straight = []
        self.wallsSign = [[],[],[],[]]
        self.areaFunctionDetection()

    def areaFunctionDetection(self, f=False, j0=2):
        walls, delta = self.WALLS, self.delta

        assert not walls.cross(self.pro.toWalls(EPS))#delta))#
        wals = self.pro.toWalls()
        if f:
            print(self.p[0]+self.nearest[0])
            print(self.p[0]-self.nearest[0])
            print(wals)    
        self.areaF = [ [ [wals[0].p,200],[wals[0].q,-1]],[[wals[1].p,200],[wals[1].q,-1]],[[wals[2].p,200],[wals[2].q,-1]],[[wals[3].p,200],[wals[3].q,-1]] ]
        for i in range(len(walls)):
            w = walls[i]
            if not w.v:
                continue
            j = [j for j in range(4) if (wals[j].n@w.n > 1-EPS)]
            j = j[0]
            d = (self.areaF[j][0][0]-w.p)@w.n#print(d)
            if d < -EPS: continue
            else: d = max(d,0)
            if f and j == j0:
                print(i,w,j,wals[j],"matched",sep="\t")
            
            #find the area for w.p and test if we need to set a breakpoint on w.p
            k = 0
            P = self.areaF[j][k][0]
            r = wals[j].rate(P)
            while r<wals[j].rate(w.p)-EPS and k < len(self.areaF[j]):
                k+= 1
                P = None if k==len(self.areaF[j]) else self.areaF[j][k][0] 
                r = 1000 if k==len(self.areaF[j]) else wals[j].rate(P)
            #w.p is in the area between self.areaF[j][k-1][0]=P + epsilon and self.areaF[j][k][0] + epsilon  self.areaF[j][k-1][0]< w.p <=self.areaF[j][k][0] ::: k+1 always exist, which means k<len()-1
            #k = 0: means  w.p <=self.areaF[j][0][0] = 0
            #it means w.p starts at somewhere before wals[j].p or at wals[j].p
            #k = k: means self.areaF[j][k-1][0]< w.p <=self.areaF[j][k][0]
            #it means w.p starts at somewhere in this area, maybe at the back of this area
            #k =len: self.areaF[j][-1][0] < w.p
            if k!= 0 and k != len(self.areaF[j]): #w surely starts in the area of wals[j] means: wals[j].q <     w.p
                if wals[j].rate(w.p) < r-EPS: #w.p is not the same as self.areaF[j][k][0], it's surely in the area; so we should set a breakpoint for it
                    R = wals[j].rate(w.p)
                    di = self.areaF[j][k-1][1]
                    self.areaF[j].insert(k,[wals[j].p*(1-R)+wals[j].q*R,di])
            kP = k
            if f and j == j0:
                print("wall %d start from [%.3f, %.3f], it start in k=%d"%(w.idx,w.p[0],w.p[2],kP))
                print(k)
                print(P)
                #print(self.areaF[j][k][0])
                #print(wals[j].rate(P))
                print(r)
                print(wals[j].rate(self.areaF[j][k-1][0]))
                print(wals[j].rate(w.p))

            #find the area for w.q and test if we need to set a breakpoint on w.q
            k = max(k-1,0)
            P = self.areaF[j][k][0]
            r = wals[j].rate(P)
            while r<wals[j].rate(w.q)-EPS and k < len(self.areaF[j]):
                k+= 1
                P = None if k==len(self.areaF[j]) else self.areaF[j][k][0]
                r = 1000 if k==len(self.areaF[j]) else wals[j].rate(P)
            #w.q is in the area between self.areaF[j][k-1][0] + epsilon and self.areaF[j][k][0]=P + epsilon  self.areaF[j][k-1][0]< w.q <= self.areaF[j][k][0]
            #k = 0: means  w.q <= self.areaF[j][0][0] = 0
            #it means w.p starts at somewhere before wals[j].p or at wals[j].p
            #k = k: means self.areaF[j][k-1][0]< w.q <=self.areaF[j][k][0]
            #it means w.p starts at somewhere in this area, maybe at the back of this area
            #k =len: self.areaF[j][-1][0] < w.q
            if k != 0 and k != len(self.areaF[j]): #w surely ends in the area of wals[j] means: self.areaF[j][k-1][0]< w.q <=self.areaF[j][k][0]
                if wals[j].rate(w.q) < r-EPS: #w.q is not the same as self.areaF[j][k][0], it's surely in the area; so we should set a breakpoint for it
                    R = wals[j].rate(w.q)
                    di = self.areaF[j][k-1][1]
                    self.areaF[j].insert(k,[wals[j].p*(1-R)+wals[j].q*R,di])
            kQ = k
            if f and j == j0:
                print("wall %d ends at [%.3f, %.3f], it ends at k=%d"%(w.idx,w.q[0],w.q[2],kQ))
                print(k)
                print(P)
                #print(self.areaF[j][k][0])
                #print(wals[j].rate(P))
                print(r)
                print(wals[j].rate(self.areaF[j][k-1][0]))
                print(wals[j].rate(w.q))
                if kP<kQ:
                    print("d is %.3f"%(d))
                
            #traverse from w.p's area to w.q's area
            kk = kP
            for t in range(kP,kQ):
                if self.areaF[j][kk][1] > d:
                    if kk>1 and abs(self.areaF[j][kk-1][1]-d)<EPS: #merge it with the area before me
                        del self.areaF[j][kk]
                    else: #cover this area
                        self.areaF[j][kk][1]=d
                        kk+=1

            if f and j == j0:
                print("→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in self.areaF[j][:-1]])+"(%.3f,%.3f)"%(self.areaF[j][-1][0][0],self.areaF[j][-1][0][2]))
            
        self.Us=[]
        self.straight = []
        for i in range(4):
            for j in self.areaF[i][:-1]:#print(wals[i].rate(j[0]))
                if (wals[i].rate(j[0])<EPS and i == 0) or not(wals[i].rate(j[0])<EPS and max(j[1],delta)==self.straight[-1][1]):
                    self.straight.append([wals[i].rate(j[0])+i,max(j[1],delta),0])
        if self.straight[-1][1]==self.straight[0][1]:
            if len(self.straight)==1:
                self.straight[0][2]=wals[0].length+wals[1].length+wals[2].length+wals[3].length
                return
            del self.straight[0]
            
        s = 0
        a = 0
        #print(self.straight)#raise NotImplementedError
        self.straight.append([self.straight[0][0]+4,self.straight[0][1],0])
        if f:
            print("straight: "+", ".join([ "(%.2f,%.2f,%.2f)"%(a[0],a[1],a[2]) for a in self.straight]))
        
        for i in range(8):
            while s<len(self.straight) and self.straight[s][0] < i+1:
                self.straight[s-1][2] += (self.straight[s][0]-a)*wals[i%4].length
                a = self.straight[s][0]
                s += 1
            a = i+1
            self.straight[s-1][2] += (a-max(self.straight[s-1][0],i))*wals[i%4].length
        if f:
            print("straight: "+", ".join([ "(%.2f,%.2f,%.2f)"%(a[0],a[1],a[2]) for a in self.straight]))
        
        self.straight = self.straight[:-1]
        
        
        for s in range(len(self.straight)):
            if self.straight[s][1]>max(self.straight[s-1][1],self.straight[(s+1)%len(self.straight)][1]):
                l = self.straight[s][2]
                #try:
                self.Us.append(l - l*np.math.exp(-0.05*(2.0*self.straight[s][1])/(self.straight[s-1][1]+self.straight[(s+1)%len(self.straight)][1])))
                #except:
                #    pass

        self.Us = sorted(self.Us,key=lambda x:-x)
            #go from w.p to w.q

        self.wallsSign = [[],[],[],[]]
        w = self.pro.toWalls()
        for i in range(4):
            for j in self.areaF[i]:
                self.wallsSign[i].append([j[0],j[1],(j[1]<self.delta),w[i].rate(j[0]),w[i].rate(j[0])*w[i].length])
    
    def __str__(self):
        #self.areaF = [ [ (wals[0].p,20),(wals[0].q,-1)],[(wals[1].p,20),(wals[1].q,-1)],[(wals[2].p,20),(wals[2].q,-1)],[(wals[3].p,20),(wals[3].q,-1)] ]
        #self.straight = [relLength,dis,absLength]
        print(self.pro)
        print("areaF:\n"+'\n'.join([ "→".join([ "(%.3f,%.3f)←↑%.3f"%(a[0][0],a[0][2],a[1]) for a in ar[:-1]])+"(%.3f,%.3f)"%(ar[-1][0][0],ar[-1][0][2])  for ar in self.areaF ]))
        print("\nstraight: "+", ".join([ "(%.3f,%.3f,%.3f)"%(a[0],a[1],a[2]) for a in self.straight]))
        print("Us:       "+str(self.Us))
        print("onWallLength: "+str(self.onWallLength())+"\tonWallSegment:"+str(self.onWallSegment())+"\tseparation:   "+str(self.separation()))
        return

    @property
    def onWallLength(self):
        return np.sum([s[2]*int(s[1]<self.delta+EPS) for s in self.straight])

    @property
    def onWallSegment(self):
        return np.sum([int(s[1]<self.delta+EPS) for s in self.straight])

    @property
    def avgOnWallLength(self):
        return self.onWallLength/max(1,self.onWallSegment)

    @property
    def separation(self):
        return 0 if len(self.Us)<2 else np.sum(self.Us[1:])

    def offWallSegment(self,f=False):
        from copy import deepcopy
        from ...Basic import wall
        #just check the off wall segments of the spce?
        height = self.wallsSign[3][-1][1]
        idx = [0,0]
        while True:
            if self.wallsSign[idx[0]][idx[1]][1]<EPS and height>EPS:
                break
            height = self.wallsSign[idx[0]][idx[1]][1]
            if len(self.wallsSign[idx[0]]) > idx[1]+2:
                idx[1] += 1
            else:
                idx[1]=0
                idx[0]=(idx[0]+1)%4

        IDX = [idx[0],idx[1]]
        #print(IDX)
        offWallSegments = []
        currentSegment = None#[None,0,None,[]] #startingpoint, length, endingpoint, walls
        tw = self.pro.toWalls()
        while True:
            height = self.wallsSign[idx[0]][idx[1]][1]
            if len(self.wallsSign[idx[0]]) > idx[1]+2:
                idx[1] += 1
            else:
                idx[1]=0
                idx[0]=(idx[0]+1)%4
                if currentSegment is not None and not (self.wallsSign[idx[0]][idx[1]][1]<EPS and height > EPS):
                    P = self.wallsSign[idx[0]][idx[1]][0]
                    currentSegment[3][-1].q = np.copy(P)
                    #currentSegment[3][-1].lengthh()
                    N = currentSegment[3][-1].n
                    from ...Basic import wall
                    currentSegment[3].append(wall(P,P,np.copy(tw[idx[0]].n),-1,-1,-1))
            
            if self.wallsSign[idx[0]][idx[1]][1]<EPS*100 and height > EPS*100:
                #end an offWallSegment
                P = self.wallsSign[idx[0]][idx[1]][0]
                currentSegment[2] = deepcopy(P)
                currentSegment[3][-1].q = np.copy(P)
                #currentSegment[3][-1].lengthh()
                currentSegment[1] = np.sum([w.length for w in currentSegment[3]])
                offWallSegments.append(deepcopy(currentSegment))
                currentSegment = None
            elif self.wallsSign[idx[0]][idx[1]][1]>EPS*100 and height > EPS*100:
                P = self.wallsSign[idx[0]][idx[1]][0]
                currentSegment[3][-1].q = np.copy(P)
                #currentSegment[3][-1].lengthh()
                #stretch the offWallSegment
                pass
            elif self.wallsSign[idx[0]][idx[1]][1]>EPS*100 and height < EPS*100: 
                #start an offWallSegment
                P = self.wallsSign[idx[0]][idx[1]][0]
                currentSegment = [P,0,None,[wall(P,P,tw[idx[0]].n,-1,-1,-1)]] #startingpoint, length, endingpoint, walls
                pass

            if idx[0] == IDX[0] and idx[1] == IDX[1]:
                break 
        
        if f:
            for c in offWallSegments:
                print("[%.3f, %.3f] -> %.3f -> [%.3f, %.3f]"%(c[0][0],c[0][2],c[1],c[2][0],c[2][2]))
                for w in c[3]:
                    print(w)
                print("\n")
        return offWallSegments

    def I_J(self):
        Area=-100
        for j in range(4):
            area=0
            i = (j-1)%4
            idx = len(self.wallsSign[i])-1
            while idx > -1:
                if self.wallsSign[i][idx][1] < 0.42:
                    area+=self.wallsSign[i][idx][-1]
                else:
                    break
                idx -= 1

            idx = 0
            while idx < len(self.wallsSign[j]):
                if self.wallsSign[j][idx][1] < 0.42:
                    area+=self.wallsSign[j][idx][-1]
                else:
                    break
                idx += 1
            if area > Area:
                I,J = (j-1)%4,j
                Area = area
        return I,J

class prob():
    def __init__(self,p,delta,WALLS):
        self.p=np.copy(p)
        self.delta = delta
        self.WALLS = WALLS
        self.res=[(20,False),(20,False),(20,False),(20,False)]
        self.AREF = None #aref(self,WALLS)

    #region: properties----------#
    def __str__(self):
        return "[%.4f,%.4f]: xmin: %.4f %s , zmin: %.4f %s , xmax: %.4f %s , zmax: %.4f %s"%(self.p[0],self.p[2],self.res[0][0],("in" if self.res[0][1] else "out"),self.res[1][0],("in" if self.res[1][1] else "out"),self.res[2][0],("in" if self.res[2][1] else "out"),self.res[3][0],("in" if self.res[3][1] else "out")) if self.AREF is None else str(self.AREF)
    
    @property
    def status(self):
        return self.res[0][1], (self.res[0][1] == self.res[1][1]) and (self.res[1][1] == self.res[2][1]) and (self.res[2][1] == self.res[3][1])

    @property
    def nearest(self):
        return [min(self.res[0][0],self.res[2][0])+EPS,min(self.res[1][0],self.res[3][0])+EPS]
    
    @property
    def ratio(self):
        a = self.nearest
        return a[0]/max(a[1],EPS/10) if a[0]<a[1] else a[1]/max(a[0],EPS/10)

    @property
    def area(self):
        return self.nearest[0]*self.nearest[1]

    #endregion: properties-------#
    
    def updates(self):
        for w in self.WALLS:
            if w.v and w.over(self.p): self.update(w)

    def update(self,w):
        dis,vec = w.distance(self.p)
        if abs(vec[2])<EPS/10: 
            if vec[0]<EPS:
                if abs(dis)-EPS < self.res[0][0]:
                    self.res[0] = (abs(dis)-EPS,(dis>-EPS))
            if vec[0]>-EPS:
                if abs(dis)-EPS < self.res[2][0]:
                    self.res[2] = (abs(dis)-EPS,(dis>-EPS))
        if abs(vec[0])<EPS/10:
            if vec[2]<EPS:
                if abs(dis)-EPS < self.res[1][0]:
                    self.res[1] = (abs(dis)-EPS,(dis>-EPS))
            if vec[2]>-EPS:
                if abs(dis)-EPS < self.res[3][0]:
                    self.res[3] = (abs(dis)-EPS,(dis>-EPS))
        if abs(vec[0])>EPS/10 and abs(vec[2])>EPS/10:
            print(str(vec)+"not vertical or horizontal wall, currently not supported by space detection or prob-update")
            print(w)
            raise NotImplementedError

    def key(self,hint=None):
        self.environ()
        if hint is None:
            return -1.0/self.ratio+np.average(self.nearest)*2 - self.AREF.separation*5 + self.AREF.onWallLength
        else:
            return -1.0/self.ratio+np.average(self.nearest)*2 - self.AREF.separation*5 + self.AREF.onWallLength

    def adjust(self,delta):
        if abs(self.res[0][0]-self.res[2][0])<delta*2:
            a = (self.res[0][0]+self.res[2][0])/2.0
            b = (self.res[0][0]-self.res[2][0])/2.0
            self.res[0],self.res[2] = (a,True),(a,True)
            self.p[0] += b
        if abs(self.res[1][0]-self.res[3][0])<delta*2:
            a = (self.res[1][0]+self.res[3][0])/2.0
            b = (self.res[1][0]-self.res[3][0])/2.0
            self.res[1],self.res[3] = (a,True),(a,True)
            self.p[2] += b

    def inner(self,w):
        dis,vec = w.distance(self.p)
        if abs(vec[2])<EPS/10: 
            if vec[0]<EPS:
                if abs(dis)-EPS*2 < self.res[0][0] and dis > -EPS: #
                    return vec,True
            if vec[0]>-EPS:
                if abs(dis)-EPS*2 < self.res[2][0] and dis > -EPS:
                    return vec,True
        if abs(vec[0])<EPS/10:
            if vec[2]<EPS:
                if abs(dis)-EPS*2 < self.res[1][0] and dis > -EPS:
                    return vec,True
            if vec[2]>-EPS:
                if abs(dis)-EPS*2 < self.res[3][0] and dis > -EPS:
                    return vec,True
        if abs(vec[0])>EPS/10 and abs(vec[2])>EPS/10:
            print(str(vec)+"not vertical or horizontal wall, currently not supported by space detection or prob-update")
            raise NotImplementedError
        return [0,0],False

    def environ(self):
        if self.AREF is None: self.AREF = aref(self,self.WALLS)

    def toWalls(self,eps=0): #,w=None,x=None,qro=None
        from ...Basic import walls
        return walls(c=[self.p[0],self.p[2]],a=[self.nearest[0]-eps,self.nearest[1]-eps])
    
