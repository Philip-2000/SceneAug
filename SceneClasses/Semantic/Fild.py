import numpy as np
class sam():
    def __init__(self,s,transl,config):
        self.transl = transl
        self.s = s
        self.wo, self.wi, self.dr = self.s.WALLS.optFields(self,config)
        self.ob, self.p_ob = self.s.OBJES.optFields(self, None, config["object"])
        self.res_p = self.wo + self.wi
        self.res_ob = self.res_p + self.dr
        self.res = self.res_ob + self.ob
        self.p = 0

    def __call__(self, config):
        self.ob, p = self.s.OBJES.optFields(self, None, config["object"])
        self.res = self.res_ob + self.ob
        self.p = self.p_ob + p
        return self.res
    
    def integral(self,sample):
        self.p_ob = sample.p_ob + 0.5*(self.res_p+sample.res_p)@(self.transl-sample.transl)
    
    def draw(self,way,colors):
        from matplotlib import pyplot as plt
        if way == "fiv":
            A = 0.33#0.001
            st,ed = [self.transl[0], self.transl[2]], [self.transl[0], self.transl[2]]#,[self.transl[0]+self.ob[0]*A, self.transl[1]+self.ob[2]*A]
            #plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], marker=".", color=colors["ob"], linewidth=0.5)
            #st = [ed[0],ed[1]]
            #ed = [st[0]+self.wo[0]*A, st[1]+self.wo[2]*A]
            st,ed = [ed[0],ed[1]],[ed[0]+self.wo[0]*A, ed[1]+self.wo[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wo"], linewidth=0.5)
            #st = [ed[0],ed[1]]
            #ed = [st[0]+self.wi[0]*A, st[1]+self.wi[2]*A]
            st,ed = [ed[0],ed[1]],[ed[0]+self.wi[0]*A, ed[1]+self.wi[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wi"], linewidth=0.5)
            #st = [ed[0],ed[1]]
            #ed = [st[0]+self.dr[0]*A, st[1]+self.dr[2]*A]
            st,ed = [ed[0],ed[1]], [ed[0]+self.dr[0]*A, ed[1]+self.dr[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["dr"], linewidth=0.5)
        else:
            pass
            #一张热力曲面图

class fild():
    def __init__(self,scene,grids,config):
        #所谓field，其实就是一系列sample，把他们结果画出来。
        L,d = grids["L"], grids["d"]
        N = int(L / d)
        self.grids = [[None for _ in range(-N,N+1,1)] for _ in range(-N,N+1,1)]
        #N = len(self.grids)
        for xi in range(-N,N+1):
            for zi in range(-N,N+1):
                self.grids[N+xi][N+zi] = sam(scene, np.array([xi*d,0,zi*d]), config)
        self.config = config
        self.scene = scene

        #calculate the p 

        for r in range(0):#N):#
            x = r
            for z in range(-r+1,r,1):
                self.grids[N+x][N+z].integral(self.grids[N+x-1][N+z]) 
            z = r
            for x in range(r,-r,-1):
                self.grids[N+x][N+z].integral(self.grids[N+x][N+z-1]) 
            x =-r
            for z in range(r,-r,-1):
                self.grids[N+x][N+z].integral(self.grids[N+x+1][N+z]) 
            z =-r
            for x in range(-r,r,1):
                self.grids[N+x][N+z].integral(self.grids[N+x][N+z+1]) 
            self.grids[N+r][N-r].integral(self.grids[N+r][N-r+1]) 
            
    def __call__(self):
        [[s(self.config) for s in g]  for g in self.grids]
                

    def draw(self,way,colors):
        # from matplotlib import pyplot as plt
        # import os
        # plt.figure(figsize=(50, 40))
        [[s.draw(way,colors) for s in g] for g in self.grids]
        
        # plt.axis('equal')
        # plt.axis('off')
        
        # plt.savefig(os.path.join(self.scene.imgDir,self.scene.scene_uid+"_"+way+".png"))
        # plt.clf()
        # plt.close()

    def show(self): #show the vector field changing through time
        #record the file names and concatenate them
        pass

    def shop(self): #show potential, rotate that field
        pass

    def update():
        pass