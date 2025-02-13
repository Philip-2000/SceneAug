import numpy as np
from numpy.linalg import norm
class sam():
    def __init__(self,scene,transl,idx,config):
        self.transl = transl
        self.idx = (int(idx[0]*config["grid"]["b"]),int(idx[1]*config["grid"]["b"]))
        wop, wip, self.dr = scene.WALLS.optFields(self, None, config)
        self.wo, self.wop, self.wi, self.wip = wop[0], wop[1], wip[0], wip[1]
        self.ob, self.obp = scene.OBJES.optFields(self, None, config["object"])
        # self.res_p = self.wo + self.wi
        # self.res_ob = self.res_p + self.dr
        # self.res = self.res_ob + self.ob
        self.res_ob = self.wo + self.wi + self.dr 
        self.p_ob  = self.wop + self.wip
        self.p = self.wop+self.wip+self.obp

    def __call__(self, scene, config):
        self.ob, self.obp = scene.OBJES.optFields(self, None, config["object"])
        self.res = self.res_ob + self.ob
        self.p = self.p_ob + self.obp
        return self.res
    
    def integral(self,sample):
        raise NotImplementedError
        self.p_ob = sample.p_ob + 0.5*(self.res_p+sample.res_p)@(self.transl-sample.transl)
    
    def draw(self,way,colors,pxs=None,b=0):
        from matplotlib import pyplot as plt
        if way == "fiv":
            A = 0.33#0.001
            st,ed = [self.transl[0], self.transl[2]], [self.transl[0], self.transl[2]]
            if "ob" in colors:
                st,ed = [ed[0],ed[1]],[ed[0]+self.ob[0]*A, ed[1]+self.ob[2]*A]
                plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["ob"], linewidth=0.5)
            if "wo" in colors:
                st,ed = [ed[0],ed[1]],[ed[0]+self.wo[0]*A, ed[1]+self.wo[2]*A]
                plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wo"], linewidth=0.5)
            if "wi" in colors:
                st,ed = [ed[0],ed[1]],[ed[0]+self.wi[0]*A, ed[1]+self.wi[2]*A]
                plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wi"], linewidth=0.5)
            if "dr" in colors:
                st,ed = [ed[0],ed[1]], [ed[0]+self.dr[0]*A, ed[1]+self.dr[2]*A]
                plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["dr"], linewidth=0.5)
        elif way == "fih":
            cl = min(norm(self.wo)/6.,1.0) * np.array(colors["wo"]) + min(norm(self.wi)/.5,1.0) * np.array(colors["wi"]) + min(norm(self.ob)/2.,1.0) * np.array(colors["ob"])
            c = (int(cl[0] * 255.0),int(cl[1] * 255.0),int(cl[2] * 255.0))
            for x in range(b):
                for y in range(b):
                    pxs[self.idx[0]+x,self.idx[1]+y]=c
        elif way == "fip":
            self.p = self.wop+self.wip+self.obp
            return self.p
        elif way == "fiq":
            cl = min(norm(self.wop)/18.,1.0) * np.array(colors["wo"]) + min(norm(self.wip)/.5,1.0) * np.array(colors["wi"]) + min(norm(self.obp)/2.,1.0) * np.array(colors["ob"])
            c = (int(cl[0] * 255.0),int(cl[1] * 255.0),int(cl[2] * 255.0))
            for x in range(b):
                for y in range(b):
                    pxs[self.idx[0]+x,self.idx[1]+y]=c
            

class fild():
    def __init__(self,scene,grids,config):
        #所谓field，其实就是一系列sample，把他们结果画出来。
        L,d,b = grids["L"], grids["d"], grids["b"]
        N = int(L / d)
        self.grids = [[sam(scene, np.array([xi*d,0,zi*d]), (N+xi,N+zi), config) for zi in range(-N,N+1,1)] for xi in range(-N,N+1,1)]
        self.N,self.d,self.b = N,d,b
        self.config = config
        self.scene = scene
        #self.integral()
        from PIL import Image
        if "fih" in config["vis"]:
            self.fih = Image.new('RGB',((2*N+1)*b,(2*N+1)*b))
        if "fiq" in config["vis"]:
            self.fiq = Image.new('RGB',((2*N+1)*b,(2*N+1)*b))
        if "fip" in config["vis"]:
            self.x, self.y = np.meshgrid(np.linspace(-self.N, self.N+1, int(2*self.N)+1)*self.d, np.linspace(-self.N, self.N+1, int(2*self.N)+1)*self.d)
            from matplotlib import pyplot as plt
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X axis')
            self.ax.set_ylabel('Y axis')
            self.ax.set_title('fip')
            
    def __call__(self):
        [[s(self.scene,self.config) for s in g]  for g in self.grids]

    def integral(self):
        raise NotImplementedError
        for r in range(self.N):#
            x = r
            for z in range(-r+1,r,1):
                self.grids[self.N+x][self.N+z].integral(self.grids[self.N+x-1][self.N+z]) 
            z = r
            for x in range(r,-r,-1):
                self.grids[self.N+x][self.N+z].integral(self.grids[self.N+x][self.N+z-1]) 
            x =-r
            for z in range(r,-r,-1):
                self.grids[self.N+x][self.N+z].integral(self.grids[self.N+x+1][self.N+z]) 
            z =-r
            for x in range(-r,r,1):
                self.grids[self.N+x][self.N+z].integral(self.grids[self.N+x][self.N+z+1]) 
            self.grids[self.N+r][self.N-r].integral(self.grids[self.N+r][self.N-r+1])             

    def draw(self,way,colors,ts):
        import os
        if way == "fiv":  # Vector Field from each points
            [[s.draw(way,colors) for s in g] for g in self.grids]
        elif way == "fih":# Heatmap of length of vectors
            [[s.draw(way,colors,self.fih.load(),self.b) for s in g] for g in self.grids]
            self.fih.save(os.path.join(self.scene.imgDir,way+"_"+str(ts)+".png"))
        elif way == "fip":# Curved Surface 
            raise NotImplementedError
            #物体包围盒矩形从房间场从外向里滑下去那个视频还做吗？咋做
            #然后鸡你太美那个视频咋做呀。那个视频最关键的一项差别就是，它的墙壁也是可以动的，所以房间场在各个时刻都在变，还考虑上一帧的影响（粘滞）吗？不一定，但可以试一试？。
            p = np.array(-[[s.draw(way,colors) for s in g] for g in self.grids])
            surf = self.ax.plot_surface(self.x, self.y, p, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
            #ax.contour(self.x, self.y, p, np.linspace(0, 10, 5)**2, zdir='z', offset=0, cmap=plt.cm.coolwarm)
            plt.show()
        elif way == "fiq":# Heatmap
            [[s.draw(way,colors,self.fiq.load(),self.b) for s in g] for g in self.grids]
            self.fiq.save(os.path.join(self.scene.imgDir,way+"_"+str(ts)+".png"))

    def show(self): #show the vector field changing through time
        #record the file names and concatenate them
        pass

    def shop(self): #show potential, rotate that field
        pass

    def update():
        pass