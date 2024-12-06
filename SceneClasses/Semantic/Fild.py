import numpy as np
from numpy.linalg import norm
class sam():
    def __init__(self,scene,transl,idx,config):
        self.transl = transl
        self.idx = (int(idx[0]*config["grid"]["b"]),int(idx[1]*config["grid"]["b"]))
        self.wo, self.wi, self.dr = scene.WALLS.optFields(self,config)
        self.ob, self.p_ob = scene.OBJES.optFields(self, None, config["object"])
        self.res_p = self.wo + self.wi
        self.res_ob = self.res_p + self.dr
        self.res = self.res_ob + self.ob
        self.p = 0

    def __call__(self, scene, config):
        self.ob, p = scene.OBJES.optFields(self, None, config["object"])
        self.res = self.res_ob + self.ob
        self.p = self.p_ob + p
        return self.res
    
    def integral(self,sample):
        raise NotImplementedError
        self.p_ob = sample.p_ob + 0.5*(self.res_p+sample.res_p)@(self.transl-sample.transl)
    
    def draw(self,way,colors,pxs=None,b=0):
        from matplotlib import pyplot as plt
        if way == "fiv":
            A = 0.33#0.001
            st,ed = [self.transl[0], self.transl[2]], [self.transl[0]+self.ob[0]*A, self.transl[2]+self.ob[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["ob"], linewidth=0.5)
            st,ed = [ed[0],ed[1]],[ed[0]+self.wo[0]*A, ed[1]+self.wo[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wo"], linewidth=0.5)
            st,ed = [ed[0],ed[1]],[ed[0]+self.wi[0]*A, ed[1]+self.wi[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["wi"], linewidth=0.5)
            st,ed = [ed[0],ed[1]], [ed[0]+self.dr[0]*A, ed[1]+self.dr[2]*A]
            plt.plot( [st[0],ed[0]], [-st[1],-ed[1]], color=colors["dr"], linewidth=0.5)
        elif way == "fih":
            cl = min(norm(self.wo)/6.,1.0) * np.array(colors["wo"]) + min(norm(self.wi)/.5,1.0) * np.array(colors["wi"]) + min(norm(self.ob)/2.,1.0) * np.array(colors["ob"])
            c = (int(cl[0] * 255.0),int(cl[1] * 255.0),int(cl[2] * 255.0))
            for x in range(b):
                for y in range(b):
                    pxs[self.idx[0]+x,self.idx[1]+y]=c
        elif way == "fip":
            return 0 # a distance?
        elif way == "fiq":
            cl = min(norm(self.wo)/6.,1.0) * np.array(colors["wo"]) + min(norm(self.wi)/.5,1.0) * np.array(colors["wi"]) + min(norm(self.ob)/2.,1.0) * np.array(colors["ob"])
            c = (int(cl[0] * 255.0),int(cl[1] * 255.0),int(cl[2] * 255.0))
            for x in range(b):
                for y in range(b):
                    pxs[self.idx[0]+x,self.idx[1]+y]=c
            

class fild():
    def __init__(self,scene,grids,config):
        #所谓field，其实就是一系列sample，把他们结果画出来。
        L,d = grids["L"], grids["d"]
        N = int(L / d)
        self.grids = [[sam(scene, np.array([xi*d,0,zi*d]), (N+xi,N+zi), config) for zi in range(-N,N+1,1)] for xi in range(-N,N+1,1)]
        # for xi in range(-N,N+1):
        #     for zi in range(-N,N+1):
        #         self.grids[N+xi][N+zi] = sam(scene, np.array([xi*d,0,zi*d]), (N+xi,N+zi), config)
        self.N = N
        self.b = grids["b"]
        self.config = config
        self.scene = scene
        #self.integral()
        from PIL import Image
        self.fih = Image.new('RGB',((2*N+1)*self.b,(2*N+1)*self.b))
        self.fiq = Image.new('RGB',((2*N+1)*self.b,(2*N+1)*self.b))
            
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

    def draw(self,way,colors):
        if way == "fiv":  # Vector Field from each points
            [[s.draw(way,colors) for s in g] for g in self.grids]
        elif way == "fih":# Heatmap of length of vectors
            [[s.draw(way,colors,self.fih.load(),self.b) for s in g] for g in self.grids]
            self.fih.save('./fih.jpg')
        elif way == "fip":# Curved Surface 
            raise NotImplementedError
            x = np.linspace(-halfSize, halfSize+1, int(2*halfSize/rate)+1)*scale
            y = np.linspace(-halfSize, halfSize+1, int(2*halfSize/rate)+1)*scale
            x, y = np.meshgrid(x, y)
            
            z = np.array(-field[3])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
            
            levels = np.linspace(0, 10, 5)**2
            
            ax.contour(x, y, z, levels, zdir='z', offset=0, cmap=plt.cm.coolwarm)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            #ax.set_zlabel('Z axis')
            ax.set_title('00c36b04-369f-4df1-9db1-b29913d2c51f_LivingDiningRoom-7050')
            plt.show()
        elif way == "fiq":# Heatmap
            raise NotImplementedError
            [[s.draw(way,colors,self.fiq.load(),self.b) for s in g] for g in self.grids]
            self.fiq.save('./fiq.jpg')

    def show(self): #show the vector field changing through time
        #record the file names and concatenate them
        pass

    def shop(self): #show potential, rotate that field
        pass

    def update():
        pass