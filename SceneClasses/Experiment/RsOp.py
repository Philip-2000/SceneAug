class rsop():
    def __init__(self, name):
        self.name = name
        from .Tmer import tmer
        self.timer = tmer()
        self.adj,self.vio,self.fit,self.dif = [],[],[],[]

    def clear(self):
        from .Tmer import tmer
        self.timer = tmer()
        self.adj,self.vio,self.fit,self.dif = [],[],[],[]

    def append(self, **kwargs):
        if "step" in kwargs:
            try:
                assert len(self) == kwargs["step"]
            except:
                print("name=%s, len=%d, step=%d"%(self.name,len(self),kwargs["step"]))
                raise AssertionError

        _ = self.adj.append(kwargs["adj"].Norm()) if "adj" in kwargs else None
        _ = self.vio.append(kwargs["vio"]) if "vio" in kwargs else None
        _ = self.fit.append(kwargs["fit"]) if "fit" in kwargs else None
        _ = self.dif.append(kwargs["dif"]) if "dif" in kwargs else None
    
    def __getitem__(self, keyidx):
        if keyidx[0] == "adj":
            return self.adj[keyidx[1]]
        elif keyidx[0] == "vio":
            return self.vio[keyidx[1]] / (4*int(self.name[-1]))
        elif keyidx[0] == "fit":
            return self.fit[keyidx[1]]
        elif keyidx[0] == "dif":
            return self.dif[keyidx[1]]
        else:
            raise KeyError("keyidx[0] should be one of [adj, vio, fit, dif]")

    def __len__(self):
        return len(self.vio)

    def save(self):
        #self.adj[0],self.dif[0] = self.adj[1],self.dif[1]
        return { "adj":[round(a,5) for a in self.adj], "vio":[round(a,5) for a in self.vio], "fit":[round(a,5) for a in self.fit], "dif":[round(a,5) for a in self.dif], "timer":self.timer.save() }

    def load(self, dct):
        self.adj, self.vio, self.fit, self.dif = dct["adj"], dct["vio"], dct["fit"], dct["dif"]
        self.timer.load(dct["timer"])

T=1.2
class rsops():
    def __init__(self,uids,dir,dirs=[]):
        self.dir = dir
        self.dirs = dirs if len(dirs) > 0 else [dir]
        self.rsops = {}
        for d in self.dirs:
            self.rsops.update({self.connect(name,d):rsop(self.connect(name,d)) for name in uids})
        self.plots = {"adj":{"step":[],"time":[]}, "vio":{"step":[],"time":[]}, "fit":{"step":[],"time":[]}, "dif":{"step":[],"time":[]}, "times":{"line":[],"hist":[]}, "steps":{"line":[],"hist":[]}}
        self.flat_list = None

    def connect(self, name, d):
        import os
        return name+" "+os.path.basename(d)

    def __getitem__(self,name):
        return self.rsops[name if type(name) == str else name.scene_uid]

    def __len__(self):
        return len(self.rsops)

    #region: to_plot
    def to_plot(self):
        for key in ["adj","vio","fit","dif"]:
            self.plots[key]["step"] = self.step_align(key)
            self.plots[key]["time"] = self.time_align(key)
        self.plots["times"]["line"],self.plots["times"]["hist"] = self.times()
        self.plots["steps"]["line"],self.plots["steps"]["hist"] = self.steps()
    #endregion: to_plot

    #region: visualization
    def line(self, key, subkey, clear=True):
        import os, matplotlib.pyplot as plt
        data = self.plots[key][subkey]
        plt.plot([d[0] for d in data], [d[1] for d in data])
        if clear:
            plt.savefig(os.path.join(self.dir,"%s_%s.png"%(key,subkey)))
            plt.clf()

    def histogram(self, key, subkey, clear=True):
        import os, matplotlib.pyplot as plt
        data = self.plots[key][subkey]
        plt.hist(data)
        if clear:
            plt.savefig(os.path.join(self.dir,"%s_%s.png"%(key,subkey)))
            plt.clf()

    def box(self, key, subkey, clear=True):
        import os, matplotlib.pyplot as plt
        data = self.plots[key][subkey]
        plt.boxplot(data)#,showfliers=False)
        if clear:
            plt.savefig(os.path.join(self.dir,"%s_%s.png"%(key,subkey)))
            plt.clf()

    def visualizes(self):
        for key in self.plots:
            for subkey in self.plots[key]:
                if subkey == "step":
                    self.box( key, subkey )
                elif subkey == "hist":
                    self.histogram( key, subkey )
                else:
                    self.line( key, subkey )
    #endregion: visualization

    #region: save and load
    def load(self, n="origin"):
        import os,json
        if n=="origin":
            for d in self.dirs:
                dct = json.loads(open(os.path.join(d,"origin.json"),"r").read())
                for name in dct: #self.rsops[name] = rsop(name)
                    self.rsops[ self.connect(name,d) ].load(dct[name])
        elif n=="plot":
            self.plots = json.loads(open(os.path.join(self.dir,"plot.json"),"r").read())

    def save(self, n="origin"):
        import os,json
        if n=="origin":
            open(os.path.join(self.dir,"origin.json"),"w").write(json.dumps({name:self.rsops[name].save() for name in self.rsops}))
        elif n=="plot":
            open(os.path.join(self.dir,"plot.json"),"w").write(json.dumps( self.plots ))
    #endregion: save and load
    
    #region: utils
    def flat_list_gen(self):
        import itertools
        full = [ [(name,i,self.rsops[name].timer[""][i]) for i in range(len(self.rsops[name])) if self.rsops[name].timer[""][i] <= T ] for name in self.rsops]#print(full)
        self.flat_list = sorted(list(itertools.chain(*full)), key=lambda x:x[2])

    def time_align(self,key):
        if self.flat_list is None:
            self.flat_list_gen()

        #res = np.zeros(( len(flat_list) , len(self.rsops)))
        cur = [ self.rsops[name][(key,0)] for name in self.rsops]
        avg = [0 for _ in self.flat_list]#np.zeros((len(flat_list)))


        for s in range(len(self.flat_list)):
            name,i,tm = self.flat_list[s]
            assert tm >= self.flat_list[s-1][2] if s > 0 else True
            cur[list(self.rsops.keys()).index(name)] = self.rsops[name][(key,i)]
            #for jj in range(len(self)): res[s][jj] = cur[jj]
            avg[s] = sum(cur)/len(self)
        return [(self.flat_list[s][2], round(avg[s],5)) for s in range(len(self.flat_list))]

    def step_align(self,key):
        S = max([len(self.rsops[name]) for name in self.rsops])
        res = [[0 for _ in self.rsops] for _ in range(S)]
        for s in range(S):
            for n,name in enumerate(self.rsops.keys()):
                res[s][n] = self.rsops[name][(key,s)] if s < len(self.rsops[name]) else res[s-1][n]
        return res

    def times(self):
        times = sorted([self.rsops[name].timer[""].last for name in self.rsops if self.rsops[name].timer[""].last <= T], key=lambda x:-x)
        times_elbow = [(0.0,len(self)/len(self))] + list(reversed([(t,i/len(self)) for i,t in enumerate(times)]))
        return times_elbow, times

    def steps(self):
        #a list of all the steps of each rsop
        steps = sorted([len(self.rsops[name]) for name in self.rsops])
        #print(steps)
        steps_elbow = [(4,len(self)/len(self))] + [(s,(len(self)-i)/len(self)) for i,s in enumerate(steps)]
        if len(self.dirs)>1:
            from collections import Counter
            import os
            a = Counter(steps)
            print(os.path.basename(self.dir) + " 4:%.2f, 5:%.2f, 6:%.2f, 7+:%.2f "%(
                a[4]/len(self),a[5]/len(self),a[6]/len(self),sum([a[i] for i in range(7,25)])/len(self)
            ))
        return steps_elbow, steps
    #endregion: utils