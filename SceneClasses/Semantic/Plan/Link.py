class link():
    def __init__(self, src, dst, idx=None, scne=None, color="gray"):
        #type?
        self.src=src
        self.dst=dst
        self.idx=idx
        self.scne=scne
        self.color=color

    def arrow(self):
        pass
    
    def draw(self):
        src,dst = self.arrow()
        from matplotlib import pyplot as plt
        plt.plot([dst[0]], [-dst[2]], marker="x", color=self.color)
        plt.plot([src[0], dst[0]], [-src[2], -dst[2]], marker=".", color=self.color)

class objLink(link):
    def __init__(self, src, dst, idx=None, scne=None, color="gray"):
        super(objLink,self).__init__(src, dst, idx, scne, color)
    
    def sets(self):
        self.scne[self.src].linkIndex.append(self.idx)
        self.scne[self.dst].destIndex.append(self.idx)

    def adjust(self, movement):
        #adjust the OBJES[dst] according to the OBJES[src] OR WALLS[src]'s movement
        #
        self.scne[self.dst].adjust(movement)
        pass

    def arrow(self):
        return self.scne[self.src].translation, self.scne[self.dst].translation
    
    def toLinkJson(self):
        return {"src":self.src, "dst":self.dst}

class walLink(link):
    def __init__(self, src, dst, idx=None, dstTranslation=None, scne=None, color="gray"):
        raise NotImplementedError("walLink unused right now")
        super(walLink,self).__init__(src, dst, idx, scne, color)
        self.scne.WALLS[self.src].linkIndex.append(idx)
        self.scne[self.dst].destIndex.append(idx)

        self.rate = 0
        self.dist = 0
        self.update(dstTranslation)

        pass

    def update(self, dstTranslation):
                
        w = self.scne.WALLS[self.src]
        a = dstTranslation-w.p
        a[1]=0

        dist = a@w.n
        b = a - dist*w.n
        rate = (b**2).sum()**0.5 / w.length

        if rate < 0 or rate > 1 :#or dist < 0:
            print("wallLink.update error", dist, rate)
        else:
            self.dist = dist
            self.rate = rate

        pass
    
    def adjust(self, movement):
        #adjust the OBJES[dst] according to the OBJES[src] OR WALLS[src]'s movement
        #
        self.scne[self.dst].adjust(movement)
        pass

    def modify(self, oldp, oldq, oldn):
        dstTranslation = self.dist * oldn + self.rate * (oldq - oldp) + oldp 

        self.update(dstTranslation)
        pass

    def arrow(self):
        return self.rate*self.scne.WALLS[self.src].q + (1-self.rate)*self.scne.WALLS[self.src].p, self.scne[self.dst].translation
    
class links():
    def __init__(self, scene):
        self.scne = scene
        self.LINKS = []
        
    def __iter__(self):
        return iter(self.LINKS)
    
    def __getitem__(self, i):
        return self.LINKS[i]
    
    def __len__(self):
        return len(self.LINKS)

    def clear(self):
        self.LINKS = []
    
    def append(self, lnk):
        lnk.idx = len(self.LINKS)
        lnk.scne = self.scne
        lnk.sets()
        self.LINKS.append(lnk)
    
    def draw(self):
        for l in self.LINKS: l.draw()
    
    def toLinksJson(self,rsj):
        rsj["links"] = [l.toLinkJson() for l in self]
        return rsj