from matplotlib import pyplot as plt
common_links={
    "Dining Chair":["Dining Table"],
    "Chaise Longue Sofa":["Coffee Table"],
    "Coffee Table":["Lazy Sofa","Three-seat / Multi-seat Sofa","Loveseat Sofa","L-shaped Sofa"],
    "Dressing Chair":["Dressing Table"],
    "Pendant Lamp":["King-size Bed", "Kids Bed", "Single bed", "Coffee Table", "Dining Table"],
    "Nightstand":["King-size Bed", "Kids Bed", "Single bed"]
}


#LINKS=[]
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
        plt.plot([dst[0]], [-dst[2]], marker="x", color=self.color)
        plt.plot([src[0], dst[0]], [-src[2], -dst[2]], marker=".", color=self.color)

class objLink(link):
    def __init__(self, src, dst, idx, scne=None, color="gray"):
        super(objLink,self).__init__(src, dst, idx, scne, color)
        self.scne.OBJES[self.src].linkIndex.append(idx)
        self.scne.OBJES[self.dst].destIndex.append(idx)
        pass

    def adjust(self, movement):
        #adjust the OBJES[dst] according to the OBJES[src] OR WALLS[src]'s movement
        #
        self.scne.OBJES[self.dst].adjust(movement)
        pass

    def arrow(self):
        return self.scne.OBJES[self.src].translation, self.scne.OBJES[self.dst].translation

class walLink(link):
    def __init__(self, src, dst, idx, dstTranslation, scne=None, color="gray"):
        super(walLink,self).__init__(src, dst, idx, scne, color)
        self.scne.WALLS[self.src].linkIndex.append(idx)
        self.scne.OBJES[self.dst].destIndex.append(idx)

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
        self.scne.OBJES[self.dst].adjust(movement)
        pass

    def modify(self, oldp, oldq, oldn):
        dstTranslation = self.dist * oldn + self.rate * (oldq - oldp) + oldp 

        self.update(dstTranslation)
        pass

    def arrow(self):
        return self.rate*self.scne.WALLS[self.src].q + (1-self.rate)*self.scne.WALLS[self.src].p, self.scne.OBJES[self.dst].translation