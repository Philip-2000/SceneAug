from . import OBJES
from . import WALLS

#LINKS=[]
class link():
    def __init__(self, src, dst, idx=None):
        #type?
        self.src=src
        self.dst=dst
        self.idx=idx
        pass
    
class objLink(link):
    def __init__(self, src, dst, idx):
        super(objLink,self).__init__(src, dst, idx)
        OBJES[self.src].linkIndex.append(idx)
        OBJES[self.dst].destIndex.append(idx)
        pass

    def adjust(self, movement):
        #adjust the OBJES[dst] according to the OBJES[src] OR WALLS[src]'s movement
        #
        OBJES[self.dst].adjust(movement)
        pass

    def arrow(self):
        return OBJES[self.src].translation, OBJES[self.dst].translation

class walLink(link):
    def __init__(self, src, dst, idx, dstTranslation):
        super(walLink,self).__init__(src, dst, idx)
        WALLS[self.src].linkIndex.append(idx)
        OBJES[self.dst].destIndex.append(idx)

        self.rate = 0
        self.dist = 0
        self.update(dstTranslation)

        pass

    def update(self, dstTranslation):
                
        w = WALLS[self.src]
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
        OBJES[self.dst].adjust(movement)
        pass

    def modify(self, oldp, oldq, oldn):
        dstTranslation = self.dist * oldn + self.rate * (oldq - oldp) + oldp 

        self.update(dstTranslation)
        pass

    def arrow(self):
        return self.rate*WALLS[self.src].q + (1-self.rate)*WALLS[self.src].p, OBJES[self.dst].translation