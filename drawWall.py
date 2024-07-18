from classes.Scne import scne
from util import fullLoadScene
"""
def recursiveRange(o,wid):
    mis,mas = o.project(wid)
    for l in o.linkIndex:
        mi,ma = recursiveRange(OBJES[LINKS[l].dst],wid)
        mis = min(mis,mi)
        mas = max(mas,ma)
    return mis,mas

def createMovements():
    #move = {id:0,length:1}
    #move = {id:0,rate:0.5}
    #what are we going to do?
    #find a legal break point. How to find the legal point?
    #scan the linked branches on this wall.
    wls = sorted([w.idx for w in WALLS if len(w.linkIndex)>0], key=lambda x:-WALLS[x].length)
    if len(wls)==0:
        return [{"id":0,"rate":0.5}, {"id":-1,"length":-0.5}]
    return [{"id":wls[0],"length":-1.5}]
    wid = 0
    while wid < len(wls)-1 and np.random.rand()<0.8:
        wid += 1
    wid = wls[wid]
    w = WALLS[wid]

    rs = [0,1]
    
    for l in w.linkIndex:
        mi,ma = recursiveRange(OBJES[LINKS[l].dst],wid)
        mi,ma = max(mi,0),min(ma,1)
        #如何将禁区mi和ma加进去
        #搜索mi所在的有效区下界，将有效区上界设置为mi。继续搜索，找到ma所在的有效区，将有效区下界设置为ma
        rss=[0.0]
        idx=1
        while idx+1<len(rs) and rs[idx+1]<mi:      #valid-lower == invalid-upper
            rss.append(rs[idx])  #valid-upper == invalid-lower
            rss.append(rs[idx+1])#valid-lower == invalid-upper
            idx+=2
        rss.append(min(mi,rs[idx]))
        while rs[idx]<ma:        #valid-upper == invalid-lower
            idx+=2
        rss.append(max(ma,rs[idx-1]))
        while idx < len(rs):
            rss.append(rs[idx])
            idx+=1
        rs=rss
    
    r = (0.5+(np.random.rand()-0.5)*0.5)*sum([rs[_+1]-rs[_] for _ in range(0,len(rs),2)])
    idx = 0
    while r > (rs[idx+1]-rs[idx]):
        r -= (rs[idx+1]-rs[idx])
        idx += 2

    return [{"id":wid,"rate":rs[idx]+r}, {"id":-1,"length":-0.5}]

"""
def test():
    for n in ["0d83ef53-4122-4678-93be-69f8b6d32c77_LivingDiningRoom-974.png"]:#os.listdir("./"):#[:20]:
        A = scne(fullLoadScene(n[:-4]),grp=True)#storeScene(n[:-4])
        A.formGraph()
        A.adjustScene([{"id":0,"rate":0.5},{"id":0,"length":-0.5}])
        A.draw("./" + n[:-4] + "test.png")#storedDraw()

if __name__ == "__main__":
    test()#main()
