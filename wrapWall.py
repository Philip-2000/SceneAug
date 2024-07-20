from classes.Scne import scne
from util import fullLoadScene

def test():
    for n in ["0d83ef53-4122-4678-93be-69f8b6d32c77_LivingDiningRoom-974.png"]:#os.listdir("./"):#[:20]:
        A = scne(fullLoadScene(n[:-4]),grp=True,cen=True)#storeScene(n[:-4])
        A.adjustGroup()
        A.draw("./" + n[:-4] + "grp.png",drawWall=False,objectGroup=True,drawUngroups=False,drawRoomMask=True)#storedDraw()
        #A.drawRoomMask("./" + n[:-4] + "mask.png")
#this graph can also be broken right?

#select a wall, a length? ||||||||| select a part of a wall, and check a length to move it?


#would door or window break by this movement? #would collision occurs? #would path still exist? 



#this tree is pre-calculated and stored? maybe.


#our augmentation based on trees?
#place the groups and wrap the walls?!!!!!!!!
#No idea anyway, fuck. 

if __name__ == "__main__":
    test()#main()

#what?

#for those not 