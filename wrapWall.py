from classes.Scne import scne
from util import fullLoadScene
import sys

def test(n):
    A = scne(fullLoadScene(n),grp=True,cen=True)
    A.adjustGroup()#"./" + n[:-4] + "grp.png",
    A.draw(drawWall=False,drawUngroups=False,drawRoomMask=True)

if __name__ == "__main__":
    test(sys.argv[1])#main()
