from .Syth import syth
class copl(syth):
    def __init__(self,pm,scene,nm="test",v=0):
        super(copl,self).__init__(pm,scene,self.__class__.__name__,nm,v)
        raise NotImplementedError

    def uncond(self):#可以类似于rearrange
        return self.scene

    def textcond(self):
        return self.scene

    def roomcond(self):
        return self.scene
