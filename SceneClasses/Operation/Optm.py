class optm():
    def __init__(self,pmVersion=None,scene=None,ObjFlag=False,WallFlag=True):
        assert ObjFlag or WallFlag
        self.ObjOpt = objOpt(pmVersion,scene) if ObjFlag else None
        self.WallOpt = wallOpt(scene) if WallFlag else None

    def __call__(self):
        r = self.WallOpt() if self.WallOpt else None
        r = self.ObjOpt()  if self.ObjOpt  else None        

class wallOpt():
    def __init__(self,scene):
        self.scene = scene
        pass

    #region: -hyper-parameter--------#
    def load(self):
        pass

    #endregion: -hyper-parameter-----#


    #region: ---real-flow------------#
    
    #就是说，如果是我，对如果是我直接从串行角度出发去开发这个东西的话，会怎么做呢？
    #我不会遵循着并行的过程，先算这个场再算那个场，
    #我会逐个物体逐个物体地调整的，
    
    #那场还会算吗？可以算，作为中间结果输出
    

    #那永久场还会算吗？可以算啊，只是用不到啊，就像在那个并行版本里也用不到一样，行把打点方法作为超参数传进两个对象里吧？
    
    #也就是说，我们的这个类的代码结构实际上和并行版本会从根本上不相同，没必要搭建类似的框架，好欸！

    #而且对于每个物体各自optimize的过程最好是能够由这个物体的类的成员函数来亲自实现比较合适
    
    #endregion: ---real-flow---------#

    #region: -room-fields------------#
    
    #什么玩意儿啊！这玩意是这么算的吗？
    #不对啊，对吧，
    #不是这么算的？
    #
    def fields(self):
        pass
    #endregion: -room-fields---------#

    #region: --use-fields------------#
    def mix_field(self):
        pass
    #endregion: --use-fields---------#

        #region: yesyes
    def yes(self):
        #啥意思！
        #就是说，在这里构建一个和那边一模一样的串行镜像，一方面可以用，另一方面可以作为那边的debug过程。

        #从实际操作，到可视化，到数值比较接口（不用做数值比较的过程脚本，）
        #所以我们有哪些部分的代码需要构建呢
        pass
        #endregion

    #1，各种场
    #2，使用场
    #3，可视化，可视化场和可视化物体的调整结果，
    #4，数值比较接口，要不然数值比较过程放在./Experiment里面吧？相当于ExOp之外的一个东西，叫啥？不知道呀？叫CpOp.py
    #   这个东西需要在环境中同时安装了这个库里的东西和那个库里的东西。对。
    #5，实验，这个才是ExOp.py


    def __call__():
        pass

class objOpt():
    def __init__(self,pmVersion,scene):
        raise NotImplementedError
        from .Patn import patternManager as PM
        self.PM = PM(pmVersion) if pmVersion else None
        self.scene = scene
        pass

    def __call__():
        raise NotImplementedError