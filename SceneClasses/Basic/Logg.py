class logg():
    def __init__(self,walls,id):
        self.id=id
        self.Walls=walls

class rtlg(logg):
    def __init__(self,walls,id,r,notOp=False):
        super(rtlg,self).__init__(walls,id)
        self.r=r
        if not notOp:
            self.operate()

    @classmethod
    def fromStr(cls,walls,string,notOp):
        return cls(walls,int(string[:string.find(':')]),float(string[string.find(':')+2:-1]),notOp)

    def operate(self):
        self.Walls.breakWall(self.id,self.r)

    def __str__(self):
        return "%d::%.2f\n"%(self.id,self.r)

class mvlg(logg):
    def __init__(self,walls,id,length,lower,upper,notOp=False):
        super(mvlg,self).__init__(walls,id)
        self.length=length
        self.lower=lower
        self.upper=upper#print(self)
        if not notOp:
            self.operate()

    @classmethod
    def fromStr(cls,walls,string,notOp):
        return cls(walls,int(string[:string.find('=')]),float(string[string.find('<')+2:string.find('>')]),float(string[string.find('=')+2:string.find('<')]),float(string[string.find('>')+2:-1]),notOp)

    def operate(self):#print(self.length)
        self.Walls[self.id].mWall(self.length)

    def __str__(self):
        return "%d==%.3f<<%.3f>>%.3f\n"%(self.id,self.lower,self.length,self.upper)

class dllg(logg):
    def __init__(self,walls,id,notOp=False):
        super(dllg,self).__init__(walls,id)
        if not notOp:
            self.operate()
    
    @classmethod
    def fromStr(cls,walls,string,notOp):
        return cls(walls,int(string[:string.find(' ')]),notOp)

    def operate(self):
        self.Walls[self.id].squeeze()#self.Walls.squeezeWall(self.id)

    def __str__(self):
        return "%d delete\n"%(self.id)

def distribute(walls,string,notOp=False):
    if string.find("delete")>0:
        return dllg.fromStr(walls,string,notOp)
    if string.find("<<")>0:
        return mvlg.fromStr(walls,string,notOp)
    if string.find("::")>0:
        return rtlg.fromStr(walls,string,notOp)
    raise NotImplementedError
    