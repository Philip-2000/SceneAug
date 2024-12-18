from math import exp
class static_shdl():
    def __init__(self,v,mode):
        self.v = v

    def __call__(self,s):
        return self.v

class linear_shdl():
    def __init__(self,k,v,mode):
        self.k, self.v = k,v

    def __call__(self,s):
        return max(self.v+self.k*s,0)
    
class quad_shdl():
    def __init__(self,k,v,mode):
        self.k, self.v = k,v

    def __call__(self,s):
        return self.v+self.k*s*s

class exp_dn_shdl():
    def __init__(self,r0,lda,mode,rinf=0.0):
        self.r0, self.lda, self.rinf = r0,lda,rinf

    def __call__(self,s):
        return self.rinf + (self.r0-self.rinf)*exp(-self.lda*s)
    
class exp_up_shdl():
    def __init__(self,lda,rinf,mode,r0=0.0):
        self.r0, self.lda, self.rinf = r0,lda,rinf

    def __call__(self,s):
        return self.r0 + (self.rinf-self.r0)*(1-exp(-self.lda*s)) 
    
def shdl_factory(**kwargs):
    if kwargs["mode"] == "static":
        return static_shdl(**kwargs)
    elif kwargs["mode"] == "linear":
        return linear_shdl(**kwargs)
    elif kwargs["mode"] == "quad":
        return quad_shdl(**kwargs)
    elif kwargs["mode"] == "exp_dn":
        return exp_dn_shdl(**kwargs)
    elif kwargs["mode"] == "exp_up":
        return exp_up_shdl(**kwargs)
    else:
        raise ValueError("Unknown shdl type: %s" % kwargs["mode"])