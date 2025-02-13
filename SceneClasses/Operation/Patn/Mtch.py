class mtch():
    def __init__(self, text, pm):
        self.text = text
        self.pm = pm
        self.records = [] #list of triplets: (text, oid, nid)

    def check_add(self, text, idx, nid):
        for r in self.text.rels:
            if r.a.text == text and r.a.idx == idx and r.b.nid != -1:
                if r.check(nid, r.b.nid, self.pm) == False: return False
            elif r.b.text == text and r.b.idx == idx and r.a.nid != -1:
                if r.check(r.a.nid, nid, self.pm) == False: return False
            else:
                continue
        return True
    
    def __str__(self):
        return '->'.join(["(%d)%s n(%d)"%(r[1],r[0],r[2]) for r in self.records])

    def __add__(self, ton):
        from copy import deepcopy as copy
        M = mtch(self.text, self.pm)
        M.records = [(copy(r[0]),copy(r[1]),copy(r[2])) for r in self.records] + [ton]
        return M
    
    @property
    def flags(self):
        return ["(%d)%s"%(r[1],r[0]) for r in self.records]

    def cover(self, MTCH=None):
        flags_set = set(self.flags)
        if MTCH is None: return len(set(self.text.flags) - flags_set) < 1
        flag_set = set(MTCH.flags)
        if flags_set & flag_set != 0: return False
        return len(set(self.text.flags) - ( flags_set | flag_set )) < 1

    def utilize(self,scene,center,ori):
        ancs = sorted(self.pm[self.records[-1][2]].ancs)
        for r in self.records: assert r[2] in ancs
        for i in ancs: self.pm.exp_object(i,scene,0.0,center,ori)#return scene
