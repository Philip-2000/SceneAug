from .Wall import *
from .Obje import *

class text():
    def __init__(self):
        pass
    
    def objTypeForm(text):
        
        return text

    def parse(self, sens):
        formatStr = [" is on the left of ", " is on the left of ", " is in front of ", " is behind ", " is close to "]
        counts=[0 for _ in object_types]
        nums = {"a":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        pairs = []
        for sen in sens:
            if sen.find("has"):
                sen = sen[sen.find("has")+3:]
                obs = sen.split(',')
                for ob in obs:
                    o = ob
                    if ob.find("and")!=-1:
                        p = ob[:ob.find("and")]
                        ns = p.split(" ")
                        counts[object_types.index(self.objTypeForm(ns[1]))] += nums[ns[0]]
                        o = ob[ob.find("and")+3:]
                    ns = o.split(" ")
                    counts[object_types.index(self.objTypeForm(ns[1]))] += nums[ns[0]]


            for fms in formatStr:
                if sen.find(fms) != -1:
                    a = self.objTypeForm(sen[:sen.find(fms)])
                    b = self.objTypeForm(sen[:sen.find(fms)+len(fms)])
                    pairs.append([a,fms,b])
        
        pass

import argparse,sys
def parse(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-v','--verbose', default=0)
    args = parser.parse_args(argv)
    return args

sens = []

if __name__ == "__main__":
    args = parse()
    tF = text()
    tF.parse(sens)
