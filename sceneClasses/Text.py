from Wall import *
from Obje import *

class text():
    def __init__(self):
        pass
    
    def objTypeForm(text):
        
        return text

    def parse(self, sens):

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
