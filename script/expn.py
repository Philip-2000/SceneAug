#Execute the experiments
from SceneClasses.ExPn import *

def parse(argv):
    import argparse
    parser = argparse.ArgumentParser(prog='ProgramName')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__": #load="testings",
    import sys
    args=parse(sys.argv[1:])