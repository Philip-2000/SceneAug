def parse():  #Execute the pattern application experiments
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Experiment of the Pattern Manager\'s Application')
    parser.add_argument("-v","--version",required=True)#,default="merg")
    parser.add_argument("-t","--task",  default="task")
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__": #load="testings",
    import os
    args,E = parse(),None