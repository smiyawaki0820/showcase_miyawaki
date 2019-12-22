import argparse
import pandas as pd
import numpy as np

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-r', dest='res', nargs='*')
    parser.set_defaults(no_thres=False)
    return parser

def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    res = [float(i) for i in args.res]

    print(np.average(res), np.std(res))

if __name__ == "__main__":
    run()
