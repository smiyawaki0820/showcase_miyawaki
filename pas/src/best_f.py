import argparse
import csv
import pandas as pd

from utils.utils import get_max_score


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('fname',help='csv file')
    parser.add_argument('--case', dest='case', type=str, default='all', help='case')
    parser.add_argument('--stype', dest='stype', type=str, default='f1', help='score type')
    parser.set_defaults(no_thres=False)
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    print(get_max_score(args.fname, case=args.case, stype=args.stype))

if __name__ == "__main__":
    run()
