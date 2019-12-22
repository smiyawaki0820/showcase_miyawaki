import argparse
import csv
import pandas as pd

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('f',help='csv file')
    parser.set_defaults(no_thres=False)
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    FILE = args.f
    df = pd.read_table(FILE)
    print(df[df["Unnamed: 0"] == "all"]["f1"].max())


if __name__ == "__main__":
    run()
