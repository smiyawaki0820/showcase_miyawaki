# -*- coding: utf-8 -*-

import sys
sys.path.append("./src")
sys.path.append("~")
sys.path.append("~/local")
sys.path.append("~/local/bin")

from mecab import parsing
from src.usrin import *
import ipdb
import argparse
import time

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-id", "--model_id", dest='model_id', type=str,
            default="model-e2e-stack_ve256_vu256_depth6_adam_lr0.002_du0.1_dh0.0_True_size60_sub1127_th0.5_it3_rs2016_preFalse_loss-sum",
            help="model_id")
    parser.add_argument("--print", dest='PRINT', type=str,
            default="False",
            help="print details")
    parser.set_defaults(no_thres=False)
    return parser

def semantic_analysis(text:str):
    """
    input:  row text
    - mecab parsing
    - ner
    - pas

    output: json
    """

    parser = create_arg_parser()
    args = parser.parse_args()
    PARSE, parse_list = parsing(text)
    if args.model_id.startswith("result/"):
        model_id = args.model_id.split("/")[-1]
        if args.model_id.endswith("h5"):
            model_id = model_id[:-3]
    else:
        if args.model_id.endswith("h5"):
            model_id = args.model_id[:-3]
        else:
            model_id = args.model_id

    if args.PRINT: print(args.model_id)

    predicate_argument_structure(text, PARSE, parse_list, model_id, get_bool(args.PRINT))


def get_bool(s):
    return True if s == "True" else False

#def run():
#    text = str(input())
#    print(" ".join([i[7] for i in parsing(text)[:-1]]))
#    print(parsing(text))
#    pred_list = [1 if "動詞" in i else 0for i in parsing(text)]
#    print(pred_list)

if __name__ == "__main__":
    print("> ")
    semantic_analysis(str(input()))

    """
    for line in open("exa.txt"):
        try:
            semantic_analysis(line)
            time.sleep(5)
        except:
            pass
    """
    #semantic_analysis("私は数学を勉強する．")

