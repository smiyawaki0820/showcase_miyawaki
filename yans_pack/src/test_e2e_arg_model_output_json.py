import argparse
import json
import math
import os
import sys
from os import path

import torch
import torch.nn as nn
from tqdm import tqdm

from arg_edit import *
from arg_edit import end2end_dataset, pretrained_word_vecs
from model_edit import E2EStackedBiRNN

import pandas as pd
import numpy as np

import pprint

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'


def test(out_dir, data, tag, model, model_id, thres, threshold, iterate_num, args, base_model=None):
    len_data = len(data)

    LOAD = get_bool(args.load)
    model.eval()
    if LOAD:
        base_model.eval()


    print('prediction mode:', model_id, tag, thres)
    file = open(out_dir + "/predict-" + tag + '-' + model_id + "-{0}".format("-".join(map(str, thres))) + "_iter" + str(iterate_num) + ".txt", "w")

    if not os.path.isfile(out_dir + "/gold.txt"):
        gold_file = open(out_dir + "/gold.txt", "w")

    labels = ["ga", "o", "ni"]
    result = {label: {"pp": 0, "np": 0, "pn": 0, "nn": 0} for label in labels}
    prob = {thres/100: {label: {"pp":0, "np":0, "pn":0} for label in labels} for thres in range(101)}
    conf_mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    pass_num = [0,0,0]

    from itertools import islice
    count = 0
    pred_count_test = 0
    for xss, yss in tqdm(data, total=len_data, mininterval=5):

        pred_count_test += len(yss)
        high_score = {}

        if LOAD:
            init, hoge = base_model(xss, torch.tensor([]), init=True)
            temp = torch.stack([i for i in init])
        else:
            temp = torch.tensor([])

        scores, passthrough = model(xss, temp)
        scores = scores[-1]
        for i in range(iterate_num):
            pass_num[i] += passthrough[i]

        for pred_no in range(len(yss)):
            predicted = scores[pred_no].cpu()
            predicted = torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data)
            import ipdb; ipdb.set_trace()

            #ys, p_id = yss[pred_no]
            ys, p_id, sent_id, file_name = yss[pred_no]
            doc_name = file_name
            out_dict = {"pred": p_id, "sent": sent_id, "file": doc_name}
            gold_dict = {"pred": p_id, "sent": sent_id, "file": doc_name}
            for label_idx, label in enumerate(labels):
                max_idx = torch.argmax(predicted[:, label_idx])
                max_score = predicted[max_idx][label_idx] - thres[label_idx]

                gold_idx = int(torch.argmax(ys == label_idx))
                if label_idx == int(ys[gold_idx]):
                    gold_dict[label] = gold_idx

                if max_score >= 0:
                    out_dict[label] = int(max_idx.data)

                # evaluation
                if label_idx not in ys:
                    if max_score < 0:
                        result[label]["nn"] += 1
                        #prob[round(max_score,2)][label]["nn"] += 1  # error
                    else:
                        result[label]["np"] += 1
                        prob[round(float(max_score), 2)][label]["np"] += 1
                        conf_mat[3][label_idx] += 1
                elif max_score >= 0:
                    conf_mat[label_idx][ys[max_idx]] += 1
                    if ys[max_idx] == label_idx:
                        result[label]["pp"] += 1
                        prob[round(float(max_score),2)][label]["pp"] += 1
                    else:
                        result[label]["np"] += 1
                        result[label]["pn"] += 1
                        prob[round(float(max_score),2)][label]["np"] += 1
                        prob[round(float(max_score),2)][label]["pn"] += 1
                else:
                    conf_mat[3][ys[max_idx]] +=1
                    result[label]["pn"] += 1
                    # prob[round(float(max_score),2)][label]["pn"] += 1


            if return_bool(args.save_json):
                out_dict = json.dumps(out_dict)
                print(out_dict, file=file)
                try:
                    gold_dict = json.dumps(gold_dict)
                    print(gold_dict, file=gold_file)
                except:
                    pass

    #print(result)
    calculate_f(result)
    print("pred_count_test", pred_count_test)
    print("pass_num ", pass_num)

    kaku = ["ga", "o", "ni", "null"]
    print("confusion matrix\n", pd.DataFrame(np.array(conf_mat).transpose(), columns=kaku, index=kaku))

    with open("result/base_prob_dist.json", "w") as js:
        json.dump(prob, js, indent=2)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath,
                        help='data path')
    parser.add_argument('--model-file', '-mf', dest='model_file', type=str,
                        help='model file path')
    parser.add_argument('--tag', '-t', dest='tag', type=str,
                        default="dev",
                        help='type of evaluation data split')
    parser.add_argument('--eval-ensemble', '-ee', dest='eval_ensemble', type=bool,
                        default=False,
                        help='threshold search mode')
    parser.add_argument('--thres', '-th', dest='thres', type=float, nargs=3,
                        default=[0.5, 0.5, 0.5],
                        help='output threshold of each argument')
    parser.add_argument('--thres-file', '-tf', dest='thres_file', type=str,
                        default="",
                        help='output threshold file')
    parser.add_argument('--batch', '-b', dest='batch_size', type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--gpu', '-g', dest='gpu', type=str,
                        default='0',
                        help='GPU ID for execution')
    parser.add_argument('--model_no', "-mn", dest='sub_model_number', type=int,
                        help='sub-model number for ensembling', default=-1)
    parser.add_argument('--model', '-m', dest='model_name', type=str,
                        default="path-pair-bin",
                        help='model name')
    parser.add_argument('--vec_e', '-ve', dest='vec_size_e', type=int,
                        default=256,
                        help='word embedding size')
    parser.add_argument('--load', '-load', dest='load', type=str,
                        default="False",
                        help='initialized vec')
    parser.add_argument('--vec_u', '-vu', dest='vec_size_u', type=int,
                        default=256,
                        help='unit vector size in rnn')
    parser.add_argument('--depth', '-dep', '-d', dest='depth', type=int,
                        default=10,
                        help='the number of hidden layer')
    parser.add_argument('--depth-path', '-dp', '-dp', dest='depth_path', type=int,
                        default=3,
                        help='the number of hidden layer')
    parser.add_argument('--depth-arg', '-da', '-da', dest='depth_arg', type=int,
                        default=2,
                        help='the number of hidden layer')
    parser.add_argument('--optimizer', '-o', dest='optim', type=str,
                        default="adagrad",
                        help='optimizer')
    parser.add_argument('--lr', '-l', dest='lr', type=float,
                        default=0.0005,
                        help='learning rate')
    parser.add_argument('--threshold', '-threshold', dest='threshold', type=float,
                        default=0.8,
                        help='threshold of score')
    parser.add_argument('--iter', '-it', dest='iter', type=int,
                        default=3,
                        help='number of iteration')
    parser.add_argument('--dropout-u', '-du', dest='drop_u', type=float,
                        default=0.1,
                        help='dropout rate of LSTM unit')
    parser.add_argument('--null_label', '-null_label', dest='null_label', type=str,
                        default="inc",
                        help='null_label')
    parser.add_argument('--no-thres', '-nt', dest='no_thres', action='store_true',
                        help='without thresholds')
    parser.add_argument('--save_json', '-save_json', dest='save_json', type=str,
                        help='save or not json file')
    parser.add_argument('--out_dir', type=str, default='result')

    parser.set_defaults(no_thres=False)
    return parser

def return_bool(s):
    return True if s == "True" else False

def set_log_file(args, tag, model_id):
    new_id = model_id if not args.eval_ensemble else model_id + "-all"
    thres = "" if args.thres_file != "" else "-{0}".format("-".join(map(str, args.thres)))
    fd = os.open(
        args.data_path + '/log/log-' + tag + '-' + new_id + thres + ".txt",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def test_batch_generator(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    sent_id = instance["sentence id"]
    doc_name = instance["file name"]
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = [int(a) for a in pas["args"]]
        ts = torch.LongTensor([int(t) for t in tokens])
        ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ts, ps], [ys, p_id], sent_id, doc_name]


def test_end2end_single_seq_instance(data, batch_generator):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence))
            xss, yss, sent_id, file_name = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), yss, sent_id, file_name]


def calculate_f(result):
    labels = ["ga", "o", "ni"]
    keys = ["pp", "pn", "np", "nn"]

    result["all"] = {key: 0 for key in keys}
    for label in labels:
        for key in keys:
            result["all"][key] += result[label][key]

    for label, value in result.items():
        p = value['pp'] / (value['pp'] + value['np']) if value['pp'] > 0 else 0.0
        r = value['pp'] / (value['pp'] + value['pn']) if value['pp'] > 0 else 0.0
        f = 2 * p * r / (p + r) if value['pp'] > 0 else 0.0
        print("{}:\tprec: {}, recall: {}, f1: {}".format(label, round(p * 100, 2), round(r * 100, 2), round(f * 100, 2)))


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    print(os.path.basename(args.model_file))
    model_id, _ = os.path.splitext(os.path.basename(args.model_file))
    print(args.model_name, model_id)

    gpu_id = 0
    print("gpu:", gpu_id)

    torch.manual_seed(args.sub_model_number)
    tag = args.tag
    LOAD = return_bool(args.load)
    # set_log_file(args, tag, model_id)

    print(args.depth, args.depth_path, args.depth_arg)

    model: nn.Module = []
    base_model: nn.Module = []
    data = end2end_dataset(args.data_path + "/{}.json".format(args.tag), 100)

    if args.model_name == 'e2e-stack':
        #data = list(test_end2end_single_seq_instance(data, test_batch_generator))
        data = NtcBucketIterator(data, args.batch_size, decode=True)

        word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt", args.vec_size_e)

        model = E2EStackedBiRNN(args.vec_size_u, args.depth, 4, word_embedding_matrix, args.drop_u, True, args.iter, args.threshold, args.null_label)
        base_model = E2EStackedBiRNN(args.vec_size_u, args.depth, 4, word_embedding_matrix, args.drop_u, True, 1, 0.0, args.null_label)

    model.load_state_dict(torch.load(args.model_file))
    base_id = get_load_model(args.model_file)
    if get_bool(args.load):
        import ipdb; ipdb.set_trace()
        print("base", pycolor.RED + base_id + pycolor.END)
        base_model.load_state_dict(torch.load(base_id))

    if torch.cuda.is_available():
        model = model.cuda()
        base_model = model.cuda()
        with torch.cuda.device(gpu_id):
            test(args.out_dir, data, tag, model, model_id, args.thres, args.threshold, args.iter, args, base_model)
    else:
        test(args.out_dir, data, tag, model, model_id, args.thres, args.threshold, args.iter, args, base_model)

def get_load_model(model_file):
    import re
    return "_th0.0_it1_".join(re.split("_th0\.._it._", model_file)).rsplit("_", 2)[0] + "_preFalse_loss-sum.h5"

def get_bool(s):
    return True if s == "True" else False

if __name__ == '__main__':
    run()
