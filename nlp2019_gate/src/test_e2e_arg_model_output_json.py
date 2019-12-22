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


def test(out_dir, data, tag, model, model_id, thres, threshold, iterate_num):
    len_data = len(data)

    model.eval()

    print('prediction mode:', model_id, tag, thres)
    file = open(out_dir + "/predict-" + tag + '-' + model_id + "-{0}".format("-".join(map(str, thres))) + "iter_" + str(iterate_num) + ".json", "w")

    labels = ["ga", "o", "ni"]
    result = {label: {"pp": 0, "np": 0, "pn": 0, "nn": 0} for label in labels}

    from itertools import islice
    count = 0
    pred_count_test = 0
    for xss, yss, sent_id, file_name in tqdm(data, total=len_data, mininterval=5):

        pred_count_test += len(yss)
        temp = torch.zeros(len(yss), len(yss[0][0]), 4)
        if torch.cuda.is_available():
          temp = temp.cuda()
        high_score = {}
        print(file_name[0], sent_id[0], sep=' ', end='\n', file=open('./result/edit/hoge/test_'+model_id+'_test-judge.txt', 'a'))

        ### FOR ITER
        scores = model(xss, yss, temp, iterate_num, threshold)
        scores = scores[-1]
        #for t in range(iterate_num):

        #  if high_score:
        #    for b, w in high_score.items():
        #      for w_i, l in w.items():
        #        temp[b, w_i, :] = l

        #  ### SCORE
        #  scores = model(xss, temp, iterate_num, threshold)
        #  for batch_idx, batch in enumerate(scores):
        #    batch_high_score = {}
        #    for word_idx in range(batch.size(0)):

        #      pred = int(torch.argmax(batch[word_idx]))
        #      gold = int(yss[batch_idx][0][word_idx])

        #      if pred <= 2:
        #        ## IF np (pred<=2, gold!=pred)
        #        if not gold == pred:
        #          print(t, batch_idx, word_idx, 'np', pred, gold, sep=' ', end='\n', file=open('./result/edit/hoge/test_'+model_id+'_test-judge.txt', 'a'))

        #        ## IF pp
        #        else:
        #          print(t, batch_idx, word_idx, 'pp', pred, gold, sep=' ', end='\n', file=open('./result/edit/hoge/test_'+model_id+'_test-judge.txt', 'a'))
        #      ## IF pn (gold<=2, pred=gold)
        #      elif pred == 3 and gold <= 2:
        #        print(t, batch_idx, word_idx, 'pn', pred, gold, sep=' ', end='\n', file=open('./result/edit/hoge/test_'+model_id+'_test-judge.txt', 'a'))

        #      if pred >= math.log(threshold) and pred <= 2:
        #        temp[batch_idx, word_idx, :] = batch[word_idx]
        #        batch_high_score[word_idx] = batch[word_idx]

        #    if not high_score.get(batch_idx):
        #      if batch_high_score:
        #        high_score[batch_idx] = batch_high_score

        for pred_no in range(len(yss)):
            predicted = scores[pred_no].cpu()
            predicted = torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data)

            ys, p_id = yss[pred_no]
            doc_name = file_name[0]
            out_dict = {"pred": p_id, "sent": sent_id[0], "file": doc_name}
            for label_idx, label in enumerate(labels):
                max_idx = torch.argmax(predicted[:, label_idx])
                max_score = predicted[max_idx][label_idx] - thres[label_idx]
                if max_score >= 0:
                    out_dict[label] = int(max_idx.data)

                # evaluation
                if label_idx not in ys:
                    if max_score < 0:
                        result[label]["nn"] += 1
                    else:
                        result[label]["np"] += 1
                elif max_score >= 0:
                    if ys[max_idx] == label_idx:
                        result[label]["pp"] += 1
                    else:
                        result[label]["np"] += 1
                        result[label]["pn"] += 1
                else:
                    result[label]["pn"] += 1

                if count <= 0:
                    f = open('./out_score.txt', 'a')
                    print('predicted', predicted, 'ys', ys, 'p_id', p_id, 'sent_id', sent_id[0], 'doc_name', doc_name, sep='\n', end='\n\n', file=f)
                    print('label_idx', label_idx, 'label', label, 'max_idx', max_idx, 'max_score', max_score, sep='\n', end='\n\n', file=f)
                    print('xss', xss, 'yss', [yss, sent_id[0], doc_name], sep='\n', end='\n\n', file=f)
                    print('scores', scores, sep='\n', end='\n', file=f)

                    count += 1
                    f.close()

            out_dict = json.dumps(out_dict)
            print(out_dict, file=file)

    from pprint import pprint
    pprint(result)
    calculate_f(result)
    print("pred_count_test", pred_count_test)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath,
                        help='data path')
    parser.add_argument('--model-file', '-mf', dest='model_file', type=str,
                        help='model file path')
    parser.add_argument('--tag', '-t', dest='tag', type=str,
                        default="test",
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
    parser.add_argument('--no-thres', '-nt', dest='no_thres', action='store_true',
                        help='without thresholds')

    parser.add_argument('--out_dir', type=str, default='result')

    parser.set_defaults(no_thres=False)
    return parser


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

    print("############", args.data_path + "/{}.json".format(args.tag), flush=True)
    torch.manual_seed(args.sub_model_number)
    tag = args.tag
    # set_log_file(args, tag, model_id)

    print(args.depth, args.depth_path, args.depth_arg)

    model: nn.Module = []
    data = end2end_dataset(args.data_path + "/{}.json".format(args.tag), 100)

    if args.model_name == 'e2e-stack':
        data = list(test_end2end_single_seq_instance(data, test_batch_generator))

        word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt", args.vec_size_e)

        model = E2EStackedBiRNN(args.vec_size_u, args.depth, 4, word_embedding_matrix, args.drop_u, True)

    model.load_state_dict(torch.load(args.model_file))

    if torch.cuda.is_available():
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            test(args.out_dir, data, tag, model, model_id, args.thres, args.threshold, args.iter)
    else:
        test(args.out_dir, data, tag, model, model_id, args.thres, args.threshold, args.iter)


if __name__ == '__main__':
    run()
