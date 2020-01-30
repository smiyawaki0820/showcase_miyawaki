import json
import math
import os
import sys
from os import path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.arg_edit import end2end_dataset, pretrained_word_vecs
from src.model_edit import E2EStackedBiRNN

import numpy as np

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


def test(out_dir, xss, tag, model, model_id, thres, threshold, iterate_num, load, tokenized_list, src, PRINT):

    model.eval()

    from itertools import islice

    SCORES, _ = model(xss, torch.tensor([]))

    #print(scores)
    kaku = {0:"ga", 1:"o", 2:"ni", 3:"null"}

    verb = []
    for b_idx, batch in enumerate(xss[1]):
        for word_idx, binary_pred in enumerate(batch):
            if binary_pred == 1.:
                verb.append(tokenized_list[word_idx])

    analyze_dic = {}

    for i, scores in enumerate(SCORES):
        print("\n\n== iter {} ==".format(i))
        temp = analyze_dic
        analyze_dic = {}
        for b_idx, batch in enumerate(scores):
            pred_dic = {}
            predicted = scores[b_idx].cpu()
            predicted = torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data)
            #print("{}:{}\n{}\n\n".format(verb[b_idx], tokenized_list[word_idx], predicted))
            print("\n", pycolor.PURPLE + str(verb[b_idx]) + pycolor.END)
            print("idx\tword\tcase\t[ga, o, ni, null]")
            for word_idx, word_score in enumerate(batch):
                # スコア出力
                np.set_printoptions(precision=2, floatmode='fixed')
                print("{}\t{}\t{}\t{}".format(word_idx, tokenized_list[word_idx],
                    color(kaku.get(int(torch.argmax(word_score).data))),
                    np.round(np.array(predicted[word_idx,:]), decimals=10)),
                    sep="\t")
                if torch.argmax(word_score) == 0:
                    pred_dic["ga"] = (word_idx, tokenized_list[word_idx])
                elif torch.argmax(word_score) == 1:
                    pred_dic["wo"] = (word_idx, tokenized_list[word_idx])
                elif torch.argmax(word_score) == 2:
                    pred_dic["ni"] = (word_idx, tokenized_list[word_idx])
            analyze_dic.setdefault(verb[b_idx], pred_dic)
        print(analyze_dic)

        if not temp == analyze_dic and i >= 1:
            print(src, file=open("zero_np_correct.txt", "a"))

def color(k):

    if k == "ga":
        return pycolor.RED + k + pycolor.END
    elif k == "o":
        return pycolor.BLUE + k + pycolor.END
    elif k == "ni":
        return pycolor.GREEN + k + pycolor.END
    else:
        return k

class args():
    data = "/home/miyawaki_shumpei/PAS/NTC_Matsu_converted"
    model_file = "result/edit/model-e2e-stack_ve256_vu256_depth6_adam_lr0.001_du0.1_dh0.0_True_size60_sub1127_th0.0_it1_rs2020_preFalse_loss-last.h5",
    tag = "dev"
    eval_ensemble = False
    thres = [0.5, 0.5, 0.5]
    thres_file = ""
    batch_size = 128
    gpu = "0"
    sub_model_number = -1
    model_name = "e2e-stack"
    depth_arg = 2
    depth_path = 3
    depth = 6
    vec_size_u = 256
    vec_size_e = 256
    optim = "adagrad"
    lr = 0.0002
    threshold = 0.8
    cache = False
    iter = 3
    drop_u = 0.1
    no_thres = 'store_true'
    out_dir = 'result'



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

def word2index(word):
    with open("/home/miyawaki_shumpei/PAS/NTC_Matsu_converted/wordIndex.txt") as f:
        for line in f:
            if line.split()[0] == word:
                return int(line.split()[1])
        return 0

def make_pred_tensor(pred_list):
    p_idx = [i for i, l in enumerate(pred_list) if l == 1]
    out = []
    for p in p_idx:
        temp = [[0.] for _ in range(len(pred_list))]
        temp[p][0] = 1.
        out.append(temp)
    return torch.tensor(out)



def run(src, PARSE, mecab_list, model_id, PRINT):

    gpu_id = 0

    torch.manual_seed(args.sub_model_number)
    tag = args.tag

    model: nn.Module = []
    #data = end2end_dataset(args.data_path + "/{}.json".format(args.tag), 100)

    #data = [torch.tensor([[ 2023, 5, 2370, 10, 24633, 33, 1486, 2402, 0, 0, 6670]]), torch.tensor([[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.]]])]
    #tokenized_base_list = ["私", "は", "学校", "で", "数学", "を", "勉強", "し", "まし", "た", "．"]

    if PARSE == "mecab":
        tokenized_base_list = [i[7] for i in mecab_list[:-1]]
        pred_list = [1 if "動詞" in i else 0 for i in mecab_list[:-1]]
    elif PARSE == "juman":
        tokenized_base_list = [m.genkei for m in mecab_list.mrph_list()]
        pred_list = [1 if "動詞" in i.hinsi else 0 for i in mecab_list.mrph_list()]

    data = [word2index(word) for word in tokenized_base_list]
    pred_tensor = make_pred_tensor(pred_list)
    sent_tensor = torch.tensor([data for _ in range(int(pred_tensor.size(0)))])
    xs_len = torch.tensor([len(tokenized_base_list) for _ in range(pred_tensor.size()[0])])
    data = [sent_tensor, pred_tensor, xs_len] ### words, is_target, xs_len = data
    if PRINT: print(data)

    params = from_model_id(model_id)

    if args.model_name == 'e2e-stack':
        word_embedding_matrix = pretrained_word_vecs("/home/miyawaki_shumpei/PAS/NTC_Matsu_converted", "/wordIndex.txt", args.vec_size_e)

        model = E2EStackedBiRNN(args.vec_size_u, params["depth"], 4, word_embedding_matrix, args.drop_u, True, params["it"], params["th"], params.get("null"))


    model_file = "result/edit/" + model_id + ".h5"
    print(pycolor.YELLOW + model_file + pycolor.END)
    model.load_state_dict(torch.load(model_file))
    #hoge = [i for i in model.parameters()]

    if torch.cuda.is_available():
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            test(args.out_dir, data, tag, model, model_id, args.thres, params["th"], params["it"], params.get("load"), tokenized_base_list, src, PRINT)
    else:
        test(args.out_dir, data, tag, model, model_id, args.thres, params["th"], params["it"], params.get("load"), tokenized_base_list, src, PRINT)


def predicate_argument_structure(src, PARSE, parse_list, model_id, PRINT):
    if PRINT: print(parse_list)
    run(src, PARSE, parse_list, model_id, PRINT)


def from_model_id(model_id):
    import re
    if "loss" in model_id:
        pattern = ".*?depth(?P<depth>.*?)_.*?lr(?P<lr>.*?)_.*?size(?P<size>.*?)_.*?sub(?P<sub>.*?)_.*?th(?P<th>.*?)_.*?it(?P<it>.*?)_.*?rs(?P<rs>.*?)_.*?pre(?P<pre>.*?)_.*?loss-(?P<loss>.*?)"#.*?null-(?P<null>.*?).*?"
    else:
        pattern = ".*?depth(?P<depth>.*?)_.*?lr(?P<lr>.*?)_.*?size(?P<size>.*?)_.*?sub(?P<sub>.*?)_.*?th(?P<th>.*?)_.*?it(?P<it>.*?)_.*?rs(?P<rs>.*?)_.*?pre(?P<pre>.*?)_"#.*?loss-(?P<loss>.*?)_.*?null-(?P<null>.*?).*?"
    pattern = re.compile(pattern)
    m = pattern.search(model_id)
    print(m.group("depth"))
    dic = { "depth" : int(m.group("depth")),
            "lr" : float(m.group("lr")),
            "size" : int(m.group("size")),
            "sub" : int(m.group("sub")),
            "th" : float(m.group("th")),
            "it" : int(m.group("it")),
            "rs" : int(m.group("rs")),
            "pre" : m.group("pre")}
    try:
        dic["loss"] = m.group("loss")
        dic["null"] = m.group("null")
    except:
        pass

    print(dic)
    return dic

if __name__ == '__main__':
    run()
