# coding: utf-8

import argparse
import os
import sys
from os import path
import time
import json,ijson
import math

import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from arg_edit import *
from model_edit import E2EStackedBiRNN

random.seed(2016)

import numpy as np

np.random.seed(2016)

from eval_edit import evaluate_multiclass_without_none

max_sentence_length = 90
prediction_model_id = 3


def train(out_dir, data_train, data_dev, model, model_id, epoch, lr_start, lr_min, threshold):
    
   
   
    len_train = len(data_train)
    len_dev = len(data_dev)

    early_stopping_thres = 4
    early_stopping_count = 0
    best_performance = -1.0
    best_epoch = 0
    best_thres = [0.0, 0.0]
    best_lr = lr_start
    lr = lr_start
    lr_reduce_factor = 0.5
    lr_epsilon = lr_min * 1e-4
    
    iterate_num = 1 # T とりあえずn回モデルに入力する
    dic_score = {}
   
    AR = True

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)
    losses = []

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    
    for ep in range(epoch):
        start_time = time.time() 
        total_loss = torch.Tensor([0])
        early_stopping_count += 1

        print(model_id, 'epoch {0}'.format(ep + 1), flush=True)

        print('Train...', flush=True)
        random.shuffle(data_train)
        
        num_of_high = 0
        count = 1
        
        ### 訓練モード ###
        model.train()
        for xss, yss, sent_id, file_name in tqdm(data_train, total=len_train, mininterval=5):
           
            #print(file_name, sent_id)           
            temp = F.log_softmax(torch.ones(yss.size()[0], yss.size()[1], 4) * 0.25)
            #if ep >= 1:
            #    print('=== ', temp)
            if yss.size(1) > max_sentence_length:
                continue

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = autograd.Variable(yss).cuda()
                temp = temp.cuda()
            else:
                yss = autograd.Variable(yss)
            
            
            with open('./work/epoch.txt', 'a') as f:
                f.write('/* epoch毎一番初めの要素のxss, yss */')
                if count <= 3:
                    print(ep, xss, yss, sep='\n\n', file=f)
            
            
            
            ### get score from json file ###
            if AR:
                if os.path.exists('./work/score.json'):
               
               
                    with open('./work/score.json', 'rb') as jsf:
                        objects = ijson.items(jsf, str(file_name[0])+'.item')
                        ob = (o[str(sent_id[0])] for o in objects if o.get(str(sent_id[0])))
                        for lis in ob:
                            for dic in lis:
                                temp_label = [float(i) for i in dic["label"]]
                                temp[int(dic["batch_id"]),int(dic["word_idx"]),:] = torch.from_numpy(np.array(temp_label))
                        
                    """
                    ### get score from var-dic ###
                    if dic_score.get(file_name[0]):
                        if dic_score[file_name[0]].get(sent_id[0]):
                
                            dic = dic_score[file_name[0]][sent_id[0]]
                            for batch, fuga in dic.items():
                                #print('#dic#', batch, fuga)
                                for word_idx, label_vec in fuga.items():
                                    #print('#dicの値#',word_idx, label_vec)
                                    #print(batch, word_idx, label_vec)
                                    temp[int(batch),int(word_idx),:] = torch.from_numpy(np.array(label_vec))
                            del dic_score[file_name[0]][sent_id[0]]
                                # print(temp)
                    """

                # IN:{word, pred, score(t-1)}, OUT:{score(t)} 
                scores = model(xss, temp)
           
                sentence_, batch_, word_  = {}, {}, {}
                lis_sent, lis_word = [], []
                for i, batch in enumerate(scores):
                    for word_idx in range(batch.size(0)):
                        if torch.max(batch[word_idx]) >= math.log(threshold) and torch.argmax(batch[word_idx]) <= 2:
                            num_of_high += 1
                            word_["batch_id"] = i
                            word_["word_idx"] = word_idx
                            word_["label"] = batch[word_idx].detach().cpu().numpy().tolist()
                            # word_.setdefault(word_idx, batch[word_idx].detach().cpu().numpy().tolist())
                    if not word_ == {}:
                        lis_word.append(word_)
                        # batch_.setdefault(i, word_)
                    word_ = {}
                    
                if not lis_word == []: 
                    sentence_.setdefault(sent_id[0], lis_word)
                    if dic_score.get(file_name[0]):
                        dic_score[file_name[0]].append(sentence_)
                    else:
                        lis_sent.append(sentence_)
                        dic_score.setdefault(file_name[0], lis_sent)
              

            else:  # AR == False
                scores = model(xss, temp)


            '''    
            ### iterate in epoch ###
            if ep >= 2:
                for t in range(iterate_num):
                    scores = model(xss, temp)
                    # lis = []
                    # for s in scores:
                        # if max(s) >= threshold:
                            # lis.append(s)
                        # else:
                            # lils.append(0)
                    # temp = torch.stack(lis. dim=0)
                    temp = torch.stack([s for s in scores], dim=0)
                    
                    # temp, yssの確認
                    if count <= 5:
                        count += 1
                        print('LABEL-EMB', temp) # ラベルベクトルの中身をチェック
                        print('yss', yss)        # yss とのロスを計算している
                    
            else:
                scores = model(xss, temp)
            '''


            loss = 0
            for i in range(yss.size()[0]):
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()
            print(np.array(total_loss).tolist()[0], file=open('./result/log/model-'+model_id+'_loss.txt', 'a'), end='\n')

            count += 1

        with open('./work/score.json', 'w') as json_f:
            json.dump(dic_score, json_f, indent=4, sort_keys=True)

        print("loss:", total_loss[0], "lr:", lr, "time:", round(time.time()-start_time,2))
        losses.append(total_loss)
        print("", flush=True)
        print('Test...', flush=True)

        
        ### 評価モード ###
        model.eval()
        print('num_of_high: ', num_of_high, end='\n')
        thres, obj_score, num_test_batch_instance = evaluate_multiclass_without_none(model, data_dev, len_dev, labels,thres_lists, model_id)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            print("save model", flush=True)
            torch.save(model.state_dict(), out_dir + "/model-" + model_id + ".h5")
        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                print("load model: epoch{0}".format(best_epoch), flush=True)
                model.load_state_dict(torch.load(out_dir + "/model-" + model_id + ".h5"))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
              "f:", best_performance)

    print(model_id, "\tbest in epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
          "f:", best_performance)


def set_log_file(args, tag, model_id):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    fd = os.open(args.out_dir + '/log/log-' + tag + '-' + model_id + ".txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def create_pretrained_model_id(args, model_name):
    sub_model_no = "" if args.sub_model_number == -1 \
        else "_sub{0}".format(args.sub_model_number)

    depth = "{0}-{1}-{2}".format(args.depth, args.depth_path, args.depth_arg)

    return "{0}_vu{1}_{2}_{3}_lr{4}_du{5}_ve{6}_b{7}_size{8}{9}".format(
        model_name,
        args.vec_size_u, depth,
        args.optim, 0.0002,
        args.drop_u,
        args.vec_size_e,
        args.batch_size,
        100,
        sub_model_no
    )


def create_model_id(args):
    sub_model_no = "" if args.sub_model_number == -1 \
        else "_sub{0}".format(args.sub_model_number)

    depth = args.depth
    dim = 've{0}_vu{0}'.format(args.vec_size_u)

    return "{0}_{1}_{2}_{3}_lr{4}_du{5}_dh{6}_{7}_size{8}{9}_th{10}".format(
        args.model_name,
        dim, depth,
        args.optim, args.lr,
        args.drop_u,
        args.drop_h,
        args.fixed_word_vec,
        args.data_size,
        sub_model_no,
        args.threshold
    )


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath,
                        help='data path')
    parser.add_argument('--model_no', "-mn", dest='sub_model_number', type=int,
                        help='sub-model number for ensembling', default=-1)
    parser.add_argument('--size', '-s', dest='data_size', type=int,
                        default=100,
                        help='data size (%)')
    parser.add_argument('--model', '-m', dest='model_name', type=str,
                        default="path-pair-bin",
                        help='model name')
    parser.add_argument('--epoch', '-e', dest='max_epoch', type=int,
                        default=150,
                        help='max epoch')
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
                        default=2,
                        help='the number of hidden layer')
    parser.add_argument('--optimizer', '-o', dest='optim', type=str,
                        default="adagrad",
                        help='optimizer')
    parser.add_argument('--lr', '-l', dest='lr', type=float,
                        default=0.005,
                        help='learning rate')
    parser.add_argument('--dropout-u', '-du', dest='drop_u', type=float,
                        default=0.2,
                        help='dropout rate of LSTM unit')
    parser.add_argument('--dropout-h', '-dh', dest='drop_h', type=float,
                        default=0.0,
                        help='dropout rate of hidden layers')
    parser.add_argument('--dropout-attention', '-da', dest='drop_a', type=float,
                        default=0.2,
                        help='dropout rate of attention softmax layer')
    parser.add_argument('--threshold', '-th', dest='threshold', type=float,
                        default=0.6,
                        help='threshold of score')
    parser.add_argument('--batch', '-b', dest='batch_size', type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int,
                        default='-1',
                        help='GPU ID for execution')
    parser.add_argument('--tune-word-vec', '-twv', dest='fixed_word_vec', action='store_false',
                        help='do not re-train word vec')

    parser.add_argument('--out_dir', type=str, default='result')

    parser.set_defaults(fixed_word_vec=True)

    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    model_id = create_model_id(args)
    #print('MODEL_ID', model_id)
    gpu_id = 0

    if not path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print("# Make directory: {}".format(args.out_dir))
    if not path.exists(args.out_dir + "/log"):
        os.mkdir(args.out_dir + "/log")
        print("# Make directory: {}".format(args.out_dir + "/log"))

    torch.manual_seed(args.sub_model_number)
    # set_log_file(args, "train", model_id)

    data_train = end2end_dataset(args.data_path + "/train.json", args.data_size)
    data_dev = end2end_dataset(args.data_path + "/dev.json", args.data_size)

    model: nn.Module = None

    if args.model_name == 'e2e-stack':
        data_train = list(end2end_single_seq_instance(data_train, e2e_single_seq_sentence_batch))
        data_dev = list(end2end_single_seq_instance(data_dev, e2e_single_seq_sentence_batch))
        #print('BATCH_GENERATOR', e2e_single_seq_sentence_batch)
        word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt", args.vec_size_e)
        model = E2EStackedBiRNN(args.vec_size_u, args.depth, 4, word_embedding_matrix, args.drop_u, args.fixed_word_vec)
        #print('EMB-MATRIX', word_embedding_matrix.size())
        # print(word_embedding_matrix)
    if torch.cuda.is_available():
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            #print('input: ', args.out_dir, len(data_train), data_train[0], len(data_dev), model, model_id, args.max_epoch, args.lr, args.lr / 20, sep='\n\n', file = open('./input.txt', 'w'))
            train(args.out_dir, data_train, data_dev, model, model_id, args.max_epoch, args.lr, args.lr / 20, args.threshold)
    else:
        train(args.out_dir, data_train, data_dev, model, model_id, args.max_epoch, args.lr, args.lr / 20, args.threshold)


if __name__ == '__main__':
    run()