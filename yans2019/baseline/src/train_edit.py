# coding: utf-8

import argparse
import os
import sys
from os import path
import time
import json
import math

import torch.nn as nn
from torch import autograd

from arg_detector_io import *
from model_edit import E2EStackedBiRNN

random.seed(2016)

import numpy as np

np.random.seed(2016)

from evaluate_end_to_end import evaluate_multiclass_without_none

max_sentence_length = 90
prediction_model_id = 3


def train(out_dir, data_train, data_dev, model, model_id, epoch, lr_start, lr_min):
    
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
    threshold = 0.75

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
        
        
        count = -1

        ### 訓練モード ###
        model.train()
        for xss, yss in tqdm(data_train, total=len_train, mininterval=5):
            
            count += 1
            
            temp = torch.ones(yss.size()[0], yss.size()[1], 4) * 0
            # print('=== ', temp.size())
            if yss.size(1) > max_sentence_length:
                continue

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = autograd.Variable(yss).cuda()
                temp = temp.cuda()
            else:
                yss = autograd.Variable(yss)
            
            '''
            with open('./work/epoch.txt', 'a') as f:
                f.write('/* epoch毎一番初めの要素のxss, yss */')
                if count <= 0:
                    print(ep, xss, yss, sep='\n\n', file=f)
            '''
            
            ### 今度はscoreをjsonファイルに書き込んで ###
            
            if dic_score.get(count):
                #dic = dic_score[1]
                for batch, hoge in dic_score.items():
                    for word_idx, word_vec in hoge.items():
                        temp[int(batch)][int(word_idx),:] = torch.from_numpy(np.array(word_vec)).cuda()
            '''
            if os.path.exists('./score.json'):
                with open('./score.json') as json_file:
                    df = json.load(json_file)
            
                print(df)
                dic = df.get(count)
                if dic:
                    size = dic[0]
            
                    for batch, hoge in dic[1].items():
                        for word_idx, word_vec in hoge.items():
                            temp[int(batch)][int(word_idx),:] = torch.from_numpy(np.array(word_vec)).cuda()
            
            '''
            scores = model(xss, temp)
           
            


            '''
            with open('./work/out_score.txt', 'w') as f:
                f.write('/* スコア一覧 */')
                print(yss, scores, sep='\n\n', file=f)
            
            # print(scores)
            '''
            big, small = {}, {}
            for i, el in enumerate(scores):
                for j in range(el.size(0)):
                    if torch.max(el[j]) >= math.log(threshold) and torch.argmax(el[j]) <= 2:
                        #print(el[j])
                        small.setdefault(j, el[j].detach().cpu().numpy().tolist())
                big.setdefault(i, small)
                small = {}
            
            if big:
                dic_score.setdefault(count, big)#(str(len(scores))+' '+str(scores[0].size(0))+' '+str(scores[0].size(1)), big))
                  

            big = {}
            #with open('./score.json', 'a') as fil:
            #    json.dump(dic_score, fil, indent=4, sort_keys=True)
           
                

            
            
            
            ### ここで繰り返し入力する ###
            '''
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

            ### memo ###
            '''
            if count <= 0:
                f = open('./out_score.txt', 'w')
                print('xss', xss, 'yss', yss, sep='\n', end='\n\n', file=f)
                print('scores', scores, sep='\n', end='\n', file=f)
                count += 1
                f.close()
            '''

            loss = 0
            for i in range(yss.size()[0]):
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()
            print(total_loss, file=open('../loss.txt', 'w'), end='\n')

        with open('./work/score.json', 'w') as json_f:
            json.dump(dic_score, json_f, indent=4, sort_keys=True)
        #print(dic_score)

        print("loss:", total_loss[0], "lr:", lr, "time:", round(time.time()-start_time,2))
        
        losses.append(total_loss)
        print("", flush=True)
        print('Test...', flush=True)

        
        ### 評価モード ###
        model.eval()
        thres, obj_score, num_test_batch_instance = evaluate_multiclass_without_none(model, data_dev, len_dev, labels,thres_lists)
        # print('###', thres, obj_score, num_test_batch_instance, '###')
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


### loss_data_file を指定してグラフをプロット ###
def loss_plot(loss_file):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    train_iter = []
    loss_lis = []
    with open(file,'r') as f:
        for i,line in enumerate(f):
            train_iter.append(i)
            new_line = line.rstrip()
            loss = int(new_line)
            loss_lis.append(loss)
            
    plt.xlabel('train_iter')
    plt.ylabel('loss')
    plt.plot(train_iter, loss_lis)
    plt.savefig(args.out_dir + 'loss.png')


### tensor を受け取って指定した dim に対してグラフ化 ###



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

    return "{0}_{1}_{2}_{3}_lr{4}_du{5}_dh{6}_{7}_size{8}{9}".format(
        args.model_name,
        dim, depth,
        args.optim, args.lr,
        args.drop_u,
        args.drop_h,
        args.fixed_word_vec,
        args.data_size,
        sub_model_no
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
    print('MODEL_ID', model_id)
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
        print('BATCH_GENERATOR', e2e_single_seq_sentence_batch)
        #print(data_dev[0])
        word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt", args.vec_size_e)
        model = E2EStackedBiRNN(args.vec_size_u, args.depth, 4, word_embedding_matrix, args.drop_u, args.fixed_word_vec)
        print('EMB-MATRIX', word_embedding_matrix.size())
        # print(word_embedding_matrix)
    if torch.cuda.is_available():
        model = model.cuda()
        with torch.cuda.device(gpu_id):
            #print('input: ', args.out_dir, len(data_train), data_train[0], len(data_dev), model, model_id, args.max_epoch, args.lr, args.lr / 20, sep='\n\n', file = open('./input.txt', 'w'))
            ### input : overwrite label vector ###
            train(args.out_dir, data_train, data_dev, model, model_id, args.max_epoch, args.lr, args.lr / 20)
    else:
        train(args.out_dir, data_train, data_dev, model, model_id, args.max_epoch, args.lr, args.lr / 20)


if __name__ == '__main__':
    run()
