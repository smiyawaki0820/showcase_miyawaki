import math
from collections import defaultdict
import csv
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import ijson, json
import os
import numpy as np

max_sentence_length = 80

def my_index(l, x, default=False):
    return [i for i,e in enumerate(l) if x == e]

def evaluate_multiclass_without_none(model, data_test, len_test,
                                     labels, thres_lists, model_id, threshold, iterate_num, ep, base=None):
    num_test_instance = 0

    results = defaultdict(dict)
    best_result = init_result_info(results, thres_lists)
    dev_num_of_high = 0
    dic_file = {}
    dic_loc = {}
    model.eval()

    pred_count_dev = 0
    arg_count_dev = 0

    for xss, yss in tqdm(data_test, total=len_test, mininterval=5):

        #temp_pred = torch.zeros(yss.size()[0] ,yss.size()[1], 1)
        #for i in range(xss[1].size(0)):
        #   for j in range(xss[1].size(1)):
        #      if xss[1][i,j,:] == 1:
        #         temp_pred[:,j,:] = torch.ones(xss[1].size(0),1)

        if xss[0].size(1) > max_sentence_length:
            continue
        num_test_instance += 1

        pred_count_dev += xss[0].size(0)
        import ipdb; ipdb.set_trace()
        for t in yss:
            arg_count_dev += int(torch.sum(t > 0))

        ### iterate in epoch ###
        if base:
            init, _ = base(xss, torch.tensor([]), init=True)
            temp = torch.stack([i for i in init])
        else:
            temp = torch.tensor([])

        scores, _ = model(xss, temp)
        scores = scores[-1]
        #print(file_name[0], sent_id[0], sep=' ', end='\n', file=open('./result/edit/hoge/'+model_id+'_WRjudge.txt', 'a'))
        high_score = {}

        for i in range(len(yss)):
            ys = yss[i]
            predicted = torch.t(scores[i].cpu())
            # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

            for label in range(len(thres_lists)):  # case label index
                p_ys = predicted[label]
                add_results_foreach_thres_without_none(label, p_ys, results[label], thres_lists[label], ys)

    ### json dump ###
    with open('./work/dev_score.json', 'w') as json_f:
        json.dump(dic_file, json_f, indent=4) #, sort_keys=True)


    best_thres, f = calc_best_thres(best_result, results, thres_lists, labels, model_id)
    print("pred_count_dev:{}\targ_count_dev:{}".format(pred_count_dev, arg_count_dev), file=open("experiment_settings.txt","a"), end="\n")
    print('best_thres', best_thres)
    print('f', f)

    return best_thres, f, num_test_instance


def open_f(lis, batch_idx, word_idx, kaku, t, gold, cpo, cpx):
    dic = {0:'ga', 1:'wo', 2:'ni'}
    if not lis[int(word_idx)].startswith('_'):
        if int(kaku) == int(gold):
            lis[int(word_idx)] = '_{}({}:t{}b{}.O)_'.format(lis[int(word_idx)], dic[int(kaku)], t, batch_idx)
            cpo += 1
        else:
            lis[int(word_idx)] = '_{}({}:t{}b{}.X)_'.format(lis[int(word_idx)], dic[int(kaku)], t, batch_idx)
            cpx += 1
    else:
        pre = lis[int(word_idx)]
        lis_b = my_index(list(pre), 'b')
        lis_pre = [int(pre[b+1]) for b in lis_b]
        if int(kaku) == int(gold):
            lis[int(word_idx)] = '{}(t{}b{}.O)_'.format(pre[:len(lis[int(word_idx)])-1], t, batch_idx)
            if not batch_idx in lis_pre:
                cpo += 1

        else:
            lis[int(word_idx)] = '{}(t{}b{}.X)_'.format(pre[:len(lis[int(word_idx)])-1], t, batch_idx)
            if not batch_idx in lis_pre:
                cpx += 1

    return lis



def create_sent_lis(PATH, sent_id):
    lis, dep_lis = [], []
    if os.path.isfile(PATH):
        count, word_count = 0, 0
        with open(PATH) as f:
            for line in f:
                temp = line.split()
                if line.startswith('*'):
                    continue
                if sent_id == 0:
                    if (line.startswith('EOS')):
                        break
                    else:
                        word_count += 1
                        lis.append(temp[0])
                        if len(temp) > 3:
                            dep_lis.append((word_count, temp[3:]))
                else:
                    if line.startswith('EOS'):
                        count += 1
                    if count == sent_id:
                        if line.startswith('EOS'):
                            continue
                        word_count += 1
                        lis.append(temp[0])
                        if len(temp) > 3:
                            dep_lis.append((word_count, temp[3:]))
                    elif count > sent_id:
                        break
    else:
      print('404 file not')
    return lis, dep_lis

   


def dic_for_json(scores, dic_file, threshold):
    lis_sent, lis_batch = [], []
    dic_sent = {}
    for i, batch in enumerate(scores):
        dic_batch = {}
        lis_word = []
        for word_idx in range(batch.size(0)):
            dic_word = {}
            if torch.max(batch[word_idx]) >= math.log(threshold) and torch.argmax(batch[word_idx]) <= 2:
                kaku = (batch[word_idx].cpu().detach()).numpy().tolist()
                dev_num_of_high += 1
                dic_word["word_idx"] = word_idx
                dic_word["word_label"] = kaku
                lis_word.append(dic_word)
        if not lis_word == []:
            dic_batch["term"] = lis_word
            dic_batch["batch_id"] = i
            lis_batch.append(dic_batch)

    if not lis_batch == []:
        dic_sent.setdefault(sent_id[0], lis_batch)
    if file_name[0] in dic_file:

        dic_file[file_name[0]].update(dic_sent)
    else:
        dic_file.setdefault(file_name[0], dic_sent)

    return dic_file


def init_result_info(results, thres_sets):
    for label in range(len(thres_sets)):
        for thres in thres_sets[label]:
            results[label][thres] = result_info()
    best_result = result_info()
    return best_result


def result_info():
    return {"pp": 0, "np": 0, "pn": 0, "nn": 0,
            "prf": {"pp": 0, "ppnp": 0, "pppn": 0}}


def add_results_foreach_thres_without_none(label, p_ys, result, thres_list, ys):
    values, assignments = p_ys.max(0)
    assignment = assignments.item()
    # print("label:", label, "assignment:", assignment, "gold label:", ys[assignment])
    for thres in thres_list:
        # prob = math.pow(math.e, values.data[0]) - thres
        prob = math.pow(math.e, values.data.item()) - thres  # fix

        if not any(y == label for y in ys):
            if prob < 0:
                result[thres]["nn"] += 1
            else:
                result[thres]["np"] += 1
        elif prob >= 0:
            if ys[assignment] == label:
                result[thres]["pp"] += 1
            else:
                result[thres]["np"] += 1
                result[thres]["pn"] += 1
        else:
            result[thres]["pn"] += 1


def calc_best_thres(best_result, results, thres_lists, labels, model_id):
    best_thres = [0.0, 0.0, 0.0]

    print("", flush=True)
    header = ['','p', 'r', 'f1', 'p_p', 'ppnp', 'pppn']
    index = ['ga', 'wo', 'ni']
    writer = csv.writer(open('./result/edit/log/model-'+model_id+'.csv', 'a'), delimiter='\t')
    for label in range(len(thres_lists)):
        def f_of_label(thres):
            return results[label][thres]["prf"]["f"]

        for thres in thres_lists[label]:
            p_p = results[label][thres]["pp"]
            n_p = results[label][thres]["np"]
            p_n = results[label][thres]["pn"]
            n_n = results[label][thres]["nn"]
            prf = results[label][thres]["prf"]

            prf["pp"] = p_p
            ppnp = prf["ppnp"] = p_p + n_p
            pppn = prf["pppn"] = p_p + p_n
            p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
            r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
            f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0

        best_thres[label] = best_t = max(thres_lists[label], key=f_of_label)
        p = results[label][best_t]["prf"]["p"]
        r = results[label][best_t]["prf"]["r"]
        f = results[label][best_t]["prf"]["f"]
        p_p = results[label][best_t]["prf"]["pp"]
        ppnp = results[label][best_t]["prf"]["ppnp"]
        pppn = results[label][best_t]["prf"]["pppn"]
        print(labels[label], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t", p_p, "\t", ppnp, "\t", pppn)
        writer.writerow([labels[label], round(p*100,2), round(r*100,2), round(f*100,2), p_p, ppnp    , pppn])

        best_result["pp"] += results[label][best_t]["pp"]
        best_result["np"] += results[label][best_t]["np"]
        best_result["pn"] += results[label][best_t]["pn"]
        best_result["nn"] += results[label][best_t]["nn"]
    p_p = best_result["pp"]
    n_p = best_result["np"]
    p_n = best_result["pn"]
    n_n = best_result["nn"]
    prf = best_result["prf"]
    prf["pp"] = p_p
    ppnp = prf["ppnp"] = p_p + n_p
    pppn = prf["pppn"] = p_p + p_n
    p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
    r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
    f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0
    writer.writerow([labels[-1], round(p*100,2), round(r*100,2), round(f*100,2), p_p, ppnp, pppn])
    writer.writerow([])
    
    return best_thres, f

