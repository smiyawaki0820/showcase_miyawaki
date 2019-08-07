import math
from collections import defaultdict
import csv
import torch
from tqdm import tqdm
import torch.nn.functional as F
from open_file import *
import torch.nn as nn
import ijson, json
import os
import numpy as np

max_sentence_length = 80

def my_index(l, x, default=False):
    return l.index(x) if x in l else default

def evaluate_multiclass_without_none(model, data_test, len_test,
                                     labels, thres_lists, model_id, threshold, iterate_num):
    num_test_instance = 0

    results = defaultdict(dict)
    best_result = init_result_info(results, thres_lists)
    
    dev_num_of_high = 0
    # dic_file = {}
    dic_loc = {}
    model.eval()
    for xss, yss, sent_id, file_name in tqdm(data_test, total=len_test, mininterval=5):
        
        ### create gold file ###
        '''
        if ep <= 0 :
            np_xss = xss[1].numpy()
            lis_batch, lis_pred = [], []
            dic_gold = {}
            for i in range(xss[1].size(0)):
                for pred_i, x in enumerate(xss[1][i,:,0]):
                    if x == 1:
                        lis_pred.append(pred_i)
            lis_yss = yss.numpy().tolist()
            for pred, term in zip(lis_pred, lis_yss):
                dic_batch = {}
                label = {"ga":my_index(term, 0), "wo":my_index(term,1), "ni":my_index(term,2)}
                dic_batch["label"] = label
                dic_batch["pred"] = pred
                lis_batch.append(dic_batch)
            
            dic_gold[sent_id[0]] = lis_batch
            if file_name[0] in dic_loc:
                dic_loc[file_name[0]].update(dic_gold)
            else:
                dic_loc[file_name[0]] = dic_gold
        '''    

        temp = torch.zeros(yss.size()[0], yss.size()[1], 4)
        if torch.cuda.is_available():
            temp = temp.cuda()

        if yss.size(1) > max_sentence_length:
            continue
        num_test_instance += 1
        scores = model(xss, temp)
        ### iterate in epoch ###
        '''
        for t in range(iterate_num):
            scores = model(xss, temp)
            for batch_idx, batch in enumerate(scores):
                for word_idx in range(batch.size(0)):
                    if torch.max(batch[word_idx]) >= math.log(threshold) and torch.argmax(batch[word_idx]) <= 2:
                        dev_num_of_high += 1
                        temp[batch_idx, word_idx, :] = batch[word_idx]  
        '''
        
        ### create dic dumped to dev_score.json ###
        '''
        lis_sent = []

        dic_sent = {}
        lis_batch = []
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
        '''
       
       
        for i in range(yss.size()[0]):
            ys = yss[i]
            predicted = torch.t(scores[i].cpu())
            # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

            for label in range(len(thres_lists)):  # case label index
                p_ys = predicted[label]
                add_results_foreach_thres_without_none(label, p_ys, results[label], thres_lists[label], ys)

    ### json dump ###
    '''
    with open('./work/dev_score.json', 'w') as json_f:
        json.dump(dic_file, json_f, indent=4) #, sort_keys=True)
    if ep <= 0:
        with open('./work/dev_gold.json', 'w') as gold:
            json.dump(dic_loc, gold, indent=4, sort_keys=True)  
    '''


    best_thres, f = calc_best_thres(best_result, results, thres_lists, labels, model_id)
    print('dev_num_of_high: ', dev_num_of_high)
    print('best_thres', best_thres)
    print('f', f)
   
    writer = csv.writer(open('./result/log/model-'+model_id+'_result.csv', 'a'))
    writer.writerow([f])
    
    return best_thres, f, num_test_instance


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
        print(labels[label], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
              p_p, "\t", ppnp, "\t", pppn)
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
    header = ['','p', 'r', 'f1', 'p_p', 'ppnp', 'pppn']
    index = ['ga', 'wo', 'ni', 'all']
    with open('./result/log/model-'+model_id+'_result.csv', 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow(header)
        for kaku in index:
            writer.writerow([kaku]+[round(p*100,2), round(r*100,2), round(f*100,2), p_p, ppnp, pppn])
    print(labels[-1], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
          p_p, "\t", ppnp, "\t", pppn)

    return best_thres, f
