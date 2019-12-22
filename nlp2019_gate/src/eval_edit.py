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
import ipdb

ANALYZE = False
max_sentence_length = 80

def my_index(l, x, default=False):
    return [i for i,e in enumerate(l) if x == e]

def evaluate_multiclass_without_none(model, data_test, len_test,
                                     labels, thres_lists, model_id, threshold, iterate_num, ep):
    num_test_instance = 0

    d = defaultdict(dict)
    best_results = init_result_info(defaultdict(dict), thres_lists)
    list_result = [d for _ in range(iterate_num)]
    iter_thres_lists = [thres_lists for _ in range(iterate_num)]
    iter_best_result = [best_results for _ in range(iterate_num)]
    model.eval()

    for xss, yss, sent_id, file_name in tqdm(data_test, total=len_test, mininterval=5):

        temp = torch.zeros(yss.size()[0], yss.size()[1], 4)
        if torch.cuda.is_available():
            temp = temp.cuda()
            #temp_pred = temp_pred.cuda()

        if yss.size(1) > max_sentence_length:
            continue
        num_test_instance += 1

        ### iterate in epoch ###

        scores = model(xss, yss, temp, iterate_num, threshold)

        if ANALYZE:
            pass
            #analyze(xss, yss, sent_id, file_name, scores)
            #_pred = []
            #for score in scores.values():
            #    _word = [int(torch.argmax(word)) for batch in score for word in batch]
            #    _pred.append(_word)


        iter_result = []
        for it, score in enumerate(scores): # iteration 毎の解析
            results = defaultdict(dict)
            hoge = init_result_info(results, thres_lists)
            for i in range(yss.size()[0]):
                ys = yss[i]
                predicted = torch.t(score[i].cpu())
                # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

                for label in range(len(thres_lists)):  # case label index
                    p_ys = predicted[label]
                    add_results_foreach_thres_without_none(label, p_ys, results, iter_thres_lists[it][label], ys, it)
            iter_result.append(results)

        # {0: {0.1: {'pp': 1, 'np': 3, 'pn': 3, 'nn': 0, 'prf': {'pp': 0, 'ppnp': 0, 'pppn': 0}}, 0.1:{}}, 1:{}}
        if len(list_result[0]) == 0:
            list_result = iter_result
        else:
            list_result = [merge_dict(dict(lres), dict(ires), lambda x, y: x + y) for lres, ires in zip(list_result, iter_result)]


    iter_best_thres, iter_f = [], []

    for it in range(iterate_num):
        best_thres, f = calc_best_thres(iter_best_result[it], list_result, iter_thres_lists[it], labels, model_id, it)
        iter_best_thres.append(best_thres)
        iter_f.append(round(f, 4))

    print('best_thres', iter_best_thres)
    print('f', iter_f)

    return best_thres, f, num_test_instance


#def file_output_iter_analyze()


def merge_dict(d1, d2, func=lambda x, y: y):

    d1 = d1.copy()
    d2 = d2.copy()
    for k, v in d2.items():
        if type(v) == dict:
            d1[k] = merge_dict(d1.get(k), d2.get(k), lambda x, y: x + y)
        else:
            d1[k] = func(d1[k], v) if k in d1 else v
    d2.update(d1)
    return d2

def init_result_info(results, thres_sets):
    for label in range(len(thres_sets)):
        for thres in thres_sets[label]:
            results[label][thres] = result_info()
    best_result = result_info()
    return best_result


def result_info():
    return {"pp": 0, "np": 0, "pn": 0, "nn": 0,
            "prf": {"pp": 0, "ppnp": 0, "pppn": 0}}


def add_results_foreach_thres_without_none(label, p_ys, iter_result, thres_list, ys, it):
    values, assignments = p_ys.max(0) # values=torch.max, assignments=torch.argmax
    assignment = assignments.item()   # assignment = int(assignments)
    for thres in thres_list:
        # prob = math.pow(math.e, values.data[0]) - thres
        prob = math.pow(math.e, values.data.item()) - thres  # prob >= 0: 予測

        if not any(y == label for y in ys): # 予測 role に対応する gold が存在しない # target=null
            if prob < 0:
                iter_result[label][thres]["nn"] += 1  # 予測しない
            else:
                iter_result[label][thres]["np"] += 1  # null に対して role を付与
        elif prob >= 0:                     # target = kaku かつ 予測あり
            if ys[assignment] == label:
                iter_result[label][thres]["pp"] += 1  # 正解
            else:                           # 予測はしたけど other kaku
                iter_result[label][thres]["np"] += 1  # gold に対して 誤り role を付与
                iter_result[label][thres]["pn"] += 1  # 同時に pn がインクリメント
        else:                               # target = kaku かつ 予測なし
            iter_result[label][thres]["pn"] += 1      # 通常の pn
    return iter_result


def calc_best_thres(best_result, results, thres_lists, labels, model_id, it):
    best_thres = [0.0, 0.0, 0.0]

    print("", flush=True)
    header = ['','p', 'r', 'f1', 'p_p', 'ppnp', 'pppn']
    index = ['ga', 'wo', 'ni']
    writer = csv.writer(open('./result/edit/log/model-'+model_id+'.csv', 'a'), delimiter='\t')
    print("\033[32m iter_" + str(it) + "\033[0m")
    writer.writerow([it])
    for label in range(len(thres_lists)):
        def f_of_label(thres):
            return results[it][label][thres]["prf"]["f"]

        for thres in thres_lists[label]:
            p_p = results[it][label][thres]["pp"]
            n_p = results[it][label][thres]["np"]
            p_n = results[it][label][thres]["pn"]
            n_n = results[it][label][thres]["nn"]
            prf = results[it][label][thres]["prf"]

            prf["pp"] = p_p
            ppnp = prf["ppnp"] = p_p + n_p
            pppn = prf["pppn"] = p_p + p_n
            p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
            r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
            f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0

        best_thres[label] = best_t = max(thres_lists[label], key=f_of_label)
        p = results[it][label][best_t]["prf"]["p"]
        r = results[it][label][best_t]["prf"]["r"]
        f = results[it][label][best_t]["prf"]["f"]
        p_p = results[it][label][best_t]["prf"]["pp"]
        ppnp = results[it][label][best_t]["prf"]["ppnp"]
        pppn = results[it][label][best_t]["prf"]["pppn"]
        print(labels[label], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t", p_p, "\t", ppnp, "\t", pppn)
        writer.writerow([labels[label], round(p*100,2), round(r*100,2), round(f*100,2), p_p, ppnp, pppn])
        best_result["pp"] += results[it][label][best_t]["pp"]
        best_result["np"] += results[it][label][best_t]["np"]
        best_result["pn"] += results[it][label][best_t]["pn"]
        best_result["nn"] += results[it][label][best_t]["nn"]
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


