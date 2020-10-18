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
    return l.index(x) if x in l else default

def evaluate_multiclass_without_none(model, data_test, len_test,
                                     labels, thres_lists, model_id):
    num_test_instance = 0

    results = defaultdict(dict)
    best_result = init_result_info(results, thres_lists)
    
    dev_num_of_high = 0
    # dic_file = {}
    dic_loc = {}
    model.eval()
    for xss, yss, sent_id, file_name in tqdm(data_test, total=len_test, mininterval=5):
        
        temp = torch.zeros(yss.size()[0], yss.size()[1], 4)
        if torch.cuda.is_available():
            temp = temp.cuda()

        if yss.size(1) > max_sentence_length:
            continue
        num_test_instance += 1
        scores = model(xss, temp)
        
        for i in range(yss.size()[0]):
            ys = yss[i]
            predicted = torch.t(scores[i].cpu())
            # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

            for label in range(len(thres_lists)):  # case label index
                p_ys = predicted[label]
                add_results_foreach_thres_without_none(label, p_ys, results[label], thres_lists[label], ys)

    best_thres, f = calc_best_thres(best_result, results, thres_lists, labels, model_id)
    print('dev_num_of_high: ', dev_num_of_high)
    print('best_thres', best_thres)
    print('f', f)
    
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
    for thres in thres_list:
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
    csvf = open('./result/base/log/model-' + model_id + '.csv', 'a')
    writer = csv.writer(csvf, delimiter='\t')
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
        writer.writerow([labels[label], round(p*100,2), round(r*100,2), round(f*100,2), p_p, ppnp, pppn])
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
    print(labels[-1], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
          p_p, "\t", ppnp, "\t", pppn)
    writer.writerow([labels[-1], round(p * 100, 2), round(r * 100, 2), round(f * 100, 2),
          p_p, ppnp, pppn])
    writer.writerow([])
    csvf.close()

    return best_thres, f
