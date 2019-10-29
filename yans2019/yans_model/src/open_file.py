import os
import torch
import numpy as np
import json

def open_file(score_path, ep):
    
    if os.path.exists(score_path):
        with open(score_path) as json_file, open('./work/dev_gold.json') as gold:
            df = json.load(json_file)
            df_gold = json.load(gold)
        
        dic = {0:'ga', 1:'wo', 2:'ni'}
        with open('./work/high_dependency.txt', 'a') as write_file:
            

            for file_name, lis_sent in df.items():
                for dic_sent in lis_sent:
                    for sent_id, lis_word in dic_sent.items():
                        print(df_gold[file_name][sent_id], file=write_file, end='\n')
                        for dic_word in lis_word:
                            #print("## dic_word: ", dic_word, file=write_file)
                            for df_batch, batch in zip(df_gold[file_name][sent_id], dic_word.values()): # for i in range(batch)
                                #print('## df_batch, batch: ', df_batch, batch, file=write_file)
                                lis = open_path(file_name, sent_id)
                                if not lis[int(df_batch["pred"])].startswith('#'):
                                    lis[int(df_batch["pred"])] = '#{}#'.format(lis[int(df_batch["pred"])])
                                '''
                                if not lis[int(batch["word_idx"])].startswith('__'):
                                    lis[int(batch["word_idx"])] = '__{0}({1}_{2})__'.format(lis[batch["word_idx"]], dic[np.argmax(np.array(batch["label"]))], str(ep))
                                '''
                                print(' '.join([i for i in lis]), file=write_file)


def open_path(path, sent_id):

    PATH = '../../PAS/train-with-juman/' + path
    #print(file_name, sent_id, batch, word_idx, kaku, yss_i, file=write_file)
    #kaku = kaku.numpy().tolist()[0]
    sent_id = int(sent_id)
    lis = []

    ### 文lis 作成 ###       
    if os.path.isfile(PATH):
        # print('PATH: ', PATH, file=write_file)
        count = 0
        with open(PATH) as f:
            for line in f:
                if line.startswith('*'):
                    continue
                if sent_id == 0:
                    if (line.startswith('EOS')):
                        break
                    else:
                        lis.append(line.split()[0])
                else:
                    if line.startswith('EOS'):
                        count += 1
                    if count == sent_id:
                        if line.startswith('EOS'):
                            continue
                        lis.append(line.split()[0])
                    elif count > sent_id:
                        break
      
    else:
        print('404 file not exist')

    return lis




if __name__ == '__main__':
    file_name = 'dev/950112-0000-950112002.ntc'
    sent_id = 1

    #open_file(file_name, sent_id, 0, 3)
    
