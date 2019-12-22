#from ._pas import usr_input_test
import ipdb

def parameters():
    model_file="../model/pas/pas_LSOFT.h5"
    dic = {"model":model_file, "depth":6, "size":60, "ve":256, "vu":256, "op":"adam", "lr":0.0002, "du":0.1, "dh":0.0, "sub":1031, "th":10, "it":3, "rs":2016, "pre":False, "null":"inc"}
    print(dic)

def pas(mecab_list):
    """
    input:  parsing list
    output: pas dict
    """
    ipdb.set_trace()
    pas_dict = parameters()
    return pas_dict


