# -*- coding: utf-8 -*-
import MeCab
import re
from pyknp import Juman
PARSE = "juman"

def parsing(text) -> list:
    """
    input:  row text
    output: mecab parsing list
    """
    if "juman":
        jumanpp = Juman()
        result = jumanpp.analysis(text)
        return PARSE, result
    elif PARSE == "mecab":
        tagger = MeCab.Tagger()
        parse = tagger.parse(text)
        return PARSE, [re.split(r"\s|,", word) for word in parse.splitlines()]

def run():
    # text = str(input())
    text = "すもももももももものうち"
    print(" ".join([i[0] for i in parsing(text)]))

if __name__ == "__main__":
    from pyknp import Jumanpp
    jumanpp = Jumanpp()
    result = jumanpp.analysis(u"ケーキを食べる")
    for mrph in result.mrph_list():
        print(u"見出し:{0}".format(mrph.midasi))

