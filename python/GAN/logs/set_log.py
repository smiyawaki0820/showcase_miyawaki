import os, sys
import re
from logging import getLogger, StreamHandler, DEBUG, FileHandler, Formatter

def set_log(FILE):
    rltpath = os.path.dirname(__file__) # working_dir
    fname = os.path.abspath(rltpath + '/' + FILE)
    
    # ログの出力名を設定
    logger = getLogger(__name__)

    # コンソール出力するための設定
    handler = StreamHandler()
    logger.addHandler(handler)

    # ログのファイル出力先の設定
    ver = 0
    while (os.path.isfile(fname)):
        fname = re.sub(r'^(.*?)(\.v[0-9])?\.log$', r'\1.v{}.log'.format(ver), fname)
        ver += 1
    
    flog = FileHandler(fname, mode="a") # default mode = "a"
    logger.addHandler(flog)

    # level
    """
    NOTSET:     設定値などの記録
    DEBUG:      動作確認などのデバッグ記録
    INFO:       正常動作の記録
    WARNING:    ログの定義名
    ERROR:      エラーなどの重大な問題
    CRITICAL:   停止などの致命的な問題
    """
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)

    # ログの出力形式の設定
    formatter = Formatter('%(asctime)s : line-%(lineno)d : __%(levelname)s__ : %(message)s')
    flog.setFormatter(formatter)
    handler.setFormatter(formatter)

    logger.propagate = False


    logger.info('log file: {}'.format(fname))
    return logger


if __name__ == '__main__':
    logger = set_log('set_log.log')
    logger.info("Hello")
