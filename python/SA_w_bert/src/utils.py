import os
import sys
import logging
import subprocess as subp
from typing import Generator, List, Tuple

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=int(os.environ['LOG_LEVEL']),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
percent = int


def unix(command:str):
    def check_subprocess(proc):
        assert proc.returncode == 0
        return proc
    
    if '|' not in command:
        proc =  subp.run(command.split(),
                    stdout=subp.PIPE,
                    stderr=subp.PIPE
                    )
        return check_subprocess(proc)

    else:
        proc_stdout = None
        for i, com in enumerate(command.split('|')):
            proc = subp.Popen(com.split(), 
                        stdin=proc_stdout,
                        stdout=subp.PIPE,
                        stderr=subp.PIPE
                        )
            proc_stdout = proc.stdout
        return ckeck_subprocess(proc)


class FileHandler(object):
        
    def len(f) -> int:
        wc = unix(f'wc -l {f}')
        return int(wc.stdout.decode('utf8').split()[0])

    def shuffle(f, size:percent, dest, fo) -> Tuple[os.path.abspath, int]:
        n_lines = int(FileHandler.len(f) * size / 100)
        fo = os.path.join(os.path.join(dest, fo))
        process = unix(f'shuf {fi} -o {fo} -n {n_lines}')
        assert os.path.isfile(fo)
        logger.debug(__name__)
        logger.debug(f'wc -l f_shuf ... {n_lines} {fo}')
        return fo, size

    def reader(fi) -> Generator[str, None, None]:
        for line in open(fi, 'r'):
            if line.rstrip():
                yield line.rstrip()

    def split(fi, fs_out:list, ratio:list):
        assert len(fs_out) == len(ratio)
        assert all([type(r)==float for r in ratio])
        def start_indices():
            wc = FileHandler.len(fi)
            index, starts = 1, []
            for r in ratio:
                starts.append(index)
                index += int(wc * r)


