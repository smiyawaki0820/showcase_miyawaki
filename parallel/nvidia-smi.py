import os
import sys
import json
from pprint import pprint
import argparse
import subprocess


DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)


def rename_dkey(d, rename_dict, default_value=None):
    for old_key, new_key in rename_dict.items():
        d[new_key] = d.pop(old_key, default_value)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-fo', '--fo', default=None, type=str)
    parser.add_argument('-p', '--print', action='store_true')
    parser.set_defaults(no_thres=False)
    return parser


def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    gpu_info = get_gpu_info()

    if args.print:
        pprint(gpu_info)

    if args.fo:
        with open(args.fo, 'w') as f_json:
            for line in gpu_info:
                rename_dkey(line, {'memory.total':'memory_total', 'memory.free':'memory_free', 'memory.used':'memory_used'})
                json.dump(line, f_json, ensure_ascii=False)
                f_json.write('\n')


if __name__ == '__main__':
    run()
