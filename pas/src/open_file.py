import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_name', help='file_name')
    parser.add_argument("sent_id")
    parser.set_defaults(no_thres=False)
    return parser

def get_file(file_name):
    try:
        return "/home/miyawaki_shumpei/PAS/train-with-juman/" + file_name
    except FileNotFoundError:
        print("FILE がないヨ")
        pass

def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    c = 0
    for line in open(get_file(args.file_name), "r"):
        #print(line)
        if "EOS" in line:
            c += 1

        if int(args.sent_id) == c:
            print(line)


if __name__ == "__main__":
    run()

