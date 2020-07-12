import emoji
import os
import argparse
from tqdm import tqdm
import json


#twiiterの発話ペアを分割する関数
def split_utterance(input_file, output_dir, file_name):
    output_file = os.path.join(output_dir, file_name + "_sentence.txt")
    with open(input_file, "r") as fi, open(output_file, "w") as fo:
        for i, line in enumerate(tqdm(fi)):
            data = json.loads(line)
            if "\n" in data[0]:
                for sent in data[0].split("\n"):
                    fo.write("{}\n".format(sent))
            else:
                fo.write("{}\n".format(data[0]))

            if "\n" in data[1]:
                for sent in data[1].split("\n"):
                    fo.write("{}\n".format(sent))
            else:
                fo.write("{}\n".format(data[1]))


#指定した絵文字が含まれる文を取ってくる関数
def extract_emoji_sentence(input_file, output_dir, filename):
    #emoji_dict = emoji.UNICODE_EMOJI
    emoji_list = ["😢", "😁", "😊", "😠"]
    output_file = os.path.join(output_dir, filename + "_emoji_sentence_undenoized.txt")
    with open(input_file, "r") as fi, open(output_file, "w") as fo:
        for line in tqdm(fi):
            sentence = line.rstrip()
            index_list = []
            for emoji_token in emoji_list:
                index = sentence.find(emoji_token)
                if index != -1:
                    index_list.append(index)
            index_list = sorted(index_list)
            for index in index_list:
                emoji_token = sentence[index]
                fo.write("{}\t{}\n".format(emoji_token, sentence[:index]))#絵文字が出現する直前までの文をその絵文字が表す文とする

#文中の絵文字を取り除く関数
def denoize(input_file, output_dir, file_name):
    emoji_dict = emoji.UNICODE_EMOJI
    output_file = os.path.join(output_dir, file_name + "_emoji_sentence.txt")
    with open(input_file, "r") as fi, open(output_file, "w") as fo:
        for line in tqdm(fi):
            if len(line.rstrip().split("\t")) != 2:
                continue
            label, sentence = line.rstrip().split("\t")
            for emoji_token in emoji_dict.keys():
                if emoji_token in sentence:
                    sentence = sentence.replace(emoji_token, " ")
            fo.write("{}\t{}\n".format(label, sentence))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    output_sub_dir = os.path.join(output_dir, "output_sub_dir")
    os.makedirs(output_sub_dir, exist_ok=True)
    file_name = args.file_name
    split_utterance(input_file, output_sub_dir, file_name)
    input_file = os.path.join(output_sub_dir, file_name + "_sentence.txt")
    extract_emoji_sentence(input_file, output_sub_dir, file_name)
    input_file = os.path.join(output_sub_dir, file_name + "_emoji_sentence_undenoized.txt")
    denoize(input_file, output_dir, file_name)


if __name__ == "__main__":
    main()