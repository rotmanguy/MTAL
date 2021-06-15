"""
Concatenates all treebanks together
"""

import os
import shutil
import logging
import argparse
import random
from io_ import io_utils

random.seed(1)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=str, help="The path to output the concatenated files")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.3", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--languages", default=[], type=str, nargs="+",
                    help="Specify a list of languages to use; leave blank to default to all languages available")
parser.add_argument("--training_set_size", default=500, type=int,
                    help="Specify the training set size you wish")

args = parser.parse_args()
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

treebanks = io_utils.get_ud_treebank_files_by_language(args.dataset_dir, args.languages)

for language, language_treebanks in treebanks.items():
    lng_dir = os.path.join(args.output_dir, language + '_' + str(args.training_set_size))
    if not os.path.isdir(lng_dir):
        os.mkdir(lng_dir)
    train, dev, test = list(zip(*[language_treebanks[k] for k in language_treebanks]))

    sentences_dict = {}
    for treebank, name in zip([train, dev, test], ["train.conllu", "dev.conllu", "test.conllu"]):
        for t in treebank:
            if not t:
                continue
            sentences = []
            sentence = []
            with open(t, 'r') as f:
                for l in f.readlines():
                    line = l.strip()
                    if len(line) == 0:
                        sentences.append(sentence)
                        sentence = []
                    else:
                        sentence.append(line)
            if name != 'test.conllu':
                random.shuffle(sentences)
                sentences_dict[name] = sentences[:args.training_set_size]
                if 'unlabeled.conllu' not in sentences_dict:
                    sentences_dict['unlabeled.conllu'] = sentences[args.training_set_size:]
                else:
                    sentences_dict['unlabeled.conllu'] = sentences_dict['unlabeled.conllu'] + sentences[args.training_set_size:]
            else:
                sentences_dict[name] = sentences

    for name, sentences in sentences_dict.items():
        with open(os.path.join(lng_dir, name), 'w') as f:
            for sentence in sentences:
                for line in sentence:
                    f.write(line + '\n')
                f.write('\n')

