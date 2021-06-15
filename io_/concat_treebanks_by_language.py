"""
Concatenates all treebanks together
"""

import os
import shutil
import logging
import argparse

from io_ import io_utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=str, help="The path to output the concatenated files")
parser.add_argument("--dataset_dir", default="data/ud-treebanks-v2.3", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--languages", default=[], type=str, nargs="+",
                    help="Specify a list of languages to use; leave blank to default to all languages available")

args = parser.parse_args()
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

treebanks = io_utils.get_ud_treebank_files_by_language(args.dataset_dir, args.languages)

for language, language_treebanks in treebanks.items():
    lng_dir = os.path.join(args.output_dir, language)
    if not os.path.isdir(lng_dir):
        os.mkdir(lng_dir)
    train, dev, test = list(zip(*[language_treebanks[k] for k in language_treebanks]))

    for treebank, name in zip([train, dev, test], ["train.conllu", "dev.conllu", "test.conllu"]):
        with open(os.path.join(lng_dir, name), 'w') as f:
            for t in treebank:
                if not t:
                    continue
                with open(t, 'r') as read:
                    shutil.copyfileobj(read, f)
