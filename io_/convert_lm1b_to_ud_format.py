from os import listdir
from os.path import isfile, join

lm1b_path = '/Data/Datasets/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/'
txt_files = [join(lm1b_path, f) for f in listdir(lm1b_path) if isfile(join(lm1b_path, f))]
sentences = []
for file in txt_files:
    with open(file, 'r') as f:
        file_sentences = f.read().splitlines()
        for sentence in file_sentences:
            sentences.append(sentence.split())
