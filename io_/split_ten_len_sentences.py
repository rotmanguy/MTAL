from os.path import join, exists
from os import mkdir

data_path = 'data/ontonotes'
data_path_below_10 = data_path + '_below_10'
data_path_above_10 = data_path + '_above_10'
if not exists(data_path_below_10):
    mkdir(data_path_below_10)
if not exists(data_path_above_10):
    mkdir(data_path_above_10)

domains = ['all', 'bc', 'bn', 'mz', 'nw', 'tc', 'wb']
splits = ['train', 'dev', 'test']

for domain in domains:
    below_10_domain_dir = join(data_path_below_10, domain)
    if not exists(below_10_domain_dir):
        mkdir(below_10_domain_dir)
    above_10_domain_dir = join(data_path_above_10, domain)
    if not exists(above_10_domain_dir):
        mkdir(above_10_domain_dir)
    for split in splits:
        domain_dir = join(data_path, domain)
        file = join(domain_dir, split + '.conllu')
        sentence_below_10 = []
        sentence_above_10 = []
        sentence = []
        with open(file, 'r') as f:
            for row in f.readlines():
                strip_row = row.strip().split()
                if len(strip_row) == 0:
                    if len(sentence) == 0:
                        sentence = []
                        continue
                    if len(sentence) <= 10:
                        for row_ in sentence:
                            sentence_below_10.append(row_)
                        sentence_below_10.append('\n')
                    else:
                        for row_ in sentence:
                            sentence_above_10.append(row_)
                        sentence_above_10.append('\n')
                    sentence = []
                    continue
                sentence.append(row)
            # sentence_below_10.append('\n')
            # sentence_above_10.append('\n')

        with open(join(below_10_domain_dir, split + '.conllu'), 'w') as f:
            for item in sentence_below_10:
                f.write(item)
        with open(join(above_10_domain_dir, split + '.conllu'), 'w') as f:
            for item in sentence_above_10:
                f.write(item)