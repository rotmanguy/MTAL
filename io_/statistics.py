from os.path import join
import matplotlib.pyplot as plt
import numpy as np

ontonotes_path = 'data/ontonotes'
domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
splits = ['train', 'dev', 'test']

all_lengths = {}
for domain in domains:
    domain_lengths = {}
    for split in splits:
        lengths = {}
        domain_dir = join(ontonotes_path, domain)
        file = join(domain_dir, split + '.conllu')
        with open(file, 'r') as f:
            len_sentence = 0
            for row in f.readlines():
                row = row.strip().split()
                len_row = len(row)
                if len_row == 0:
                    if len_sentence in lengths:
                        lengths[len_sentence] += 1
                    else:
                        lengths[len_sentence] = 1
                    len_sentence = 0
                else:
                    len_sentence += 1
        domain_lengths[split] = sorted(lengths.items(), key=lambda kv: kv[0])
    all_lengths[domain] = domain_lengths

fig, axs = plt.subplots(len(domains), 3, figsize=(20,20))
for d_idx, domain in enumerate(domains):
    for idx, split in enumerate(splits):
        x, y = list(zip(*all_lengths[domain][split]))
        axs[d_idx, idx].bar(x, y)
        axs[d_idx, idx].set_title(domain + '_' + split)
#fig.show()
plt.tight_layout()
plt.savefig('sentence_lengths_ontonotes.png')
