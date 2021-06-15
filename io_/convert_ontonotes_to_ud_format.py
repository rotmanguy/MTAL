from os import mkdir
from os.path import join, exists

ontonotes_path = 'data/ontonotes'
domains = ['all', 'bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
splits = ['train', 'dev', 'test']
for domain in domains:
    for split in splits:
        sentences = []
        file = join(ontonotes_path, 'ontonotes_pos_ner_dp_' + split + '_' + domain + '.txt')
        with open(file, 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.strip().split()
                if len(sentence) < 6:
                    row = []
                else:
                    idx = sentence[0]
                    word = sentence[1]
                    if word == '/.':
                        word = '.'
                    if word == '/?':
                        word = '?'
                    pos = sentence[2]
                    ner = sentence[3]
                    head_idx = sentence[4]
                    head_tag = sentence[5]
                    row = [idx, word, '_', pos, '_', '_', head_idx, head_tag, '_', '_', ner]
                sentences.append('\t'.join(row))
        domain_dir = join(ontonotes_path, domain)
        if not exists(domain_dir):
            mkdir(domain_dir)
        with open(join(domain_dir, split + '.conllu'), 'w') as f:
            for item in sentences:
                f.write("%s\n" % item)