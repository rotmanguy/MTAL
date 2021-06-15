import os
import pickle
import os

if os.path.exists('hie_dict.pkl'):
    with open('hie_dict.pkl', 'rb') as f:
        hie_dict = pickle.load(f)

    import matplotlib.pyplot as plt

    for key, d in hie_dict.items():
        fig, ax = plt.subplots()
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.bar(list(d.keys()), d.values(), color='g')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(key + '.png')
        plt.show()
        plt.close()
else:
    hie_dict = {'dist_from_root': {}, 'num_siblings': {}, 'rel_pos_enc': {}}
    for split_ in ['train', 'dev', 'test']:
        for batch in dataloaders[split_]:
            for sent in batch['metadata']:
                for task in ['dist_from_root', 'num_siblings', 'rel_pos_enc']:
                    for word in sent[task]:
                        if word in hie_dict[task]:
                            hie_dict[task][word] += 1
                        else:
                            hie_dict[task][word] = 1
    with open('hie_dict.pkl', 'wb') as f:
        pickle.dump(hie_dict, f)
