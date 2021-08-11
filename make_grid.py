from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

imgs_str = [
    'plots/plots_entropy_ls/Num_Sentences_bc_deps.png',
    'plots/plots_entropy_ls/Num_Sentences_bn_deps.png',
    'plots/plots_entropy_ls/Num_Sentences_mz_deps.png',
    'plots/plots_entropy_ls/Num_Sentences_nw_deps.png',
    'plots/plots_entropy_ls/Num_Sentences_bc_ner.png',
    'plots/plots_entropy_ls/Num_Sentences_bn_ner.png',
    'plots/plots_entropy_ls/Num_Sentences_mz_ner.png',
    'plots/plots_entropy_ls/Num_Sentences_nw_ner.png',
]

imgs = [Image.open(file) for file in imgs_str]
grid = image_grid(imgs, rows=2, cols=4)
grid.save('plots/plots_entropy_ls/entropy_sentences_grid.png')
