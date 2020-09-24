import pickle
import torch
from tqdm import tqdm
import numpy as np
from datasets.imagenet64 import ImageNet64
from knockoff.victim.blackbox import Blackbox
from datasets import modelfamily_to_transforms

path = "results/models/adversary/try_with_original_test_label/selection.15000.pickle"

indexes = pickle.load(open(path, 'rb'))
queryset = ImageNet64(train=True, transform=modelfamily_to_transforms['custom_cnn']['train'])
device = torch.device('cuda')
blackbox_dir = 'results/models/victim/gtsrb-cnn32'
blackbox = Blackbox.from_modeldir(blackbox_dir, device)
transferset = []
query = []


def get_transferset(budget, idx_set, queryset, blackbox, batch_size=128):
    start_B = 0
    end_B = budget
    with tqdm(total=budget) as pbar:
        for t, B in enumerate(range(start_B, end_B, batch_size)):
            idxs = np.random.choice(list(idx_set), replace=False,
                                    size=min(batch_size, budget - len(transferset)))
            idx_set = idx_set - set(idxs)

            if len(idx_set) == 0:
                print('=> Query set exhausted. Now repeating input examples.')
                idx_set = set(range(len(queryset)))

            x_t = torch.stack([queryset[i][0] for i in idxs]).to(blackbox.device)
            y_t = blackbox(x_t).cpu()

            if hasattr(queryset, 'samples'):
                # Any DatasetFolder (or subclass) has this attribute
                # Saving image paths are space-efficient
                img_t = [queryset.samples[i][0] for i in idxs]  # Image paths
            else:
                # Otherwise, store the image itself
                # But, we need to store the non-transformed version
                img_t = [queryset.data[i] for i in idxs]
                if isinstance(queryset.data[0], torch.Tensor):
                    img_t = [x.numpy() for x in img_t]

            for i in range(x_t.size(0)):
                img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                transferset.append((img_t_i, y_t[i].cpu().squeeze()))

            pbar.update(x_t.size(0))

    return transferset


ts = get_transferset(15000, indexes, queryset, blackbox)

destination = "results/models/adversary/transfer_manner/transferset.7500.pickle"
with open(destination, 'wb') as f:
    pickle.dump(ts, f)

from models import zoo
import torch

checkpoint_path = "results/models/adversary/try_with_original_test_label/checkpoint.28.iter.pth.tar"
model = zoo.get_pretrainednet('CNN32', 'custom_cnn', checkpoint_path, 43)

device = torch.device('cuda')
model = model.to(device)


from datasets.gtsrb import GTSRB

dataset = GTSRB(train=False, transform=modelfamily_to_transforms['custom_cnn']['train'])

def compare_model(model1, model2, dataset):
    size = len(dataset)
    coherence = 0
    for img, target in dataset:
        img = img.unsqueeze(0).to(device)
        y_1 = model1(img).argmax(1).cpu()
        y_2 = model2(img).argmax(1).cpu()
        if y_1 == y_2:
            coherence += 1

    print("{}, {}, {}".format(coherence, size, coherence/size))