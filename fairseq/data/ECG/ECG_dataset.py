"""
for speech command dataset
Adapted from https://github.com/HazyResearch/state-spaces/blob/main/src/dataloaders/sc.py
which is
adapted from https://github.com/dwromero/ckconv/blob/dc84dceb490cab2f2ddf609c380083367af21890/datasets/speech_commands.py
which is
adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

import pathlib
import tarfile
import urllib.request

import sklearn.model_selection
from fairseq.data import fairseq_dataset
# import torchaudio

from .. import FairseqDataset, BaseWrapperDataset

logger = logging.getLogger(__name__)

NORM = "NORM"
MI = "MI"
STTC = "STTC"
HYP = "HYP"
CD = "CD"

def pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[: channel.size(0)] = channel
    return out


def subsample(X, y, subsample_rate):
    if subsample_rate != 1:
        X = X[:, ::subsample_rate, :]
    return X, y


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
        elif filename.endswith(".npy"):
            tensor_name = filename.split(".")[0]
            tensor_value = np.load(str(dir / filename), allow_pickle=True)
            if "X" in tensor_name:
                #print("Tensor_value shape:",  tensor_value.shape)
                tensor_value = tensor_value.transpose((0,2,1)).astype(np.float32)
            tensors[tensor_name] = tensor_value
            
    return tensors


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out

def normalize_all_data(X_train, X_val, X_test):

    for i in range(X_train.shape[-1]):
        mean = X_train[:, :, i].mean()
        std = X_train[:, :, i].std()
        X_train[:, :, i] = (X_train[:, :, i] - mean) / (std + 1e-5)
        X_val[:, :, i] = (X_val[:, :, i] - mean) / (std + 1e-5)
        X_test[:, :, i] = (X_test[:, :, i] - mean) / (std + 1e-5)

    return X_train, X_val, X_test

def minmax_scale(tensor):
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return (tensor - min_val) / (max_val - min_val)

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor(2**bits - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = 2 * minmax_scale(audio) - 1

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Quantize signal to the specified number of levels.
    return ((encoded + 1) / 2 * mu + 0.5).to(torch.int32)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = 2**bits - 1
    # Invert the quantization
    x = (encoded / mu) * 2 - 1

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu
    return x

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor


class ECGDataset(FairseqDataset):

    SUBSET_CLASSES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    # SUBSET_CLASSES = [
    #     "yes",
    #     "no",
    # ]
    ALL_CLASSES = [
        "NORM", 
        "MI",
        "STTC",
        "CD",
        "HYP",
    ]
    label_ids = {"NORM":0, "MI":1, "STTC":2, "CD":3, "HYP":4}
    priorities = {"NORM":4, "MI":0, "STTC":2, "CD":1, "HYP":3}
    

    def __init__(
            self,
            partition: str,  # `train`, `val`, `test`
            length: int, # sequence length
            sr: int,  # subsampling rate: default should be 1 (no subsampling); keeps every kth sample
            path: str,
            all_classes: bool = False,
            gen: bool = False,  # whether we are doing speech generation
            resolution: int = 1,  # resolution of the input
            
    ):
        # compatible with fairseq
        if partition == 'valid':
            partition = 'val'

        self.all_classes = all_classes
        self.gen = gen
        self.resolution = resolution

        self.root = pathlib.Path(path)  # pathlib.Path("./data")
        base_loc = self.root / "processed_data"

        # import pdb; pdb.set_trace()

        if gen:
            data_loc = base_loc / "gen"
        else:
            data_loc = base_loc / "raw"

        
        if self.all_classes:
            data_loc = pathlib.Path(str(data_loc) + "_all_classes")
        # import pdb; pdb.set_trace()

        X, y = self.load_data(data_loc, partition) # (batch, length, 1)
        
        
        if self.gen: 
            y = y.transpose(1, 2)
            
            """
            if not self.gen:
                X = F.pad(X, (0, 0, 0, length-1000))

            """

        # Subsample
        X, y = subsample(X, y, sr)

        # import pdb; pdb.set_trace()
        self.src = X
        self.tgt = y
        # super(SpeechCommands, self).__init__(X, y)

    def __getitem__(self, index):

        example = {
            "id": index,
            "source": self.src[index],
            "target": self.tgt[index],
        }

        return example

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        return len(self.src[index])

    def size(self, index):
        return len(self.src[index])

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        #print("Targets:", targets)
        sizes = [len(s) for s in sources]
        def multilabel_binary(classes, n_classes=len(ECGDataset.ALL_CLASSES)): # TODO: do this during preprocessing
            class_ids = list(map(ECGDataset.label_ids.get, classes))
            label = np.zeros(n_classes)
            label[class_ids] = 1
            #return np.expand_dims(label, -1)
            return label
        
        targets = list(map(multilabel_binary, targets))
        print(targets)
        
        #print("targets3:", targets)
        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                # if torch.utils.data.get_worker_info() is not None:
                #     # If we're in a background process, concatenate directly into a
                #     # shared memory tensor to avoid an extra copy
                #     numel = sum(x.numel() for x in batch)
                #     storage = elem.storage()._new_shared(numel)
                #     out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                #print("batch:", batch)
                batch = torch.tensor(batch)
                if resolution is not None:
                    batch = batch[:, ::resolution] # assume length is first axis after batch
                #print(batch.shape)    
                return batch

        src_lengths = torch.LongTensor(sizes)
        src_tokens = _collate(sources, resolution=self.resolution)
        src_tokens = src_tokens.squeeze(-1)
        target = _collate(targets, resolution=None)

        ntokens = src_lengths.sum().item()

        id = torch.LongTensor([s["id"] for s in samples])

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }

        return batch

    # def set_epoch(self, epoch):
    #     super().set_epoch(epoch)

    def download(self):
        print("ECG download")
        """
        base_loc = self.root / "SpeechCommands"
        loc = base_loc / "speech_commands.tar.gz"
        if os.path.exists(loc):
            return

        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", loc
        )  # TODO: Add progress bar
        with tarfile.open(loc, "r") as f:
            f.extractall(base_loc)
        """

    @staticmethod
    def load_data(data_loc, partition):

        # import pdb; pdb.set_trace()
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))

        return X, y
