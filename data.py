
import random
import re
from itertools import chain
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from dataclasses import dataclass
import json
import h5py
from tqdm import tqdm
import numpy as np

@dataclass
class DataConfig:
    train_path: str = "data/merged_dataset_train.csv"
    valid_path: str = "data/merged_dataset_valid.csv"
    test_path: str =  "data/merged_dataset_test.csv"
    pretraining_path: str = "data/books_large.txt"
    pretraining_h5_path: str = "data/pretraining_split.h5"
    pretraining_mask_selection_prob: float = 0.1
    pretraining_mask_mask_prob: float = 0.8
    pretraining_mask_random_selection_prob: float = 0.1
    load_vocab_from_disk: bool = False
    vocab_path: str = "data/vocab.json"
    vocab_generation_text_path: str = "data/books_large.txt"
    vocab_generation_num_sentences: int = 10000000 # num of sentences that get randomly sampled for vocab generation
    vocab_size: int = 30000
    pretraining_batch_size_train: int = 64
    pretraining_batch_size_eval: int = 64
    finetune_batch_size_train: int = 64
    finetune_batch_size_eval: int = 64
    finetune_batch_size_test: int = 64
    pretraining_mask_token: str = "<MASK>"
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


class Vocabulary:
    # <PAD> token always gets index 0
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>", mask_token: str = "<MASK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.special_tokens = [pad_token, unk_token, mask_token]
        self.word_to_index = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.index_to_word = list(self.word_to_index.keys())

    def build(self, data: list[list[str]], vocab_size: int):
        words = chain.from_iterable(data)
        counter = Counter(words).most_common(vocab_size-len(self.special_tokens))

        self.word_to_index.update({word: idx + len(self.special_tokens) for idx, (word, _) in enumerate(counter)})
        self.index_to_word = list(self.word_to_index.keys())

    def words_to_indices(self, words: list[str]) -> list[int]:
        return [self.word_to_index.get(word, self.word_to_index.get(self.unk_token)) for word in words]

    def index_to_words(self, indices: list[int]) -> list[str]:
        return [self.index_to_word[idx] for idx in indices]

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                "word_to_index": self.word_to_index,
                "index_to_word": self.index_to_word
            }, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.word_to_index = data["word_to_index"]
        vocab.index_to_word = data["index_to_word"]
        return vocab

    def size(self):
        return len(self.index_to_word)

    def print_batch(self, batch: torch.Tensor):
        for sample in batch:
            indices = sample[~(sample==0)]  # remove padding
            print(self.index_to_words(indices))

def load_data(datapath: str) -> list[list[str, list[int]]]:
    with open(datapath) as f:
        f.readline() # ignore first line 
        data = f.readlines()
    data = [[l.split('\t')[1], list(map(int, l[:-1].split('\t')[2:]))] for l in data]
    return data


def process_text(data: list[list[str, list[int]]]) -> list[list[list[str], list[int]]]:
    for d in data:
        d[0] = d[0].lower()
        d[0] = re.sub(r'([.,!?()"\'])', r' \1 ', d[0]) # add whitespace around punctuation
        d[0] = d[0].split()
    return data

def finetune_prep_batch(batch):
    inputs = [torch.tensor(x[0]) for x in batch]
    targets = torch.tensor([x[1] for x in batch])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # corresponds to <PAD>
    mask = ~(inputs_padded == 0)
    return inputs_padded, mask, targets

def get_data(config: DataConfig) -> (DataLoader, DataLoader, DataLoader): 
    train_data = load_data(config.train_path)
    eval_data = load_data(config.valid_path)
    test_data = load_data(config.test_path)

    train_tokens = process_text(train_data)
    eval_tokens = process_text(eval_data)
    test_tokens = process_text(test_data)

    vocab = Vocabulary(pad_token=config.pad_token, unk_token=config.unk_token)
    text = [t[0] for t in train_tokens+eval_tokens]
    vocab.build(text, vocab_size=config.vocab_size)

    train_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in train_tokens]
    eval_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in eval_tokens]
    test_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in eval_tokens]

    trainloader = DataLoader(train_indices, batch_size=config.finetune_batch_size_train, collate_fn=finetune_prep_batch, shuffle=True)
    evalloader = DataLoader(eval_indices, batch_size=config.finetune_batch_size_eval, collate_fn=finetune_prep_batch, shuffle=True)
    testloader = DataLoader(test_indices, batch_size=config.finetune_batch_size_test, collate_fn=finetune_prep_batch, shuffle=True)

    return trainloader, evalloader, testloader

def get_vocab(config: DataConfig):
    if config.load_vocab_from_disk:
        vocab = Vocabulary.load(config.vocab_path)
    else:
        with open(config.vocab_generation_text_path, 'r') as f: 
            text = f.readlines()
        text = random.choices(text, k=config.vocab_generation_num_sentences)
        vocab = Vocabulary(pad_token=config.pad_token, unk_token=config.unk_token)
        vocab.build([t.split() for t in text], vocab_size=config.vocab_size)
        vocab.save(config.vocab_path)
    return vocab


def generate_pretraining_data(config: DataConfig, vocab: Vocabulary):
    with open(config.pretraining_path) as f:
        total_lines = sum(1 for _ in f)
        print("total_lines:", total_lines)
        f.seek(0)
        with h5py.File(config.pretraining_h5_path, "w") as h5f:
            dt = h5py.special_dtype(vlen=int)
            ds = h5f.create_dataset("data", shape=(total_lines,), dtype=dt)

            for i, line in enumerate(tqdm(f, total=total_lines)):
                indices = vocab.words_to_indices(line.split())
                ds[i] = indices


def pretrain_prep_batch(batch: list[np.array], vocab: Vocabulary,
                        mask_selection_prob: float, mask_mask_prob: float, mask_random_replace_prob: float):

    pad_token_id = vocab.word_to_index[vocab.pad_token]
    mask_token_id = vocab.word_to_index[vocab.mask_token]

    inputs = [torch.from_numpy(arr) for arr in batch]
    targets = [t.clone() for t in inputs]
    for t in inputs:
        # select words with mask_prob probability
        mask_words = torch.rand(t.shape) < mask_selection_prob
        mask_indices = torch.nonzero(mask_words, as_tuple=True)[0]
        # from the selected words, mask away mask_mask_prob of them
        rnd_nums = torch.rand(mask_words.sum())
        mask_away_mask  = rnd_nums < mask_mask_prob
        t[mask_indices] = torch.where(mask_away_mask, mask_token_id, t[mask_words])
        # from the selected words, randomly replace mask_random_replace_prob of them
        random_replace_mask = (rnd_nums >= mask_mask_prob) & (rnd_nums < mask_mask_prob + mask_random_replace_prob)
        random_tokens = torch.randint(len(vocab.special_tokens), vocab.size(), (random_replace_mask.sum(),))
        t[mask_indices[random_replace_mask]] = random_tokens

    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
    input_mask = ~(inputs == pad_token_id)
    return inputs, input_mask, targets

def load_pretraining_data(config: DataConfig, vocab: Vocabulary):

    h5f = h5py.File(config.pretraining_h5_path, "r")
    training_ds = h5f["training"]

    validation_ds = h5f["validation"]
    test_ds = h5f["test"]

    trainloader = DataLoader(training_ds, batch_size=config.pretraining_batch_size_train, shuffle=True, num_workers=2,
                             collate_fn=lambda batch: pretrain_prep_batch(batch, vocab, 
                                                                          config.pretraining_mask_selection_prob,
                                                                          config.pretraining_mask_mask_prob,
                                                                          config.pretraining_mask_random_selection_prob))
    validloader = DataLoader(validation_ds, batch_size=config.pretraining_batch_size_eval, num_workers=2,
                             collate_fn=lambda batch: pretrain_prep_batch(batch, vocab, 
                                                                          config.pretraining_mask_selection_prob,
                                                                          config.pretraining_mask_mask_prob,
                                                                          config.pretraining_mask_random_selection_prob))
    testloader = DataLoader(test_ds, batch_size=config.pretraining_batch_size_eval, num_workers=0,
                             collate_fn=lambda batch: pretrain_prep_batch(batch, vocab, 
                                                                          config.pretraining_mask_selection_prob,
                                                                          config.pretraining_mask_mask_prob,
                                                                          config.pretraining_mask_random_selection_prob))
    return trainloader, validloader, testloader


def split_hdf5_data(h5_file_path, train_ratio=0.98, val_ratio=0.01):
    with h5py.File(h5_file_path, "r") as h5f:
        dataset = h5f["data"]
        total_size = len(dataset)
        
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        with h5py.File("data/pretraining_split.h5", "w") as split_h5f:
            dt = h5py.special_dtype(vlen=int)
            valid_ds = split_h5f.create_dataset("validation", shape=(len(valid_indices),), dtype=dt)
            for i, valid_idx in enumerate(tqdm(valid_indices, desc="generating valid split")):
                valid_ds[i] = dataset[valid_idx]
            test_ds = split_h5f.create_dataset("test", shape=(len(test_indices),), dtype=dt)
            for i, test_idx in enumerate(tqdm(test_indices, desc="generating test split")):
                test_ds[i] = dataset[test_idx]
            train_ds = split_h5f.create_dataset("training", shape=(len(train_indices),), dtype=dt)
            for i, train_idx in enumerate(tqdm(train_indices, desc="generating training split")):
                train_ds[i] = dataset[train_idx]
        
        print(f"Data split completed: {train_size} training, {val_size} validation, {test_size} test samples.")

#split_hdf5_data("/home/raphael/tmp/home/raphael/pretraining.h5")
