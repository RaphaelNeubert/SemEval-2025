
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
    finetuning_h5_path: str = "data/finetuning_split.h5"
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
    pretraining_label_mask_token: str ="<LABEL_MASK>"
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


class Vocabulary:
    # <PAD> token always gets index 0
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>", mask_token: str = "<MASK>", extra_tokens: list[str] = None):

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.special_tokens = [pad_token, unk_token, mask_token] + (extra_tokens if extra_tokens is not None else [])

    def build(self, data: list[list[str]], vocab_size: int):
        words = chain.from_iterable(data)
        counter = Counter(words).most_common(vocab_size-len(self.special_tokens))

        self.word_to_index = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.index_to_word = [None] * vocab_size
        self.index_to_word[:len(self.special_tokens)] = self.special_tokens

        for idx, (word, _) in enumerate(counter, start=len(self.special_tokens)):
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word

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


def finetune_prep_batch(batch):
    inputs = [torch.tensor(x["feature"]) for x in batch]
    targets = torch.tensor(np.array([x["label"] for x in batch]))
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # corresponds to <PAD>
    mask = ~(inputs_padded == 0)
    return inputs_padded, mask, targets

def load_finetuning_data(config: DataConfig, vocab: Vocabulary):
    h5f = h5py.File(config.finetuning_h5_path, "r")
    training_ds = h5f["training"]
    validation_ds = h5f["validation"]

    trainloader = DataLoader(training_ds, batch_size=config.finetune_batch_size_train, collate_fn=finetune_prep_batch, shuffle=True)
    validloader = DataLoader(validation_ds, batch_size=config.finetune_batch_size_eval, collate_fn=finetune_prep_batch, shuffle=True)

    return trainloader, validloader

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


def pretrain_prep_batch(batch: list[np.array], vocab: Vocabulary, label_mask_token_id: int,
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

        # make sure the label token is always masked away
        t[-1] = label_mask_token_id

    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
    input_mask = ~(inputs == pad_token_id)
    return inputs, input_mask, targets

def load_pretraining_data(config: DataConfig, vocab: Vocabulary):

    h5f = h5py.File(config.pretraining_h5_path, "r")
    training_ds = h5f["training"]

    validation_ds = h5f["validation"]

    trainloader = DataLoader(training_ds, batch_size=config.pretraining_batch_size_train, shuffle=True, num_workers=2,
                             collate_fn=lambda batch: pretrain_prep_batch(batch, vocab, 
                                                                          vocab.word_to_index[config.pretraining_label_mask_token],
                                                                          config.pretraining_mask_selection_prob,
                                                                          config.pretraining_mask_mask_prob,
                                                                          config.pretraining_mask_random_selection_prob))
    validloader = DataLoader(validation_ds, batch_size=config.pretraining_batch_size_eval, num_workers=2,
                             collate_fn=lambda batch: pretrain_prep_batch(batch, vocab, 
                                                                          vocab.word_to_index[config.pretraining_label_mask_token],
                                                                          config.pretraining_mask_selection_prob,
                                                                          config.pretraining_mask_mask_prob,
                                                                          config.pretraining_mask_random_selection_prob))
    return trainloader, validloader

