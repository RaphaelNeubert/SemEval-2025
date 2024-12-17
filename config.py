from dataclasses import dataclass
from model import ModelConfig
from training import TrainingConfig
from data import DataConfig
import torch


@dataclass
class Config:
    load_weights: bool = False
    load_weights_from: str = "./weights/weights-44000.pth"

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        num_classes = 5,
        dim_embeddings = 512,
        num_heads = 8,
        num_encoder_layers = 6,
        dropout = 0.3
    )
    training_config = TrainingConfig(
        device = device,
        learning_rate = 0.0001,
        num_epochs = 20,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 1000,         # Evaluate the model every eval_interval steps
        save_interval = 1000,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/weights-<training_step>.pth", # <training_step> placeholder will be replaced
        disable_tqdm = True
    )
    pretraining_config = TrainingConfig(
        device = device,
        learning_rate = 0.0001,
        num_epochs = 20,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 1000,         # Evaluate the model every eval_interval steps
        save_interval = 1000,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/pretrain_weights-<training_step>.pth", # <training_step> placeholder will be replaced
        disable_tqdm = False
    )
    data_config = DataConfig(
        train_path="data/merged_dataset_train.csv",
        valid_path="data/merged_dataset_valid.csv",
        test_path="data/merged_dataset_test.csv",
        pretraining_path="data/books_large.txt",
        pretraining_h5_path="data/pretraining_split.h5",
        pretraining_mask_selection_prob=0.1,
        pretraining_mask_mask_prob=0.8,
        pretraining_mask_random_selection_prob=0.1,
        load_vocab_from_disk=True,
        vocab_path="data/vocab.json",
        vocab_generation_text_path="data/books_large.txt",
        vocab_generation_num_sentences=10000000, # num of sentences randomly sampled for vocab generation
        vocab_size=30000,
        pretraining_batch_size_train=512,
        pretraining_batch_size_eval=512,
        finetune_batch_size_train=64,
        finetune_batch_size_eval=64,
        finetune_batch_size_test=64,
        pretraining_mask_token="<MASK>",
        pad_token="<PAD>",
        unk_token="<UNK>"
    )
