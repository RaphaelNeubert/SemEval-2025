from dataclasses import dataclass

from model import ModelConfig
from training import TrainingConfig
from data import DataConfig
import torch


@dataclass
class Config:
    load_weights: bool = False
    load_weights_from: str = "./weights/weights-pre-fine-orig-834.pth"

    load_pretrain_weights: bool = False
    #load_pretrain_weights_from: str = "./weights/pretrain_weights-490000.pth"
    load_pretrain_weights_from: str = "./weights/weights-pre-fine-goemo-31500.pth"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device: str = "cpu"
    #                           Anger     Fear      Joy      Sadness   Surprise
    #label_set_thresholds = [0.8, 0.1, 0.8, 0.4, 0.3]
    label_set_thresholds = [0.5]*5

    model_config = ModelConfig(
        num_classes = 5,
        dim_embeddings = 768,
        num_heads = 12,
        num_encoder_layers = 12,
        dropout = 0.4
    )
    #model_config = ModelConfig(
    #    num_classes = 5,
    #    dim_embeddings = 256,
    #    num_heads = 4,
    #    num_encoder_layers = 4,
    #    dropout = 0.4
    #)
    finetune_config = TrainingConfig(
        device = device,
        learning_rate = 0.00001,
        num_epochs = 40,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 139,         # Evaluate the model every eval_interval steps
        save_interval = 139,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/weights-pre-fine-orig-<training_step>.pth", # <training_step> placeholder will be replaced
        unfreeze_count = 4,
        #loss_label_weights = (6.24, 7.73, 2.80, 3.19, 21.59)
        #loss_label_weights = (2.01, 0.41, 0.99, 0.76, 0.80)
        loss_label_weights = (7.312312, 0.718187, 3.106825, 2.152620, 2.299166)
        #loss_label_weights = (1,1,1,1,1)

    )
    pretraining_config = TrainingConfig(
        device = device,
        learning_rate = 0.00001,
        num_epochs = 20000,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 500,         # Evaluate the model every eval_interval steps
        save_interval = 10000,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/pretrain_weights-<training_step>.pth", # <training_step> placeholder will be replaced
    )
    data_config = DataConfig(
        finetuning_h5_path = "data/finetuning_split_orig.h5",
        pretraining_h5_path="data/sentiment140.h5",
        pretraining_mask_selection_prob=0.1,
        pretraining_mask_mask_prob=0.8,
        pretraining_mask_random_selection_prob=0.1,
        load_vocab_from_disk=True,
        vocab_path="data/vocab2.json",
        vocab_generation_text_path="data/books_large.txt",
        vocab_generation_num_sentences=10000000, # num of sentences randomly sampled for vocab generation
        vocab_size=30000,
        pretraining_batch_size_train=512,
        pretraining_batch_size_eval=512,
        finetune_batch_size_train=16,
        finetune_batch_size_eval=128,
        finetune_batch_size_test=32,
        pretraining_mask_token="<MASK>",
        pretraining_label_mask_token="<LABEL_MASK>",
        pad_token="<PAD>",
        unk_token="<UNK>"

    )
