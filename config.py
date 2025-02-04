from dataclasses import dataclass

from model import ModelConfig
from training import TrainingConfig
from data import DataConfig
import torch


@dataclass
class Config:
    load_weights: bool = False
    load_weights_from: str = "./weights/weights-pre-fine-goemo2-13000.pth"

    load_pretrain_weights: bool = True
    load_pretrain_weights_from: str = "./weights/pretrain_weights-490000.pth"

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device: str = "cpu"
    #                           Anger     Fear      Joy      Sadness   Surprise
    #label_set_thresholds = [ 0.160263, 0.129352, 0.357197, 0.313231, 0.046307]
    label_set_thresholds = [0.5]*28

    model_config = ModelConfig(
        num_classes = 28,
        dim_embeddings = 768,
        num_heads = 12,
        num_encoder_layers = 12,
        dropout = 0.3
    )
    finetune_config = TrainingConfig(
        device = device,
        learning_rate = 0.00001,
        num_epochs = 50,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 200,         # Evaluate the model every eval_interval steps
        save_interval = 500,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/weights-pre-fine-goemo-<training_step>.pth", # <training_step> placeholder will be replaced
        unfreeze_count = 4,
        #loss_label_weights = (6.24, 7.73, 2.80, 3.19, 21.59)
        #loss_label_weights = (2.01, 0.41, 0.99, 0.76, 0.80)
        loss_label_weights = (1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1)
        #loss_label_weights = (9.538843, 17.841435, 26.669688, 16.504301, 13.844073, 38.964812, 31.108481, 18.732121, 66.454420, 33.392254,
        #                      20.335518, 52.315502, 142.638235, 50.084728, 71.458457, 15.203384, 587.397590, 29.277123, 20.014200, 260.160428, 
        #                      26.638370, 383.543307, 37.913944, 296.786585, 80.259567, 31.953441, 39.663614, 2.051168)


    )
    pretraining_config = TrainingConfig(
        device = device,
        learning_rate = 0.0001,
        num_epochs = 20000,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 500,         # Evaluate the model every eval_interval steps
        save_interval = 10000,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/pretrain_weights-<training_step>.pth", # <training_step> placeholder will be replaced
    )
    data_config = DataConfig(
        finetuning_h5_path = "data/finetuning_split_goemotion.h5",
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
        finetune_batch_size_train=64,
        finetune_batch_size_eval=64,
        finetune_batch_size_test=64,
        pretraining_mask_token="<MASK>",
        pretraining_label_mask_token="<LABEL_MASK>",
        pad_token="<PAD>",
        unk_token="<UNK>"

    )
