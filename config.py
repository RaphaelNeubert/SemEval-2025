from dataclasses import dataclass

from model import ModelConfig
from training import TrainingConfig
from data import DataConfig
import torch


@dataclass
class Config:
    load_weights: bool = False
    load_weights_from: str = "./weights/self-pretrained-finetuned.pth"

    load_pretrain_weights: bool = False
    load_pretrain_weights_from: str = "./weights/pre-goemotion.pth"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_set_thresholds = [0.5]*5

    model_config = ModelConfig(
        num_classes = 5,
        dim_embeddings = 768,
        num_heads = 12,
        num_encoder_layers = 12,
        dropout = 0.4
    )
    finetune_config = TrainingConfig(
        device = device,
        learning_rate = 0.00002,
        num_epochs = 40,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 139,         # Evaluate the model every eval_interval steps
        save_interval = 139,         # Save weights every save_interval steps
        save_weights = False,
        save_weights_to = "./weights/weights-pre-emo-fine-orig-<training_step>.pth", # <training_step> placeholder will be replaced
        unfreeze_count = 2,
        loss_label_weights = (7.312312, 0.718187, 3.106825, 2.152620, 2.299166)
    )
    pretraining_config = TrainingConfig(
        device = device,
        learning_rate = 0.00001,
        num_epochs = 20000,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 500,         # Evaluate the model every eval_interval steps
        save_interval = 10000,         # Save weights every save_interval steps
        save_weights = False,
        save_weights_to = "./weights/pretrain_weights-v2-<training_step>.pth", # <training_step> placeholder will be replaced
    )
    data_config = DataConfig(
        finetuning_h5_path = "data/finetuning_split_orig.h5",
        pretraining_h5_path="data/sentiment140-v2.h5",
        pretraining_mask_selection_prob=0.1,
        pretraining_mask_mask_prob=0.8,
        pretraining_mask_random_selection_prob=0.1,
        pretraining_batch_size_train=512,
        pretraining_batch_size_eval=512,
        finetune_batch_size_train=16,
        finetune_batch_size_eval=128,
    )
