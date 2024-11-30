from dataclasses import dataclass
from model import ModelConfig
from training import TrainingConfig
from data import DataConfig
import torch


@dataclass
class Config:
    load_weights: bool = False
    load_weights_from: str = "./weights/weights1.pth"

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(
        num_classes = 5,
        dim_embeddings = 512,
        num_heads = 8,
        num_encoder_layers = 6,
        dropout = 0.1
    )
    training_config = TrainingConfig(
        device = device,
        learning_rate = 0.001,
        num_epochs = 20,
        log_interval = 100,           # Log training loss every log_interval steps
        eval_interval = 1000,         # Evaluate the model every eval_interval steps
        save_interval = 1000,         # Save weights every save_interval steps
        save_weights = True,
        save_weights_to = "./weights/weights-<training_step>.pth", # <training_step> placeholder will be replaced
        disable_tqdm = True
    )
    data_config = DataConfig(
        train_path = "data/merged_dataset_train.csv",
        valid_path = "data/merged_dataset_valid.csv",
        test_path = "data/merged_dataset_test.csv",
        vocab_size = 30000,
        batch_size_train = 64,
        batch_size_eval = 64,
        batch_size_test = 64,
        pad_token = "<PAD>",
        unk_token = "<UNK>"
    )
