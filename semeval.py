import argparse
import torch
import torch.nn.functional as F
from data import get_data
from model import SemEvalModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training import training
from config import Config

torch.set_printoptions(profile="full")


def init_argparser():
    parser = argparse.ArgumentParser(description="Transformer based semantic evaluation")

    parser.add_argument("-t", "--training", action="store_true", help="Run training")
    parser.add_argument("-l", "--log", action="store_true", help="Enable tensorboard logging")
    return parser

if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()

    config = Config()
    writer = SummaryWriter() if args.log else None

    trainloader, evalloader, testloader, vocab = get_data(config.data_config)

    model = SemEvalModel(vocab.size(), config.model_config).to(config.device)
    if config.load_weights:
        model.load_state_dict(torch.load(config.load_weights_from, weights_only=True))

    if args.training:
        training(config.training_config, model, trainloader, evalloader, log_writer=writer)

