import argparse
import torch
import torch.nn.functional as F
from data import get_data, get_vocab, generate_pretraining_data, load_pretraining_data
from model import SemEvalModel, PretrainModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training import training, pretraining
from config import Config
import re
from submit import submit

torch.set_printoptions(profile="full")


def init_argparser():
    parser = argparse.ArgumentParser(description="Transformer based semantic evaluation")

    parser.add_argument("-t", "--training", action="store_true", help="Run training")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run model queries interactively")
    parser.add_argument("-s", "--submit", action="store_true", help="Run model to label submission file")
    parser.add_argument("-p", "--pretraining", action="store_true", help="Run pretraining")
    parser.add_argument("-l", "--log", action="store_true", help="Enable tensorboard logging")
    return parser

def interactive(model, vocab):
    model.eval()
    device = next(model.parameters()).device
    class_labels = ["anger", "fear", "joy", "sadness", "suprise"]
    while True:
        sentence = input("Enter text:").lower()
        sentence = re.sub(r'([.,!?()"\'])', r' \1 ', sentence).split()# add whitespace around punctuation
        indices = torch.tensor(vocab.words_to_indices(sentence), device=device).unsqueeze(0)
        print(indices)
        pred = torch.sigmoid(model(indices).squeeze())
        print("pred_probs:", pred)
        pred_classes = (pred > 0.5)
        prediction_words = " ".join([class_labels[i] for i in range(len(pred_classes)) if pred_classes[i]==True])
        print(prediction_words)


if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()

    config = Config()
    print(config.device)
    writer = SummaryWriter() if args.log else None

    vocab = get_vocab(config.data_config)
    #generate_pretraining_data(config.data_config, vocab)
    #trainloader, evalloader, testloader = get_data(config.data_config)

    #model = SemEvalModel(vocab.size(), config.model_config).to(config.device)
    #if config.load_weights:
    #    model.load_state_dict(torch.load(config.load_weights_from, weights_only=True, map_location=torch.device(config.device)))

    if args.pretraining:
        trainloader, validloader, testloader = load_pretraining_data(config.data_config, vocab)

        model = PretrainModel(vocab.size(), config.model_config).to(config.device)
        pretraining(config.pretraining_config, model, trainloader, validloader, log_writer=writer)
    if args.training:
        training(config.training_config, model, trainloader, evalloader, 
                 log_writer=writer, print_test_evals=True, vocab=vocab)
    elif args.interactive:
        interactive(model, vocab)
    elif args.submit:
        submit(model, vocab)

