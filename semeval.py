import argparse
import torch
import torch.nn.functional as F
from data import load_finetuning_data, get_vocab, load_pretraining_data
from model import SemEvalModel, PretrainModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training import finetuning, pretraining
from config import Config
import re
from submit import submit

torch.set_printoptions(profile="full")


def init_argparser():
    parser = argparse.ArgumentParser(description="Transformer based semantic evaluation")

    parser.add_argument("-t", "--enable-tqdm", action="store_true", help="Finetune model")
    parser.add_argument("-f", "--finetune", action="store_true", help="Finetune model")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run model queries interactively")
    parser.add_argument("-s", "--submit", action="store_true", help="Run model to label submission file")
    parser.add_argument("-p", "--pretraining", action="store_true", help="Run pretraining")
    parser.add_argument("-l", "--log", action="store_true", help="Enable tensorboard logging")
    return parser

def interactive(model, vocab, label_set_thresholds):
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
        pred_classes = (pred > torch.tensor(label_set_thresholds, device=device))
        prediction_words = " ".join([class_labels[i] for i in range(len(pred_classes)) if pred_classes[i]==True])
        print(prediction_words)


if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()

    config = Config()
    print(config.device)

    writer = SummaryWriter() if args.log else None
    disable_tqdm = not args.enable_tqdm

    vocab = get_vocab(config.data_config)


    if args.pretraining:
        trainloader, validloader, testloader = load_pretraining_data(config.data_config, vocab)

        model = PretrainModel(vocab.size(), config.model_config).to(config.device)
        pretraining(config.pretraining_config, model, trainloader, validloader, 
                    mask_token_id=vocab.word_to_index[vocab.mask_token], log_writer=writer, disable_tqdm=disable_tqdm)

    if args.finetune or args.interactive or args.submit:
        model = SemEvalModel(vocab.size(), config.model_config).to(config.device)
        if config.load_weights:
            model.load_state_dict(torch.load(config.load_weights_from, weights_only=True, map_location=torch.device(config.device)))
        elif config.load_pretrain_weights:
            pretrain_state_dict = torch.load(config.load_pretrain_weights_from, weights_only=True, map_location=torch.device(config.device))
            pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if not k.startswith('fc.')} # remove linear layer weights
            model.load_state_dict(pretrain_state_dict, strict=False)

    if args.finetune:
        trainloader, evalloader = load_finetuning_data(config.data_config, vocab)
        finetuning(config.finetune_config, model, trainloader, evalloader, config.label_set_thresholds,
                   log_writer=writer, print_test_evals=True, vocab=vocab, disable_tqdm=disable_tqdm)

    elif args.interactive:
        interactive(model, vocab, config.label_set_thresholds)
    elif args.submit:
        submit(model, vocab, config.label_set_thresholds)

