import argparse
import torch
import torch.nn.functional as F
from data import load_finetuning_data, get_vocab, load_pretraining_data
from model import SemEvalModel, PretrainModel, SemEvalBertModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training import finetuning, pretraining, finetune_evaluate
from config import Config
import re
from submit import submit
from transformers import AutoTokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_printoptions(profile="full")
torch.set_float32_matmul_precision('high')

def init_argparser():
    parser = argparse.ArgumentParser(description="Transformer based semantic evaluation")

    parser.add_argument("-t", "--enable-tqdm", action="store_true", help="Enable TQDM progress bar")
    parser.add_argument("-f", "--finetune", action="store_true", help="Finetune model")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Run finetune evaluation on the evaluation dataset")
    parser.add_argument("-s", "--submit", action="store_true", help="Run model to label submission file")
    parser.add_argument("-p", "--pretraining", action="store_true", help="Run pretraining")
    parser.add_argument("-l", "--log", action="store_true", help="Enable tensorboard logging")
    return parser


if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()

    config = Config()
    print(config.device)

    writer = SummaryWriter() if args.log else None
    disable_tqdm = not args.enable_tqdm


    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if args.finetune or args.submit or args.evaluate:
        model = SemEvalBertModel(config.model_config).to(config.device)
   #     #model = torch.compile(model)
        if config.load_weights:
            print("loading weights")
            model.load_state_dict(torch.load(config.load_weights_from, weights_only=True, map_location=torch.device(config.device)))

        elif config.load_pretrain_weights:
            print("loading pretrain weights")
            pretrain_state_dict = torch.load(config.load_pretrain_weights_from, weights_only=True, map_location=torch.device(config.device))
            pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if not k.startswith('fc.')} # remove linear layer weights
            model.load_state_dict(pretrain_state_dict, strict=False)

    if args.finetune:
        trainloader, evalloader = load_finetuning_data(config.data_config, tokenizer)
        finetuning(config.finetune_config, model, trainloader, evalloader, config.label_set_thresholds,
                   log_writer=writer, print_test_evals=True, tokenizer=tokenizer, disable_tqdm=disable_tqdm)
    if args.evaluate:
        _, validloader = load_finetuning_data(config.data_config, tokenizer)
        eval_loss, acc, precision, recall, f1 = finetune_evaluate(model, validloader, config.label_set_thresholds)
        print(f"Eval loss: {eval_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

    elif args.submit:
        submit(model, tokenizer, config.label_set_thresholds)

