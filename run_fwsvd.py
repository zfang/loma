import json

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import LlamaConfig
from transformers import LlamaForCausalLM

from loma import LomaConfig
from loma import LomaForCausalLM
from utils.common import empty_cuda_cache
from utils.common import perplexity
from utils.common import set_seed
from utils.data import get_datasets
from utils.fwsvd import fwsvd_model_copy

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


def run(args):
    bnb_config = None

    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    set_seed(args.seed)
    llama = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", quantization_config=bnb_config)
    llama.eval()
    llama.seqlen = args.seqlen
    if args.fast_tokenizer:
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path)
    else:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)

    tokenized_datasets = get_datasets(
        name=args.dataset,
        tokenizer=tokenizer,
        train_nsamples=args.train_iter * args.batch_size * 5,
        test_nsamples=args.test_iter * args.batch_size * 5,
        seqlen=args.seqlen
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_data_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    llama_train_nlls = []
    train_iter = min(len(tokenized_datasets["train"]), args.train_iter)
    llama.eval()
    for i, data in tqdm(
        enumerate(train_data_loader),
        total=train_iter,
        desc="Computing Llama loss and gradient on train set",
    ):
        if i >= args.train_iter:
            break

        loss, _, _ = llama.forward(return_dict=False, **data)
        loss.backward()
        neg_log_likelihood = loss.float().cpu().detach() * args.seqlen
        llama_train_nlls.append(neg_log_likelihood)

        if args.log_wandb:
            wandb.log({"fwsvd/llama_train_loss": loss})

        empty_cuda_cache()

    llama_train_perplexity = perplexity(
        llama_train_nlls,
        train_iter * args.batch_size,
        args.seqlen
    )

    llama_config = LlamaConfig.from_pretrained(args.model_name_or_path)
    loma_config_kwargs = {}
    if args.loma_rank is not None:
        loma_config_kwargs = dict(
            self_attn_q_proj_r=args.loma_rank,
            self_attn_k_proj_r=args.loma_rank,
            self_attn_v_proj_r=args.loma_rank,
            self_attn_o_proj_r=args.loma_rank,
            mlp_gate_proj_r=args.loma_rank,
            mlp_up_proj_r=args.loma_rank,
            mlp_down_proj_r=args.loma_rank,
        )

    loma_config = LomaConfig.from_llama_config(llama_config, **loma_config_kwargs)
    loma = LomaForCausalLM(loma_config)
    loma.eval()
    loma.seqlen = args.seqlen
    fwsvd_model_copy(llama.model, loma.model, gradient_scale=args.train_iter, show_progress=True)

    optimizer = SGD(llama.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    empty_cuda_cache()

    test_data_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    llama_test_nlls = []
    loma_test_nlls = []
    test_iter = min(len(tokenized_datasets["test"]), args.train_iter)
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(test_data_loader),
            total=test_iter,
            desc="Computing loss and perplexity scores on test set",
        ):
            if i >= args.test_iter:
                break

            llama_loss, _, _ = llama.forward(return_dict=False, **data)
            llama_test_nlls.append(llama_loss.float().cpu().detach() * args.seqlen)

            loma_loss, _, _ = llama.forward(return_dict=False, **data)
            loma_test_nlls.append(loma_loss.float().cpu().detach() * args.seqlen)

            if args.log_wandb:
                wandb.log({
                    "fwsvd/llama_test_loss": llama_loss,
                    "fwsvd/loma_test_loss": loma_loss,
                })

            empty_cuda_cache()

    llama_test_perplexity = perplexity(
        llama_test_nlls,
        test_iter * args.batch_size,
        args.seqlen
    )
    loma_test_perplexity = perplexity(
        loma_test_nlls,
        test_iter * args.batch_size,
        args.seqlen
    )

    stats = {
        "fwsvd/llama_train_perplexity": llama_train_perplexity,
        "fwsvd/llama_test_perplexity": llama_test_perplexity,
        "fwsvd/loma_test_perplexity": loma_test_perplexity,
    }

    if args.log_wandb:
        wandb.log(stats)
    else:
        print(json.dumps(stats, indent=4))

    if args.model_save_dir:
        loma.save_pretrained(save_directory=args.model_save_dir, safe_serialization=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="c4", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model_save_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_iter", type=int, default=1000)
    parser.add_argument("--test_iter", type=int, default=1000)
    parser.add_argument("--loma_rank", type=int)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast_tokenizer", action="store_true")
    parser.add_argument("--log_wandb", action="store_true", help="Whether to log to wandb.")

    quantization_group = parser.add_mutually_exclusive_group()
    quantization_group.add_argument("--load_in_8bit", action="store_true")
    quantization_group.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()
    if args.log_wandb:
        wandb.init(
            project="loma",
            config=args.__dict__
        )
    else:
        print(json.dumps(args.__dict__, indent=4))

    run(args)
