import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from neat_llama_pytorch import TransformerDecoder, Tokenizer, LlamaConfig


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample(
    model,
    tokenizer,
    config,
    text_prompts: List[str],
    device="cuda",
    show_progress: bool = True,
):
    if show_progress:
        text_column = TextColumn("{task.description}")
        bar_column = BarColumn(bar_width=None)
        m_of_n_complete_column = MofNCompleteColumn()
        time_elapsed_column = TimeElapsedColumn()
        time_remaining_column = TimeRemainingColumn()

        progress = Progress(
            text_column,
            bar_column,
            m_of_n_complete_column,
            time_elapsed_column,
            time_remaining_column,
            expand=True,
        )

    logprobs = False
    temperature = 0.9
    max_gen_len = 4095
    top_p = 1

    with progress if show_progress else nullcontext():
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in text_prompts]
        print(prompt_tokens)
        bsz = len(prompt_tokens)
        assert bsz <= config.max_batch_size, (bsz, config.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= config.max_seq_len
        total_len = min(config.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = model(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if True else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="Georges Seurat painting of a lemur on Saturn",
    )

    with torch.inference_mode():
        args = parser.parse_args()

        tokenizer = Tokenizer(model_path="../tokenizer.model")
        config = LlamaConfig(max_seq_len=2048, vocab_len=tokenizer.n_words)
        model = TransformerDecoder(
            vocab_len=config.vocab_len,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            norm_eps=config.norm_eps,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            multiple_of=config.multiple_of,
        ).to(args.device)

        checkpoint = torch.load(
            "../llama-2-7b/consolidated.00.pth", map_location=args.device
        )
        checkpoint.pop("rope.freqs")
        model.load_state_dict(checkpoint)

        generations = sample(
            model=model,
            tokenizer=tokenizer,
            config=config,
            text_prompts=[args.text_prompt],
        )
