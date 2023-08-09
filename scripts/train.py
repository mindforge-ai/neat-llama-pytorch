import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from neat_llama_pytorch import TransformerDecoder, Tokenizer, LlamaConfig
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import os
import bitsandbytes as bnb

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenized_text = tokenizer.encode(text, bos=True, eos=True)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokenized_text) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokenized_text[idx : idx + self.block_size + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data file')
    parser.add_argument('--seed', type=int, default=65536)
        
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
    
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    torch.manual_seed(args.seed)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # set to half to use float16

    tokenizer = Tokenizer(model_path="../tokenizer.model")
    config = LlamaConfig(max_seq_len=128, vocab_len=tokenizer.n_words)
    model = TransformerDecoder(
        vocab_len=config.vocab_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        norm_eps=config.norm_eps,
        max_batch_size=config.max_batch_size,
        max_seq_len=config.max_seq_len,
        multiple_of=config.multiple_of,
        ffn_dim_multiplier=config.ffn_dim_multiplier
    ).to(args.device)

    model.train()

    checkpoint = torch.load(
        "../llama-2-7b/consolidated.00.pth", map_location=args.device
    )
    checkpoint.pop("rope.freqs")
    model.load_state_dict(checkpoint)

    # model = torch.compile(model)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)

    train_dataset = TextDataset(args.train_data, tokenizer, config.max_seq_len)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, generator=generator)

    task1 = progress.add_task("Training...", total=args.num_epochs * len(train_loader))

    with progress:
        for epoch in range(args.num_epochs):
            for batch in train_loader:
                inputs, targets = batch[0].to(args.device), batch[1].to(args.device)
                logits = model(inputs, 0)
                loss = F.cross_entropy(logits.view(-1, config.vocab_len), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
                progress.update(task1, advance=1)

    torch.save(model.state_dict(), "trained_model.pth")
