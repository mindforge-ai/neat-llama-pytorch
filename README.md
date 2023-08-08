# neat-llama-pytorch

Minimal, annotated implementation of LLaMA and LLaMA-2 in PyTorch. STILL IN DEVELOPMENT.

This is a lightly modified version of Meta's [original `llama` repository](https://github.com/facebookresearch/llama).

## Notes:

I tried changing Meta's original code to replace the `fairscale` parallel linear layers / embedding layers with `nn.Linear` and `nn.Embedding` respectively, but received different generation results. So they have to stay for now.

# Commands:

Run the inference script with `torchrun --nproc_per_node 1 scripts/infer.py`.