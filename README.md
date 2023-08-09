# neat-llama-pytorch

Minimal, annotated implementation of LLaMA and LLaMA-2 in PyTorch. STILL IN DEVELOPMENT.

This is a lightly modified version of Meta's [original `llama` repository](https://github.com/facebookresearch/llama).

## Notes:

I tried changing Meta's original code to replace the `fairscale` parallel linear layers / embedding layers with `nn.Linear` and `nn.Embedding` respectively, but received different generation results. So they have to stay for now.

So training works on the 7B model with default tensor being `FloatTensor` (aka in float32) precision, with bnb's 8-bit Adam. It seems to _just_ fit on an 80gb A100 if the sequence length is 128. So we need further optimisations / loras!

# Commands:

Run the inference script with `torchrun --nproc_per_node 1 scripts/infer.py`.

Run the training script with `torchrun --nproc_per_node 1 scripts/train.py --train-data scripts/test.txt`.