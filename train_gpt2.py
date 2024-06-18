import tiktoken
import time
import torch
import math
import os
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from torch.distributed import init_process_group, destroy_process_group

#setup the DDP (Distributed Data Parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1 # check if we are in a DDP context
if ddp:
    #use of DDP atm demands CUDA, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "for now i think DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # global rank of the process: ex:- GPU 0, GPU 1, etc.
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # local rank of the GPU on single node (a node refers to a single computer or server that is part of a larger network or cluster of computers)
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of processes
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc
else:
    #vanilla, non-DDP training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autoconnect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"running on device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2^19, ~0.5M tokens
B = 4 # micro batch size
T = 256 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure the total batch size is divisible by B * T * ddp_world_size"
grad_acc_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size:,} | micro batch size: {B} | sequence length: {T} | gradient accumulation steps: {grad_acc_steps}")

print("GPU", ddp_rank)
import sys; sys.exit(0)

train_loader = DataLoaderLite(B=B, T=T)

torch.set_float32_matmul_precision("high")
    
model = GPT(GPTConfig(vocab_size=50304))
model = model.to(device)
# model = torch.compile(model)

max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
def get_lr(step):
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2) if step > lr_decay_iters, return min_lr
    if step > max_steps:
        return min_lr
    # 3) cosine decay from lr_max to lr_min
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff is in [0, 1]
    return min_lr + (max_lr - min_lr) * coeff

# optimize
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and get the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_acc_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss_accum.item():.6f} | learning_rate: {lr:.4f} |norm: {norm:.4f} | dt: {dt:.2f} s | tokens/sec: {tokens_per_sec:.2f}")


import sys; sys.exit(0)

model.eval()
num_return_sequences = 5
max_length = 30


tokens = enc.encode('Hello, I am a language model')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, T)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # do topf sampling of 50 (from huggingface pipeline default)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
        # select a token from the topk options
        ix =  torch.multinomial(topk_probs, 1)
        # gather the corresponding token indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1) # output = torch.gather(input, dim, index)
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)