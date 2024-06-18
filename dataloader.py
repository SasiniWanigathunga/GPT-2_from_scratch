import torch
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open("shakespere.txt", "r") as f:   
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T) # (B, T)
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset the position
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y