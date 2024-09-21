import os
import pickle
from contextlib import nullcontext
import torch

from model import GPT, GPTConfig


class nanoGPT:

    def __init__(self):

        self.init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        self.out_dir = 'out-function-call-char'  # ignored if init_from is not 'resume'
        self.start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        self.num_samples = 10  # number of samples to draw
        self.max_new_tokens = 500  # number of tokens generated in each sample
        self.temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
        self.seed = 1337
        self.device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
        self.compile = False  # use PyTorch 2.0 to compile the model to be faster

        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.model = None
        self.stoi = None
        self.itos = None

        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # for later use in torch.autocast
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.ptdtype
        )

        self.load_model()
        self.load_tokenizer()

    def load_model(self):

        # init from a model saved in a specific directory
        ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.to(self.device)
        # self.model = torch.compile(self.model)

    def load_tokenizer(self):
        meta_path = os.path.join(self.out_dir, 'meta.pkl')
        print(f"Loading meta from {meta_path}...")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.stoi = meta['stoi']
        self.itos = meta['itos']

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def predict(self, prompt: str):
        start_ids = self.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                print(self.decode(y[0].tolist()))
