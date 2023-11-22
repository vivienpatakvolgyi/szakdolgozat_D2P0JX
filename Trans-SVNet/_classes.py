# some code adapted from https://wmathor.com/index.php/archives/1455/
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from dotenv import load_dotenv
from torch.utils.data import Dataset


class Params:
    def __init__(self) -> None:
        if os.path.isfile(".env"):
            load_dotenv()
        else:
            print('Cannot find environment file\n')
        self._gpu: int = os.getenv('GPU')
        self._train_set: int = os.getenv('TRAIN_SET')
        self._valid_set: int = os.getenv('VALID_SET')
        self._test_set: int = os.getenv('TEST_SET')
        self._sequence: int = os.getenv('SEQUENCE')
        self._train_batch: int = os.getenv('TRAIN_BATCH')
        self._valid_batch: int = os.getenv('VALID_BATCH')
        self._test_batch: int = os.getenv('TEST_BATCH')
        self._epochs: int = os.getenv('EPOCHS')
        self._workers: int = os.getenv('WORKERS')
        self._lr: float = os.getenv('LR')
        self._lr_decay: float = os.getenv('LR_DECAY')
        # self._momentum: float = os.getenv('MOMENTUM')
        self._weight_decay: float = os.getenv('WEIGHTDECAY')
        self._dampening: float = os.getenv('DAMPENING')
        self._classes: int = os.getenv('NUM_CLASSES')
        self._data_dir: str = os.getenv('DATA_DIR')
        self._log_dir: str = os.getenv('LOG_DIR')
        self._num_gpu: int = torch.cuda.device_count()
        self._use_gpu: bool = torch.cuda.is_available()
        self._optimizer: str = os.getenv('OPTIMIZER')
        self._resize: float = os.getenv('IMG_RESIZE')
        self._train_resize: float = os.getenv('TRAIN_RESIZE')
        self._test_resize: float = os.getenv('TEST_RESIZE')

    @property
    def get_gpu(self):
        return int(self._gpu)

    @property
    def get_train_set(self):
        return int(self._train_set)

    @property
    def get_valid_set(self):
        return int(self._valid_set)

    @property
    def get_test_set(self):
        return int(self._test_set)

    @property
    def get_sequence(self):
        return int(self._sequence)

    @property
    def get_train_batch(self):
        return int(self._train_batch)

    @property
    def get_valid_batch(self):
        return int(self._valid_batch)

    @property
    def get_test_batch(self):
        return int(self._test_batch)

    @property
    def get_epochs(self):
        return int(self._epochs)

    @property
    def get_workers(self):
        return int(self._workers)

    @property
    def get_lr(self):
        return float(self._lr)

    @property
    def get_lr_decay(self):
        return float(self._lr_decay)

    # @property
    # def get_momentum(self):
    #   return float(self._momentum)

    @property
    def get_weight_decay(self):
        return float(self._weight_decay)

    @property
    def get_dampening(self):
        return float(self._dampening)

    @property
    def get_classes(self):
        return int(self._classes)

    @property
    def get_data_dir(self):
        return str(self._data_dir)

    @property
    def get_log_dir(self):
        return str(self._log_dir)

    @property
    def get_num_gpu(self):
        return int(self._num_gpu)

    @property
    def get_use_gpu(self):
        return bool(self._use_gpu)

    @property
    def get_optimizer(self):
        return str(self._optimizer)

    @property
    def get_resize(self):
        return float(self._resize)

    @property
    def get_train_resize(self):
        return float(self._train_resize)

    @property
    def get_test_resize(self):
        return float(self._test_resize)


class ResizeImg(object):
    def __init__(self, factor, padding=0):
        self.count = 0
        self.factor = factor

    def __call__(self, img):
        self.count += 1
        img = ImageOps.scale(img, self.factor)
        return img


class CholecDataset(Dataset):
    def pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels

    def __len__(self):
        return len(self.file_paths)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_q]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_intpus: [batch_size, src_len, d_model]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = dec_inputs  # self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs


class Transformer2_3_1(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Transformer2_3_1, self).__init__()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q).cuda()
        self.decoder = Decoder(d_model, d_ff, d_k, d_v, 1, n_heads, len_q).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        return dec_outputs
