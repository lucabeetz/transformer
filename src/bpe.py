import os
import json
import torch
import requests
import regex as re


def bytes_to_unicode():
    # 188 bytes that render fine and don't need shifting
    fine_bytes = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'),
                                                                ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    byte_mappings = fine_bytes.copy()

    n = 0
    for b in range(256):
        if b not in fine_bytes:
            fine_bytes.append(b)
            byte_mappings.append(256 + n)
            n += 1

    byte_mappings = [chr(b) for b in byte_mappings]
    return dict(zip(fine_bytes, byte_mappings))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder():
    """
    Byte pair encoder
    Original: https://github.com/openai/gpt-2/blob/master/src/encoder.py 
    """

    def __init__(self, encoder, bpe_merges):
        # Splitting pattern
        self.splitting_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.encoder = encoder
        self.decoder = {v: k for k, v in encoder.items()}

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                # No more bigrams are eligible to be merged
                break

            first, second = bigram

            # Replace all occurences of (first, second) in the list of current words
            # into one merged token first_second
            new_word = []
            i = 0
            while i < len(word):
                # Find next occurence of first in sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)

        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_idx = []
        tokens = re.findall(self.splitting_pattern, text)

        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_idx = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_idx)

        return bpe_idx

    def encode_debug(self, text):
        # Store debug info
        parts = []

        bpe_idx = []
        tokens = re.findall(self.splitting_pattern, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_idx = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_idx)

            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_idx': token_idx
            })

        out = {
            'bpe_idx': bpe_idx,
            'tokens': tokens,
            'parts': parts
        }

        return out

    def decode(self, bpe_idx):
        tokens_merged = [self.decoder[idx] for idx in bpe_idx]
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text


def download_file(remote_file, local_file):
    if not os.path.exists(local_file):
        print(f'Downloading {remote_file} to {local_file}')
        res = requests.get(remote_file)
        open(local_file, 'wb').write(res.content)


def get_encoder():
    encoder_local_file = 'encoder.json'
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    download_file(encoder_remote_file, encoder_local_file)

    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)

    assert len(encoder) == 50_257

    vocab_local_file = 'vocab.bpe'
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    download_file(vocab_remote_file, vocab_local_file)

    with open(vocab_local_file, 'r', encoding='utf-8') as f:
        bpe_data = f.read()

    # Remove first and last lines
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50_000

    encoder = Encoder(encoder, bpe_merges)
    return encoder


class BPETokenizer():
    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text):
        idx = [self.encoder.encode(text)]
        tensor = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        text = self.encoder.decode(idx.tolist())
        return text


if __name__ == '__main__':
    text = 'Hey everyone, Transformers are fun! 🤗'

    e = get_encoder()
    out = e.encode_debug(text)

    print(f'Input: {text}')
    print(f'After tokenization using regex: {out["tokens"]}')

    for part in out['parts']:
        print(part)

    print(f'Final encoding: {out["bpe_idx"]}')
