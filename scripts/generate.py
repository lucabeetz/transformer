import argparse
from src.model import GPT2
from src.bpe import BPETokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
    parser.add_argument('--new-tokens', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('prompt', type=str)

    args = parser.parse_args()

    text = args.prompt

    model = GPT2.from_pretrained(model_type=args.model)
    model.eval()

    tokenizer = BPETokenizer()

    encoded = tokenizer.encode(text)
    preds = model.generate(encoded, max_new_tokens=args.new_tokens, temp=args.temperature top_k=args.top_k)
    out = tokenizer.decode(preds[0])

    print(out)
