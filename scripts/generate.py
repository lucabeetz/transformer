import sys
from src.model import GPT2
from src.bpe import BPETokenizer

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python generate.py <text>')
        sys.exit(1)

    text = sys.argv[1]

    model = GPT2.from_pretrained()
    model.eval()

    tokenizer = BPETokenizer()

    encoded = tokenizer.encode(text)
    preds = model.generate(encoded, max_new_tokens=25, top_k=10)
    out = tokenizer.decode(preds[0])

    print(out)
