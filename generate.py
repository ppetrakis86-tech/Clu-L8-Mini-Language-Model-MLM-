import sys
import os
import random
import argparse
import torch

from tokenizer import load_vocab, encode, decode
from model import MLM


# --- Generation ---

def generate(model, prompt, token_to_id, id_to_token, device, max_new_tokens=32):
    model.eval()
    bos = token_to_id["<BOS>"]
    eos = token_to_id["<EOS>"]

    ids = [bos] + encode(prompt, token_to_id)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids)
            next_id = torch.argmax(logits[0, -1, :]).item()
            if next_id == eos:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    output_ids = ids[0, 1:].tolist()
    return decode(output_ids, id_to_token)


# --- Split line to prompt + expected ---

SEPARATORS = [" = ", "= ", " equals ", " is ", "? ", "=", "?"]

def split_line(line):
    best_pos = -1
    best_sep = None
    for sep in SEPARATORS:
        pos = line.rfind(sep)
        if pos > best_pos or (pos == best_pos and best_sep and len(sep) > len(best_sep)):
            best_pos = pos
            best_sep = sep
    if best_sep is None or best_pos < 0:
        return None, None
    prompt = line[:best_pos + len(best_sep)]
    expected = line[best_pos + len(best_sep):].strip()
    return prompt, expected


# --- Load checkpoint ---

def load_model(model_path, vocab_size, device):
    model = MLM(vocab_size=vocab_size).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        ep = ckpt.get("epoch", "?")
        print(f"✅ Model loaded (epoch {ep})")
    else:
        model.load_state_dict(ckpt)
        print("✅ Model loaded (raw state_dict)")
    return model


# --- Args ---

def parse_args():
    p = argparse.ArgumentParser(description="MLM Clu-L8 evaluation")
    p.add_argument("data", nargs="?", default=None,
                   help="Test data file. Αν λείπει, τρέχει hardcoded prompts.")
    p.add_argument("--lines", type=int, default=10,
                   help="Πόσες τυχαίες γραμμές να δοκιμάσει. Default: 10")
    p.add_argument("--reverse", action="store_true",
                   help="Το μοντέλο είναι reversed — γύρνα output/expected σε κανονική μορφή για ανάγνωση.")
    return p.parse_args()


def unrev(s):
    """Γυρίζει reversed αριθμό σε κανονικό. Πρόσημο μπροστά μένει."""
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


# --- Main ---

def main():
    args = parse_args()
    vocab_file = "vocab.json"
    model_path = "mlm.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    token_to_id, id_to_token = load_vocab(vocab_file)
    model = load_model(model_path, len(token_to_id), device)
    print()

    # --- Mode 1: hardcoded prompts ---
    if args.data is None:
        prompts = [
            "cl: 2 + 3 = ",
            "cl: 15 * 4 = ",
            "cl: 100 - 37 = ",
            "hex 255 = ",
            "bin 10 = ",
            "dec 0xFF = ",
            "cl: what is 8 + 5? ",
            "bf: +++++. = ",
        ]
        for p in prompts:
            result = generate(model, p, token_to_id, id_to_token, device)
            print(f"🧪 '{p}' → '{result}'")
        return

    # --- Mode 2: evaluation ---
    if not os.path.exists(args.data):
        print(f"❌ Δεν βρέθηκε: '{args.data}'")
        return

    with open(args.data, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    sample = random.sample(lines, min(args.lines, len(lines)))

    correct = 0
    for line in sample:
        prompt, expected = split_line(line)
        if prompt is None:
            continue

        full_output = generate(model, prompt, token_to_id, id_to_token, device)
        answer = full_output[len(prompt):].strip()

        ok = (answer == expected)
        correct += ok
        mark = "✅" if ok else "❌"

        is_numeric = prompt.startswith("cl:") or prompt.startswith("dec ")
        if args.reverse and is_numeric:
            human = f"  [normal: '{unrev(answer)}' vs '{unrev(expected)}']"
        else:
            human = ""

        print(f"{mark} '{prompt}' → '{answer}'{human}  (expected: '{expected}')")

    acc = 100 * correct / len(sample)
    print(f"\n📊 Accuracy: {correct}/{len(sample)} = {acc:.1f}%")


if __name__ == "__main__":
    main()
