import json


def load_vocab(filename="vocab.json"):
    """loads vocab from JSON and make the reverse mapping."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    token_to_id = data["tokens"]
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    return token_to_id, id_to_token


def encode(text, token_to_id):
    """convert text to list from ids. unknown characters are <UNK>."""
    unk_id = token_to_id["<UNK>"]
    return [token_to_id.get(char, unk_id) for char in text]


def decode(ids, id_to_token):
    """convert list from ids back to text. unknown ids are <UNK>."""
    return "".join(id_to_token.get(i, "<UNK>") for i in ids)
