from pathlib import Path
import numpy as np
import json


def load_poetry_text(folder):

    text_data = ""

    for file in Path(folder).rglob("*.txt"):

        try:
            content = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            text_data += content + "\n"

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return text_data


def build_character_vocab(text):

    chars = sorted(list(set(text)))

    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}

    return chars, stoi, itos


def encode(text, stoi):

    return [stoi[c] for c in text]


def decode(tokens, itos):

    return "".join([itos[t] for t in tokens])


def split_dataset(data, ratio=0.9):

    cutoff = int(len(data) * ratio)

    return data[:cutoff], data[cutoff:]


def save_binary(data, filename):

    arr = np.array(data, dtype=np.uint16)

    arr.tofile(filename)


def main():

    print("Loading text corpus...")

    corpus = load_poetry_text(
    "/kaggle/input/datasets/sharanyabhatia/data-task3"
    )

    print(f"Corpus length: {len(corpus):,} characters")

    chars, stoi, itos = build_character_vocab(corpus)

    vocab_size = len(chars)

    print(f"Character Vocabulary Size: {vocab_size}")

    print("Building character mappings...")

    with open("char_stoi.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=4)

    with open("char_itos.json", "w", encoding="utf-8") as f:
        json.dump(itos, f, ensure_ascii=False, indent=4)

    print("Encoding corpus...")

    encoded_data = encode(corpus, stoi)

    train_ids, val_ids = split_dataset(encoded_data)

    save_binary(train_ids, "char_train.bin")
    save_binary(val_ids, "char_val.bin")

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Validation tokens: {len(val_ids):,}")

    sample = decode(encoded_data[:200], itos)

    print("\nDecoded Sample:\n")
    print(sample)

    print("\nSample Encoded Tokens:")
    print(encoded_data[:50])

    print("\nDecoded Back:")
    print(decode(encoded_data[:50], itos))


if __name__ == "__main__":
    main()

