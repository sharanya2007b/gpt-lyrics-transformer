import os
import json
from pathlib import Path

import numpy as np
import tiktoken


TOKENIZER = tiktoken.get_encoding("gpt2")
EOT = TOKENIZER.encode(
    "<|endoftext|>",
    allowed_special={"<|endoftext|>"}
)[0]


def collect_text_files(folder):
    return list(Path(folder).rglob("*.txt"))


def tokenize_poetry(source_folder):

    print("\nTokenizing poetry corpus...")

    poem_paths = collect_text_files(source_folder)

    print(f"Discovered {len(poem_paths)} poem files.")

    token_buffer = []

    for poem_path in poem_paths:

        try:
            raw_text = poem_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if len(raw_text) == 0:
                continue

            encoded = TOKENIZER.encode(
                raw_text,
                allowed_special={"<|endoftext|>"}
            )

            token_buffer.extend(encoded)
            token_buffer.append(EOT)

        except Exception as err:
            print(f"Failed processing {poem_path}: {err}")

    return token_buffer


def build_condition_header(meta):

    genre = meta.get("genre", "unknown")
    mood = meta.get("mood", "unknown")
    rhyme = meta.get("rhyme_scheme", "unknown")
    cadence = meta.get("cadence", "unknown")

    return (
        f"[Genre: {genre}] "
        f"[Mood: {mood}] "
        f"[Rhyme: {rhyme}] "
        f"[Cadence: {cadence}]\n"
    )


def tokenize_rap_dataset(json_path):

    print("\nTokenizing conditioned rap dataset...")

    dataset_tokens = []

    if not os.path.exists(json_path):
        print("Rap dataset not found.")
        return dataset_tokens

    with open(json_path, "r", encoding="utf-8") as handle:
        songs = json.load(handle)

    print(f"Loaded {len(songs)} rap entries.")

    for item in songs:

        metadata = item.get("annotations", {})
        lyrics = item.get("lyrics", "").strip()

        if not lyrics:
            continue

        prefix = build_condition_header(metadata)

        combined_text = prefix + lyrics

        encoded = TOKENIZER.encode(
            combined_text,
            allowed_special={"<|endoftext|>"}
        )

        dataset_tokens.extend(encoded)
        dataset_tokens.append(EOT)

    return dataset_tokens


def split_data(sequence, split_ratio=0.9):

    cutoff = int(len(sequence) * split_ratio)

    return sequence[:cutoff], sequence[cutoff:]


def export_binary(token_sequence, output_name):

    array_data = np.asarray(token_sequence, dtype=np.uint16)

    array_data.tofile(output_name)

    return array_data


def process_and_save(tokens, train_file, val_file, label):

    train_tokens, val_tokens = split_data(tokens)

    train_arr = export_binary(train_tokens, train_file)
    val_arr = export_binary(val_tokens, val_file)

    print(
        f"{label} -> "
        f"train: {len(train_arr):,} | "
        f"val: {len(val_arr):,}"
    )

    if len(train_arr) > 0:
        print(
            f"{label} -> "
            f"unique tokens: {len(np.unique(train_arr)):,} | "
            f"max token id: {np.max(train_arr)}"
        )


def main():

    print(f"GPT2 Vocabulary Size: {TOKENIZER.n_vocab}")

    poem_tokens = tokenize_poetry(
    "/kaggle/input/datasets/sharanyabhatia/data-task3"
    )

    process_and_save(
        poem_tokens,
        "poems_train.bin",
        "poems_val.bin",
        "Poems"
    )
    print("\nSample GPT2 Tokens:")
    print(poem_tokens[:50])
    
    decoded = TOKENIZER.decode(poem_tokens[:50])
    
    print("\nDecoded Sample:")
    print(decoded)

    rap_tokens = tokenize_rap_dataset(
    "/kaggle/input/datasets/sharanyabhatia/data-task3/cleaned_combined.json"
    )

    process_and_save(
        rap_tokens,
        "raps_train.bin",
        "raps_val.bin",
        "Raps"
    )


if __name__ == "__main__":
    main()