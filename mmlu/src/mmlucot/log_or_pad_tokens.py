import argparse
import json
from functools import lru_cache
from pathlib import Path
from statistics import mean
from dataclasses import dataclass

from transformers import AutoTokenizer
import warnings


def calculate_tokens(text: str, model_name: str) -> int:
    if AutoTokenizer is None:
        return len(text.split())

    tokenizer = _get_tokenizer(model_name)
    return len(tokenizer.encode(text))


@lru_cache(maxsize=None)
def _get_tokenizer(model_name: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers library is required to load model tokenizers.")
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_name)



@dataclass
class RandomTokenPrefixBuilder:
    model_name: str
    seed: int

    def __post_init__(self) -> None:
        self._tokenizer = _get_tokenizer(self.model_name)
        self._fixed_token_id = 15

    @property
    def tokenizer(self):
        return self._tokenizer

    def build_prefix_ids(self, length: int) -> list[int]:
        if length <= 0:
            return []
        return [self._fixed_token_id for _ in range(length)]


def align_question_tokens(
    question: str,
    answer: str,
    target_token_length: int,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    seed: int = 0,
) -> None:
    if target_token_length <= 0:
        raise ValueError("--target-tokens must be a positive integer.")

    prefix_builder = RandomTokenPrefixBuilder(model_name=model_name, seed=seed)
    tokenizer = prefix_builder.tokenizer

    combined_tokens = calculate_tokens(question + answer, model_name)

    if combined_tokens < target_token_length:
        ids = prefix_builder.build_prefix_ids(target_token_length - combined_tokens)
        ids += tokenizer.encode(question, add_special_tokens=False)
        question = tokenizer.decode(ids, skip_special_tokens=False)

        new_combined_tokens = calculate_tokens(question + answer, model_name)
        if new_combined_tokens != target_token_length:
            warnings.warn(
                f"Token mismatch: expected {target_token_length}, "
                f"but observed {new_combined_tokens} after padding.",
                UserWarning
            )
    return question, answer
    


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="log_output")
    parser.add_argument("--fields", nargs="+", default=("system", "input", "output"))
    parser.add_argument("--max-combined", type=int, default=None)
    parser.add_argument("--min-combined", type=int, default=None)
    args = parser.parse_args()

    fields = tuple(dict.fromkeys(args.fields))
    filter_requested = args.max_combined is not None or args.min_combined is not None

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input_path.name
    if output_name.endswith(".jsonl"):
        output_name = output_name[: -len(".jsonl")]
    else:
        output_name = input_path.stem
    output_path = output_dir / f"{output_name}_token_lengths.log"

    token_lengths: list[tuple[int, int, dict[str, int]]] = []
    filtered_records: list[str] = []

    with input_path.open("r", encoding="utf-8") as src:
        for idx, raw_line in enumerate(src):
            if not raw_line.strip():
                continue

            record = json.loads(raw_line)
            parts = {name: str(record.get(name, "") or "") for name in fields}
            combined_text = "".join(parts.values())

            lengths = {name: calculate_tokens(text, args.model) for name, text in parts.items()}
            combined_length = calculate_tokens(combined_text, args.model)
            token_lengths.append((idx, combined_length, lengths))
            stripped_line = raw_line.rstrip("\n")
            if filter_requested:
                include_record = True
                if args.max_combined is not None:
                    include_record = include_record and combined_length <= args.max_combined
                if args.min_combined is not None:
                    include_record = include_record and combined_length >= args.min_combined
                if include_record:
                    filtered_records.append(stripped_line)

    token_lengths.sort(key=lambda item: item[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dest:
        for idx, combined_len, lengths in token_lengths:
            parts_detail = "\t".join(f"{name}={lengths[name]}" for name in fields)
            line = f"{idx}\tcombined={combined_len}\t{parts_detail}"
            dest.write(line + "\n")
            print(line)

        lengths_only = [length for _, length, _ in token_lengths]
        average_line = f"average\t{mean(lengths_only):.2f}"
        max_line = f"max\t{max(lengths_only)}"
        min_line = f"min\t{min(lengths_only)}"
        for summary_line in (min_line, average_line, max_line):
            dest.write(summary_line + "\n")
            print(summary_line)

    if filter_requested:
        filtered_path = input_path.parent / f"{output_name}_filtered.jsonl"
        with filtered_path.open("w", encoding="utf-8") as filtered_dest:
            for raw_record in filtered_records:
                filtered_dest.write(raw_record + "\n")
        print(f"Wrote filtered records to {filtered_path}")
        print(f"Filtered record count: original={len(token_lengths)} -> filtered={len(filtered_records)}")


if __name__ == "__main__":
    main()
