import os
import argparse
import random
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable

from datasets import load_dataset

from utils import SUBJECT_TO_CAT
from utils.jsonl import save_jsonl


@dataclass
class Sample:
    subject: str
    category: str
    question: str
    choices: List[str]
    answer_idx: int


def norm_sample(raw: dict, subject_to_cat: Dict[str, str]) -> Sample:
    subject = raw.get("subject", "unknown")
    category = subject_to_cat.get(subject, "Other")

    return Sample(
        subject=subject,
        category=category,
        question=raw["question"],
        choices=list(raw["choices"]),
        answer_idx=int(raw["answer"]),
    )

def _get_balanced_n(n_total: int, categorys: list, by_cat: Dict) -> Dict[str, int]:
    k = len(categorys)
    base = n_total // k
    rem = n_total % k
    targets = {c: base for c in categorys}

    for c in random.sample(categorys, rem):
        targets[c] += 1
    for c in categorys:
        targets[c] = min(targets[c], len(by_cat[c]))
    deficit = n_total - sum(targets.values())

    if deficit > 0:
        pool = [c for c in categorys if len(by_cat[c]) > targets[c]]
        while deficit > 0 and pool:
            c = random.choice(pool)
            if targets[c] < len(by_cat[c]):
                targets[c] += 1
                deficit -= 1
            else:
                pool.remove(c)
    return targets


def split_balanced_by_category(
    all_samples: List[Sample],
    train_size: int = 1000,
    test_size: int = 1000,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample]]:
    random.seed(seed)
    
    categorys = ['STEM', 'Social Sciences', 'Humanities', 'Other']
    by_cat: Dict[str, List[Sample]] = {}
    for s in all_samples:
        by_cat.setdefault(s.category, []).append(s)

    # sample train
    train_n_by_cat = _get_balanced_n(train_size, categorys, by_cat)

    train: List[Sample] = []
    remaining: Dict[str, List[Sample]] = {}
    for c in categorys:
        cand = by_cat[c][:]
        random.shuffle(cand)
        k = train_n_by_cat[c]
        train.extend(cand[:k])
        remaining[c] = cand[k:]

    # sample test
    test_n_by_cat = _get_balanced_n(test_size, categorys, remaining)

    test: List[Sample] = []
    for c in categorys:
        k = test_n_by_cat.get(c, 0)
        test.extend(remaining[c][:k])

    return train, test


def load_all_test_and_split(
    train_size: int = 1000,
    test_size: int = 1000,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample]]:
    ds = load_dataset("cais/mmlu", "all")["test"]
    samples = [norm_sample(x, SUBJECT_TO_CAT) for x in ds]

    # remove duplicates (by subject + answer)
    unique = {}
    for s in samples:
        key = (s.subject, s.choices[s.answer_idx])
        if key not in unique:
            unique[key] = s
    samples = list(unique.values())
    
    return split_balanced_by_category(samples, train_size, test_size, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data/mmlu`")
    ap.add_argument("--train_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    train, test = load_all_test_and_split(args.train_size, args.test_size, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_p = os.path.join(args.out_dir, f"train_n{args.train_size}_seed{args.seed}.jsonl")
    test_p  = os.path.join(args.out_dir, f"test_n{args.test_size}_seed{args.seed}.jsonl")

    save_jsonl(train_p, (asdict(s) for s in train))
    save_jsonl(test_p,  (asdict(s) for s in test))

    print(f"saved {len(train)} -> {train_p}")
    print(f"saved {len(test)}  -> {test_p}")


if __name__ == "__main__":
    main()