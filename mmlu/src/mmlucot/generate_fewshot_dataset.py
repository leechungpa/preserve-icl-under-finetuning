import argparse
import random
from typing import List, Dict, Iterable
import copy
import json
import re

from utils import LETTER
from utils.jsonl import save_jsonl, read_jsonl
from log_or_pad_tokens import align_question_tokens


SYSTEM_PROMPT = {
    "multiple_choice": (
        "[Query]\n"
        "You are solving the multiple-choice question. For each question:\n"
        "- Show your reasoning first, then give the final answer on a new line in this format: 'Answer: <choice>', where <choice> is one of A, B, C, or D.\n"
        "- If sample queries and answers are provided, study them first and match their reasoning style, structure, and level of detail in your responses.\n"
    ),
    "word_count": (
        "[Query]\n"
        "You are solving the multiple-choice question. For each question:\n"
        "- Count the number of words inside the quotes after 'Question:', where a word is any sequence of characters separated by spaces.\n"
        "- Give the final answer on a new line in this format: 'Answer: <choice>', where <choice> is one of A, B, C, or D.\n"
        "- If sample queries and answers are provided, review them first.\n"
    )
}

def get_last_sentence(text: str, placeholder="dummy") -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences or not sentences[-1].strip():
        return "Parsing failed, so this sentence was generated."
    last = sentences[-1].strip()
    return re.sub(r'_+', placeholder, last)

def count_words(text):
    return len(re.sub(r'[^\w\s]', '', text).strip().split())



def render_qa(s, qa_type="multiple_choice") -> str:
    if qa_type == "multiple_choice":
        # Question
        lines = [SYSTEM_PROMPT[qa_type], f"Subject: {s['subject']}", f"Question: {s['question']}"]
        for i, c in enumerate(s["choices"]):
            lines.append(f"{LETTER[i]}) {c}")
        # Answer
        if "cot_answer" in s.keys():
            answer = "\nReasoning: "+s["cot_answer"]
        else:
            answer = f"Answer: {LETTER[s['answer_idx']]}"
    elif qa_type == "word_count":
        # Question
        last_sentece = get_last_sentence(s["question"])
        lines = [SYSTEM_PROMPT[qa_type], f"Question: '{last_sentece}'"]
        # # Answer
        # answer = f"Answer: {count_words(last_sentece)}
        ###############
        # v9
        # Question
        lines = lines + ["A) 0–9", "B) 10–19", "C) 20–29", "D) 30 or more"]
        # Answer
        def to_choice(n):
            return "A" if n<10 else "B" if n<20 else "C" if n<30 else "D"
        answer = f"Answer: {to_choice(count_words(last_sentece))}"
        ###############
    return "\n".join(lines), answer


def make_record(
    target,
    shots,
    n_dummy_shots=0,
    token_length_for_shot=None,
    qa_type="multiple_choice",
) -> Dict:
    input_msg = []
    # dummy
    if n_dummy_shots > 0:
        input_msg.append("0" * token_length_for_shot * (n_dummy_shots - len(shots)))
    # few-shot
    for ex in shots:
        question, answer = render_qa(ex, qa_type=qa_type)
        if token_length_for_shot is not None:
            question, answer = align_question_tokens(question, answer, token_length_for_shot)
        input_msg.append("\n".join((question, answer)))
    # question
    question, answer = render_qa(target, qa_type=qa_type)
    input_msg.append(question)

    return {
        "system": "",
        "input": "\n\n".join(input_msg),
        "output": answer,
        "subject": target["subject"],
        "category": target["category"],
    }

def make_jsonl(
    dataset: List,
    shot_dataset: List,
    qa_type: str,
    n_shots: int,
    n_dummy_shots: int,
    token_length_for_shot: int, 
    subset_category: None|str = None,
    seed: int = 0,
) -> Iterable[Dict]:
    random.seed(seed)
    for t in dataset:
        if (subset_category is None) or (t["category"] == subset_category):
            shots = []
            if n_shots > 0:
                # same subject shots
                same_subject = [s for s in shot_dataset if s["subject"] == t["subject"] and s["question"] != t["question"]]
                pool = list(same_subject)
                if len(pool) < n_shots:
                    same_category = [s for s in shot_dataset if s["category"] == t["category"] and s["question"] != t["question"] and s not in pool]
                    random.shuffle(same_category)
                    pool.extend(same_category[: n_shots - len(pool)])
                # # same category shots
                # same_category = [s for s in shot_dataset if s["category"] == t["category"] and s["question"] != t["question"]]
                # pool = same_category if same_category else [s for s in shot_dataset if s["question"] != t["question"]]
                random.shuffle(pool)
                shots = pool[: min(n_shots, len(pool))]

            yield make_record(target=t, shots=shots, n_dummy_shots=n_dummy_shots, token_length_for_shot=token_length_for_shot, qa_type=qa_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--org_path_few_shot_data", type=str, default=None)
    ap.add_argument("--qa_type", type=str, default="multiple_choice")
    ap.add_argument("--n_shots", type=int, default=0)
    ap.add_argument("--n_dummy_shots", type=int, default=0)
    ap.add_argument("--token_length_for_shot", type=int, default=None)
    ap.add_argument("--subset_category", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset = read_jsonl(args.org_path)
    if args.org_path_few_shot_data is None:
        shot_dataset = dataset
    else:
        shot_dataset = read_jsonl(args.org_path_few_shot_data)

    rows = list(make_jsonl(
        dataset=dataset, 
        shot_dataset=shot_dataset,
        qa_type=args.qa_type,
        n_shots=args.n_shots, n_dummy_shots=args.n_dummy_shots,
        token_length_for_shot=args.token_length_for_shot,
        subset_category=args.subset_category, seed=args.seed,
    ))
    save_jsonl(args.out_path, rows)
    print(f"saved {len(rows)} -> {args.out_path}")


if __name__ == "__main__":
    main()
