import os
import re
import time
import argparse, json
import random
from typing import List, Dict, Iterable
from copy import deepcopy

from openai import OpenAI

from utils import LETTER, SUBJECT_TO_CAT
from utils.jsonl import read_jsonl


PROMPT = (
    "You are solving the multiple-choice question. For each question:\n"
    "- Show your reasoning first, then give the final answer on a new line in this format: 'Answer: <choice>', where <choice> is one of A, B, C, or D.\n"
)


def render_qa(s) -> str:
    lines = [f"Subject: {s['subject']}", f"Question: {s['question']}"]
    for i, c in enumerate(s['choices']):
        lines.append(f"{LETTER[i]}) {c}")
    return "\n".join(lines)

def _extract_answer(text: str) -> str:
    if not text:
        return ""
    last_line = text.strip().splitlines()[-1]

    m = re.search(r'(?i)answer\s*:\s*([A-D])\b', last_line)
    if m:
        return m.group(1)
    m2 = re.search(r'(?i)answer\s*:\s*(.+)$', last_line, flags=re.MULTILINE)
    return m2.group(1).strip() if m2 else ""

def ask_llm(
    system_prompt,
    user_prompt,
    model_name:str = "gpt-4.1-mini",
    temperature:float = 0.7,
    max_retries:int = 5,
    verbose:bool = False,
):
    model_prefix = model_name.lower().split("-")[0]

    if model_prefix == "gpt":
        api_base = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
    elif model_prefix == "gemini":
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = model_name.split(":", 1)
    else:
        raise ValueError(f"Unsupported model prefix '{model_prefix}'. Please specify a model starting with 'gpt-' or 'gemini-'.")

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    api_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "timeout": 180.0,
    }
    content = ""

    for attempt in range(max_retries):
        if attempt > 0 and verbose:
            print(f"attempt {attempt}", flush=True)

        try:
            response = client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content
            break
        except Exception as e:
            if verbose:
                print(f"[{attempt}/{max_retries}] LLM did not respond: {e}", flush=True)
            time.sleep(2)

    if verbose:
        print("┌──LLM input───┐")
        print(messages[1]["content"])
        print("├──LLM output──┤")
        print(content)
        print("└──────────────┘")

    return content


def generate_cot_answer(
    sample: Sample,
    model_name: str = "gpt-4.1-mini",
    rate_limit_per_sec: float = 1.0,
    max_retries:int = 5,
    verbose: bool = False,
    return_log: bool = True,
):
    delay = 1.0 / max(rate_limit_per_sec, 1e-6)

    user_prompt = render_qa(sample)
    true_answer = LETTER[sample.answer_idx]

    for attempt in range(max_retries):
        llm_output = ask_llm(
            PROMPT,
            user_prompt,
            model_name=model_name,
            verbose=verbose,
        )
        time.sleep(delay)

        pred = _extract_answer(llm_output)
        if pred == true_answer:
            failure_case = None
            break
        if verbose:
            print(f"[{attempt+1}try] pred({pred})!=true({true_answer})")
    else:
        failure_case = f"|Failed| p:{pred} != t:{true_answer}\n - Input: {user_prompt}\n - Output: {llm_output}"
        if verbose:
            print(failure_case)

    lines = llm_output.strip().splitlines()
    llm_output = "\n".join(lines[:-1] + [f"Answer: {true_answer}"])

    if return_log:
        return llm_output, failure_case
    else:
        return llm_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model_name", default="gpt-4.1-mini")
    args = ap.parse_args()

    verbose = True
    max_retries = 5

    org_dataset = read_jsonl(args.org_path)
    failure_cases = []

    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(org_dataset):
            cot_answer, failure_case = generate_cot_answer(s, model_name=args.model_name, max_retries=max_retries, verbose=verbose, return_log=True)

            if failure_case is not None:
                failure_cases.append(f"|{i}"+failure_case)
            else:
                # only write successful cases
                row = deepcopy(s)
                row["cot_answer"] = cot_answer
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Failure Cases ===")
    print("\n".join(failure_cases))

    print("=== Summary ===")
    print(f"wrote {len(org_dataset)} with cot_answers -> {args.out_path}")

    with open(args.out_path.removesuffix(".jsonl") + ".log", "w", encoding="utf-8") as f:
        f.write("=== Failure Cases ===\n")
        f.write("\n".join(failure_cases))
        f.write("\n\n=== Summary ===\n")
        f.write(f"wrote {len(org_dataset)} with cot_answers -> {args.out_path}\n")


if __name__ == "__main__":
    main()