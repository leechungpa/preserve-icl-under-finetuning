import json
import os
from typing import TYPE_CHECKING, Any, Optional
import re
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file
from prettytable import PrettyTable

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from ..data import Role

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERBOSE = True
NUM_RETURN_SEQUENCES = 10

class Evaluator:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)

    def _parse_answer(self, text: str) -> str:
        try:
            for line in reversed(text.splitlines()):
                match = re.search(r"Answer\s*:\s*([A-D]|\d+)", line, flags=re.IGNORECASE)
                if match:
                    return match.group(1)
            return None
        except Exception:
            return None

    @torch.inference_mode()
    def batch_inference(self, batch_input: dict[str, "torch.Tensor"]) -> tuple[list[list[Optional[str]]], list[list[str]]]:
        gen_kwargs = {
            "tokenizer": self.tokenizer,
            "stop_strings": ["[Query]", "Answer: A", "Answer: B", "Answer: C", "Answer: D"],
            "repetition_penalty": 1.0,
            "temperature": self.eval_temperature,
            "top_k": 20,
            "top_p": 0.95,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "do_sample": True,
        }

        batch_size = batch_input["input_ids"].shape[0]
        input_len = batch_input["input_ids"].shape[1]
        gen_kwargs["max_length"] = input_len+2000

        gen_ids = self.model.generate(**batch_input, **gen_kwargs)
        grouped = [
            gen_ids[i*gen_kwargs["num_return_sequences"]:(i+1)*gen_kwargs["num_return_sequences"]] for i in range(batch_size)
        ]
        texts = [
            [self.tokenizer.decode(ids[input_len:], skip_special_tokens=True) for ids in group]
            for group in grouped
        ]

        parsed = [[self._parse_answer(t) for t in group] for group in texts]
        return parsed, texts

    def _print_args(self) -> None:
        table = PrettyTable()
        table.field_names = ["Argument", "Value"]
        for key, value in vars(self.eval_args).items():
            if value is not None:
                table.add_row([key, value])
        for key, value in vars(self.data_args).items():
            if value is not None:
                table.add_row([key, value])
        print(table)
        return table

    def eval(self) -> None:
        if self.eval_args.save_dir is None:
            raise ValueError("Please specify --save_dir to save evaluation results.")
        else:
            if os.path.exists(os.path.join(self.eval_args.save_dir, 'results.json')):
                warning_path = os.path.join(self.eval_args.save_dir, "warning.log")
                with open(warning_path, "a", encoding="utf-8", newline="\n") as wf:
                    wf.write(f"{datetime.utcnow().isoformat()} - Results already exist. Evaluation skipped.\n")
                    wf.write(f"{self._print_args()}\n")
                return None

        self._print_args()
        eval_n, eval_temperature = self.eval_args.task.split("_")
        self.eval_temperature = float(eval_temperature[1:])

        dataset_org = load_dataset(
            "json",
            data_files=f"{self.eval_args.task_dir}/{eval_n}_shot{self.eval_args.n_shot}.jsonl",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        categories = ['STEM', 'Social Sciences', 'Humanities', 'Other']
        category_corrects = {cat: np.array([], dtype="bool") for cat in categories+["Average"]}

        dataset = {cat: dataset_org.filter(lambda x: x["category"] == cat) for cat in categories}

        if VERBOSE:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)
            self.log_temp_path = os.path.join(self.eval_args.save_dir, "temp_outputs.log")
            self.log_final_path = os.path.join(self.eval_args.save_dir, "outputs.log")
            output_log_file = open(self.log_temp_path, "w", encoding="utf-8", newline="\n")

        pbar = tqdm(categories, desc="Processing categories", position=0)
        results = {}
        try:
            for categ in pbar:
                pbar.set_postfix_str(categ)
                inputs, outputs, accuracies, labels, msg_records = [], [], [], [], []
                for i in trange(len(dataset[categ]['train']), desc=f"Formatting batches ({categ})", position=1, leave=False):
                    instance = dataset[categ]['train'][i]
                    messages = [
                        {"role": Role.USER.value, "content": instance['system'] + instance['input']},
                        {"role": Role.ASSISTANT.value, "content": ""}
                    ]
                    input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                    inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                    labels.append(self._parse_answer(instance['output']))
                    msg_records.append(messages)

                for i in trange(
                    0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
                ):
                    batch_input = self.tokenizer.pad(
                        inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                    ).to(self.model.device)
                    pred_groups, text_groups = self.batch_inference(batch_input)

                    for j, (preds, texts) in enumerate(zip(pred_groups, text_groups)):
                        label = labels[i + j]
                        accuracies.append(float(np.mean([p == label for p in preds])))
                        outputs.append([preds, [label]])

                        if VERBOSE:
                            output_log_file.write(f"[{categ}] sample {i + j}\n")
                            output_log_file.write(f"messages: {json.dumps(msg_records[i + j])}\n")
                            output_log_file.write(f"texts: {json.dumps(texts)}\n")
                            output_log_file.write(f"parsed: {json.dumps(preds)}\n")
                            output_log_file.write(f"label: {label}\n\n")
                            output_log_file.flush()

                corrects = np.array(accuracies, dtype=float)
                category_corrects[categ] = np.concatenate([category_corrects[categ], corrects], axis=0)
                category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
                results[categ] = {str(i): outputs[i] for i in range(len(outputs))}
        finally:
            if VERBOSE:
                output_log_file.close()

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(
        self,
        category_corrects: dict[str, "NDArray"],
        results: dict[str, dict[int, str]],
    ) -> None:
        score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}\n{category_name:>15}_std: {100 * np.std(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
            f.write(score_info)

        if VERBOSE:
            os.replace(self.log_temp_path, self.log_final_path)

def run_eval() -> None:
    Evaluator().eval()
