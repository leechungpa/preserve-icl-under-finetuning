from typing import List

import torch


def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    return torch.device("cpu")


def parse_seed_list(s: str) -> List[int]:
    """
    Available formats:
      - Comma list: "0,1,2"
      - Range exclusive: "0:10" => 0..9
      - Range inclusive: "0-9"  => 0..9
      - Single integer: "3"
    """
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        start, end = int(a), int(b)
        return list(range(start, end))
    if "-" in s:
        a, b = s.split("-")
        start, end = int(a), int(b)
        return list(range(start, end + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x != ""]
    return [int(s)]