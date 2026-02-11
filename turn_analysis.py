import json
import os
from collections import Counter

from lean_interact import AutoLeanServer, LeanREPLConfig, Command
from lean_interact.project import TempRequireProject, LeanRequire

from tqdm import tqdm

# Wait for mathlib. The function is called within beq_plus automatically, so there's not need to call it here.
from beq_plus import wait_for_mathlib


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        proofnet = [json.loads(line) for line in f]
    return {p["full_name"]: p for p in proofnet}


def typecheck(server, command):
    try:
        typecheck = server.run(
            Command(
                cmd=command,
            ),
            timeout=120,
        )

        return typecheck.lean_code_is_valid()
    except TimeoutError:
        return False


def main(autoformalization_path, dataset_path):
    with open(autoformalization_path, 'r') as f:
        results = json.load(f)

    repl_config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.8.0", require="mathlib"), verbose=True)
    server = AutoLeanServer(config=repl_config)

    dataset = load_dataset(dataset_path)

    counter = Counter()

    for key in tqdm(results):
        header = dataset[key]["header"]
        for prediction_json in results[key]:
            for i, turn in enumerate(prediction_json["turns"]):
                if typecheck(server, header + "\n" + turn["lean"]):
                    counter[i] += 1
                    break

    print (counter)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("autoformalization", type=str, help="Path to autoformalization")
    parser.add_argument("--dataset", type=str, required=True, help="Either 'proofnet', 'connf', or path to benchmark.jsonl")
    args = parser.parse_args()

    args.dataset = {
        "proofnet": "data/proofnet/benchmark.jsonl",
        "connf": "data/connf/benchmark.jsonl",
    }.get(args.dataset, args.dataset)

    main(args.autoformalization, args.dataset)
