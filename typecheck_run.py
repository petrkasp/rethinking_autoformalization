import json
import os

from lean_interact import AutoLeanServer, LeanREPLConfig, Command
from lean_interact.project import TempRequireProject, LeanRequire

from tqdm import tqdm

# Wait for mathlib. The function is called within beq_plus automatically, so there's not need to call it here.
from beq_plus import wait_for_mathlib


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        proofnet = [json.loads(line) for line in f]
    return {p["full_name"]: p for p in proofnet}


def get_typecheck(json):
    if "typecheck_result" not in json or "is_success" not in json["typecheck_result"]:
        return None
    return json["typecheck_result"]["is_success"]


def main(dataset_path, autoformalization_path, output_path):
    repl_config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.8.0", require="mathlib"), verbose=True)
    server = AutoLeanServer(config=repl_config)

    with open(autoformalization_path, 'r') as f:
        results = json.load(f)

    dataset = load_dataset(dataset_path)

    # Typecheck@N, at least one correct
    successes = 0

    for key in tqdm(results):
        this_run_success = False

        header = dataset[key]["header"]
        PREDICTION_KEY = "formal_stmt_pred"

        for prediction_json in results[key]:
            if PREDICTION_KEY not in prediction_json:
                # print(f"{key} is missing an autoformalization.")
                continue
            prediction = prediction_json[PREDICTION_KEY]
            if not prediction:
                continue

            prediction_json["typecheck_result"] = {}

            try:
                typecheck = server.run(
                    Command(
                        cmd=header + "\n" + prediction,
                    ),
                    timeout=120,
                )

                prediction_json["typecheck_result"]["is_success"] = typecheck.lean_code_is_valid()
                prediction_json["typecheck_result"]["result"] = str(typecheck)
            except TimeoutError:
                prediction_json["typecheck_result"]["is_success"] = False
                prediction_json["typecheck_result"]["result"] = "Timeout"

            if prediction_json["typecheck_result"]["is_success"]:
                this_run_success = True

        if this_run_success:
            successes += 1

    print ("Typecheck@N:", successes / len(results))
    print ("Typecheck@N:", f"{successes}/{len(results)}")

    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("autoformalization", type=str, help="Path to autoformalization")
    parser.add_argument("--dataset", type=str, required=True, help="Either 'proofnet', 'connf', or path to benchmark.jsonl")
    parser.add_argument("--output", type=str, help="Path to output file")
    args = parser.parse_args()

    args.dataset = {
        "proofnet": "data/proofnet/benchmark.jsonl",
        "connf": "data/connf/benchmark.jsonl",
    }.get(args.dataset, args.dataset)

    if not args.output:
        args.output = os.path.join(os.path.dirname(args.autoformalization), "typecheck.json")

    main(args.dataset, args.autoformalization, args.output)
