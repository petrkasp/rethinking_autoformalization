import json
import os

from beq_plus import beq_plus

from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.project import TempRequireProject, LeanRequire

from tqdm import tqdm


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        proofnet = [json.loads(line) for line in f]
    return {p["full_name"]: p for p in proofnet}


def main(dataset_path, autoformalization_path, output_path):
    repl_config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.8.0", require="mathlib"), verbose=True)
    server = AutoLeanServer(config=repl_config)

    with open(autoformalization_path, 'r') as f:
        results = json.load(f)

    dataset = load_dataset(dataset_path)

    # BEq@8, at least one correct
    successes = 0

    for key in tqdm(results):
        this_run_success = False

        header = dataset[key]["header"]
        reference = dataset[key]["formal_stmt"] 
        PREDICTION_KEY = "formal_stmt_pred"
        GET_TYPECHECK = lambda json: json["typecheck_result"]["is_success"]

        for prediction_json in results[key]:
            if PREDICTION_KEY not in prediction_json:
                # print(f"{key} is missing an autoformalization.")
                continue
            prediction = prediction_json[PREDICTION_KEY]
            typecheck = GET_TYPECHECK(prediction_json)

            if not typecheck:
                continue

            beq_result = beq_plus(
                reference,
                prediction,
                header,
                server=server,
                timeout_per_proof=120
            )

            prediction_json["beq_plus"] = beq_result

            if beq_result:
                this_run_success = True

        if this_run_success:
            successes += 1

    print ("BEq+@8:", successes / len(results))
    print ("BEq+@8:", f"{successes}/{len(results)}")

    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    PATH = lambda split: f"results/v470/result_af_{split}/autoformalization.json"
    parser.add_argument("--dataset", type=str, required=True, help="Either 'proofnet', 'connf', or path to benchmark.jsonl")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--autoformalization", type=str, help="Path to autoformalization.json")
    group.add_argument("--split", type=str, help="Either 'nora', 'ra', or 'gtra'")
    parser.add_argument("--output", type=str, help="Path to output file")
    args = parser.parse_args()

    args.dataset = {
        "proofnet": "data/proofnet/benchmark.jsonl",
        "connf": "data/connf/benchmark.jsonl",
    }.get(args.dataset, args.dataset)

    if args.split:
        args.autoformalization = PATH(args.split)

    if not args.output:
        args.output = os.path.join(os.path.dirname(args.autoformalization), "beq_plus.json")

    main(args.dataset, args.autoformalization, args.output)
