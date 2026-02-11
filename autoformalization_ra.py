import os
import os.path as osp
import json
import re

import torch
from tqdm import tqdm

RETRIEVE_NUM = 5
NUMBER_OF_ATTEMPTS = 3 # Including the first attempt


def load_model(model: str):
    if model == "apriel_api":
        from apriel_api import Apriel
        return Apriel()
    elif model == "apriel_local":
        from apriel_model import Apriel
        return Apriel()
    else:
        raise ValueError(f"Unknown model: {model}")


def load_lean_server():
    from lean_interact import AutoLeanServer, LeanREPLConfig
    from lean_interact.project import TempRequireProject
    from beq_plus import wait_for_mathlib # Needs not to be called as it's called in beq_plus
    
    repl_config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.8.0", require="mathlib"), verbose=True)
    server = AutoLeanServer(config=repl_config)
    return server


def format_prompt(informal_stmt, retrieved_premises, header):
    dependencies = []
    for dependency in retrieved_premises:
        declaration = dependency['header'].replace('🔗<|PREMISE|>🔗', '').strip()[:768]
        code = None
        if (not any([s in dependency.get('ptype', '') for s in {'theorem', 'lemma', 'axiom'}])) or 'header' not in dependency.keys():
            code = (dependency['code'] if 'code' in dependency.keys() else dependency['formal_stmt'])[:768].strip()

        dependency_code = f"<dependency><declaration>{declaration}</declaration>"
        if code:
            dependency_code += f"<code>{code}</code>"
        dependency_code += "</dependency>"
        dependencies.append(dependency_code)
    
    return f"<dependencies>{''.join(dependencies)}</dependencies><header>{header}</header><informal>{informal_stmt}</informal>\n\n" + \
        "<task>Translate the `informal` statement into Lean4. Do NOT attempt to prove it. Do end the Lean4 definition with := sorry. " + \
        "`dependencies` have been automatically retrieved – some or all might not be relevant. " + \
        "The `header` will be automatically added to your code. You should not need to add any other imports.</task>"


def extract_lean_from_output(response: str):
    if response is None:
        return None

    matches = re.findall(r"```(?:lean\d?|)\n?(.*?)```", response, re.DOTALL)
    if not matches:
        return response

    code = matches[0].strip()
    code = "\n".join([line for line in code.splitlines() if not (line.startswith("open") or line.startswith("import"))])
    return code


def main(args):
    output_dir = osp.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.benchmark_file, 'r') as f:
        samples = [json.loads(l) for l in f.readlines()]
    
    with open(args.library_file, 'r') as f:
        premises = [json.loads(l) for l in f.readlines()]
    
    premises_dict = {p['full_name'] : p for p in premises}

    _, top_kmax_indices_dict= torch.load(args.retrieval_result_path, weights_only=True)
    
    model = load_model(args.model)
    if NUMBER_OF_ATTEMPTS > 1:
        from lean_interact import Command
        lean_server = load_lean_server()

    queries = {}

    for sample in samples:
        retrieved_premises = [premises[i]['full_name'] for i in top_kmax_indices_dict[sample['full_name']][:RETRIEVE_NUM].tolist()]
        full_premises = [premises_dict[name] for name in retrieved_premises]
        combined_premises = sample["hard_dependencies"] + full_premises[:RETRIEVE_NUM - len(sample["hard_dependencies"])]
        prompt = format_prompt(sample["informal_stmt"], combined_premises, sample["header"])
        
        queries[sample['full_name']] = (prompt, sample["header"])

    results = {}

    for name, (prompt, header) in tqdm(queries.items()):
        print(name)

        current_prompt = [prompt]
        turns = []
        last_lean = None
        for attempt_number in range(NUMBER_OF_ATTEMPTS):
            print(current_prompt)
            response, reasoning = model.generate(current_prompt)
            print(response)
            print(reasoning)
            last_lean = extract_lean_from_output(response)
            turns.append({
                "lean": last_lean,
                'response': response,
                'reasoning': reasoning
            })

            if last_lean is None:
                break

            if attempt_number + 1 < NUMBER_OF_ATTEMPTS:
                current_prompt.append(response)
                typecheck = lean_server.run(
                    Command(
                        cmd=header + "\n" + last_lean,
                    ),
                    timeout=120,
                )
                if typecheck.lean_code_is_valid():
                    break
                current_prompt.append("Your code\n```lean4\n" + last_lean + "\n```\n produced some errors.\n" + str(typecheck) + "\nTry to fix them.")

        results[name] = [{
                'formal_stmt_pred': last_lean,
                "turns": turns
            }]

        if args.save_partial:
            partial_results = {name: [{"formal_stmt_pred": results[name][0]["formal_stmt_pred"]}] for name in results}
            with open(args.output.replace(".json", ".partial.json"), 'w') as f:
                json.dump(partial_results, f)

    with open(args.output, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="apriel_api", help="apriel_api or apriel_local")
    parser.add_argument('--retrieval_result_path', type=str, default="result_dense/retrieval_result_proofnet.pt")

    parser.add_argument('--dataset', type=str)
    # OR BOTH OF BELOW
    parser.add_argument('--benchmark_file', type=str)
    parser.add_argument('--library_file', type=str)

    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--save_partial', default=False, action='store_true')
    args = parser.parse_args()

    if ((args.benchmark_file is not None) != (args.library_file is not None)) or \
        ((args.dataset is None) == (args.benchmark_file is None)):
        parser.error('Specify either --dataset or both --benchmark_file and --library_file.')

    if args.dataset is not None:
        DATASET_ROOT = 'data'
        args.benchmark_file = osp.join(DATASET_ROOT, args.dataset, 'benchmark.jsonl')
        args.library_file = osp.join(DATASET_ROOT, args.dataset, 'library.jsonl')

    if not args.output.endswith('.json'):
        args.output = osp.join(args.output, 'autoformalization.json')
        
    main(args)
