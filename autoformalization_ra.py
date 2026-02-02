import os
import os.path as osp
import json
import torch
from tqdm import tqdm

RETRIEVE_NUM = 5


def load_model(model: str):
    if model == "apriel_api":
        from apriel_api import Apriel
        return Apriel()
    elif model == "apriel_local":
        from apriel_model import Apriel
        return Apriel()
    else:
        raise ValueError(f"Unknown model: {model}")


def format_prompt(informal_stmt, retrieved_premises):
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
    
    return f"<dependencies>{''.join(dependencies)}</dependencies><informal>{informal_stmt}</informal>\n\n" + \
        "<task>Translate the <informal> statement into Lean4. Do NOT attempt to prove it. Do end the Lean4 definition with := sorry. " + \
        "<dependencies> have been automatically retrieved – some or all might not be relevant.</task>"


def main(args):
    with open(args.benchmark_file, 'r') as f:
        samples = [json.loads(l) for l in f.readlines()]
    
    with open(args.library_file, 'r') as f:
        premises = [json.loads(l) for l in f.readlines()]
    
    premises_dict = {p['full_name'] : p for p in premises}

    _, top_kmax_indices_dict= torch.load(args.retrieval_result_path, weights_only=True)
    
    model = load_model(args.model)

    queries = {}

    for sample in samples:
        retrieved_premises = [premises[i]['full_name'] for i in top_kmax_indices_dict[sample['full_name']][:RETRIEVE_NUM].tolist()]
        full_premises = [premises_dict[name] for name in retrieved_premises]
        combined_premises = sample["hard_dependencies"] + full_premises[:RETRIEVE_NUM - len(sample["hard_dependencies"])]
        prompt = format_prompt(sample["informal_stmt"], combined_premises)
        
        queries[sample['full_name']] = prompt

    results = {}

    for name, prompt in tqdm(queries.items()):
        print(f"{name}: {prompt}")
        response, reasoning = model.generate(prompt)
        print(reasoning)
        results[name] = [{
                'formal_stmt_pred': response,
                'reasoning': reasoning
            }]

    output_dir = osp.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
    args = parser.parse_args()

    if ((args.benchmark_file is not None) != (args.library_file is not None)) or \
        ((args.dataset is None) == (args.benchmark_file is None)):
        parser.error('Specify either --dataset or both --benchmark_file and --library_file.')

    if args.dataset is not None:
        DATASET_ROOT = 'data'
        args.benchmark_file = osp.join(DATASET_ROOT, args.dataset, 'benchmark.jsonl')
        args.library_file = osp.join(DATASET_ROOT, args.dataset, 'library.jsonl')
        
    main(args)
