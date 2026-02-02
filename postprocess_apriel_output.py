import os.path as osp
import re
import json


def main(autoformalization_path, output_path):
    with open(autoformalization_path, 'r') as f:
        autoformalization = json.load(f)
    
    for _, result in autoformalization.items():
        result = result[0]
        if result['reasoning'] is None:
            result['formal_stmt_pred'], result["reasoning"] = result["reasoning"], result["formal_stmt_pred"]
            continue

        if result["formal_stmt_pred"] is None:
            continue

        result['full_output'] = result['formal_stmt_pred']

        matches = re.findall(r"```(?:lean\d?|)\n?(.*?)```", result['formal_stmt_pred'], re.DOTALL)
        if matches:
            code = matches[0].strip()
            code = "\n".join([line for line in code.splitlines() if not (line.startswith("open") or line.startswith("import"))])
            result['formal_stmt_pred'] = code

    if not output_path:
        output_path = osp.dirname(autoformalization_path) + '/autoformalization_postprocessed.json'

    with open(output_path, 'w') as f:
        json.dump(autoformalization, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('autoformalization', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    main(args.autoformalization, args.output)
