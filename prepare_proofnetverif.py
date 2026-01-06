import json
import os
import re

import datasets


def main():
    dataset = datasets.load_dataset("PAug/ProofNetVerif", split="valid+test")#.select(range(25))
    
    autoformalization = {}
    labels = {}
    jsonl = []
    i = 0

    for data_point in dataset:
        my_id = data_point["id"] + "_" + str(i)
        i += 1

        # BEq wants "theorem thm_Q" in the prediction
        new_prediction = data_point["lean4_prediction"].replace("theorem dummy", "theorem thm_Q")
        assert "theorem thm_Q" in new_prediction
        # BEq wants the theorems to end with ":= by sorry"
        new_prediction = re.sub(r":= sorry$", ":= by sorry", new_prediction)
        autoformalization[my_id] = [
            {
                "formal_stmt_pred": new_prediction,
                "typecheck_result": {
                    "is_success": True
                }
            }
        ]

        labels[my_id] = data_point["correct"]

        # BEq wants "theorem thm_P" in the ground truth formalization
        new_formalization = re.sub(r"theorem [\w_]+", "theorem thm_P", data_point["lean4_formalization"])
        assert "theorem thm_P" in new_formalization
        # BEq wants the theorems to end with ":= by sorry"
        new_formalization = re.sub(r":=$", ":= by sorry", new_formalization)
        jsonl.append({
            "informal_stmt": data_point["nl_statement"],
            "formal_stmt": new_formalization,
            "header": data_point["lean4_src_header"],
            "source": "ProofNetVerif", # Required
            "problem_name": my_id,
            "full_name": my_id
        })

    PATH = "data/proofnetverif"
    os.makedirs(PATH, exist_ok=True)
    with open(os.path.join(PATH, "autoformalization.json"), "w") as f:
        json.dump(autoformalization, f)
    with open(os.path.join(PATH, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(PATH, "benchmark.jsonl"), "w") as f:
        for line in jsonl:
            f.write(json.dumps(line) + "\n")

    # Dummy file. Has to exist because BEq touches it but does not use it.
    with open(os.path.join(PATH, "library.jsonl"), "w") as f:
        pass

if __name__ == "__main__":
    main()
