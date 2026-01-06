import json
import re


# SPLIT = "o1"
# SPLIT = "ra"
# SPLIT = "proofnetverif_small"
SPLIT = "proofnetverif"
# SPLIT = "proofnetverif_partial"

LABELS_PREDICTION = True
RESULTS_FROM_LOG = False

if SPLIT == "o1":
    RESULTS = "result_o1_v480_2/autoformalization_equiv_checked_beq_normal.json"
    LABELS = "data/human_equivalence/o1-generated/labels.json"
elif SPLIT == "ra":
    RESULTS = "result_ra_v480_2/autoformalization_equiv_checked_beq_normal.json"
    LABELS = "data/human_equivalence/rautoformalizer-generated/labels.json"
elif SPLIT == "proofnetverif_small":
    RESULTS = "result_verifsmall/autoformalization_equiv_checked_beq_normal.json"
    LABELS = "data/proofnetverif_small/labels.json"
    LABELS_PREDICTION = False
elif SPLIT == "proofnetverif":
    RESULTS = "glued_result.json"
    LABELS = "data/proofnetverif/labels.json"
    LABELS_PREDICTION = False
elif SPLIT == "proofnetverif_partial":
    RESULTS = "result_verif/equivalence_server.log"
    LABELS = "data/proofnetverif/labels.json"
    LABELS_PREDICTION = False
    RESULTS_FROM_LOG = True
else:
    raise ValueError(f"Unknown split: {SPLIT}")

assert not (LABELS_PREDICTION and RESULTS_FROM_LOG)

def main():
    if RESULTS_FROM_LOG:
        with open(RESULTS, 'r') as f:
            results = f.readlines()
    else:
        with open(RESULTS, 'r') as f:
            results = json.load(f)
    with open(LABELS, 'r') as f:
        labels = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for key in results:
        if not RESULTS_FROM_LOG:
            data_point = results[key][0]
            beq_result = data_point["equivcheck_results_PQ"]["is_success"] and data_point["equivcheck_results_QP"]["is_success"]
        else:
            line = key
            if "check:445" not in line:
                continue
            beq_result = line.endswith("1 1 1 1\n")
            key = re.search(r"Check\(ProofNetVerif\.(.*?)\)", line).group(1)

        if LABELS_PREDICTION:
            prediction = data_point["formal_stmt_pred"]
            correct = labels[prediction]
        else:
            correct = labels[key]

        if beq_result and correct:
            true_positives += 1
        elif beq_result and not correct:
            false_positives += 1
        elif not beq_result and correct:
            false_negatives += 1
        else:  # not beq_result and not correct
            true_negatives += 1

    print("True positives:", true_positives)
    print("False positives:", false_positives)
    print("False negatives:", false_negatives)
    print("True negatives:", true_negatives)

    print("Accuracy:", (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives))
    print("Precision:", true_positives / (true_positives + false_positives))
    print("Recall:", true_positives / (true_positives + false_negatives))
    print("F1:", 2 * true_positives / (2 * true_positives + false_positives + false_negatives))

if __name__ == '__main__':
    main()
