import json


# SPLIT = "o1"
SPLIT = "ra"

if SPLIT == "o1":
    RESULTS = "result_o1/autoformalization_equiv_checked_beq_normal.json"
    LABELS = "data/human_equivalence/o1-generated/labels.json"
elif SPLIT == "ra":
    RESULTS = "result_ra/autoformalization_equiv_checked_beq_normal.json"
    LABELS = "data/human_equivalence/rautoformalizer-generated/labels.json"
else:
    raise ValueError(f"Unknown split: {SPLIT}")


def main():
    with open(RESULTS, 'r') as f:
        results = json.load(f)
    with open(LABELS, 'r') as f:
        labels = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for key in results:
        data_point = results[key][0]
        prediction = data_point["formal_stmt_pred"]
        correct = labels[prediction]

        beq_result = data_point["equivcheck_results_PQ"]["is_success"] and data_point["equivcheck_results_QP"]["is_success"]
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
