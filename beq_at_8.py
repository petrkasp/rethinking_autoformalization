import json


def beq_success(prediction):
    return "equivcheck_results_PQ" in prediction and "equivcheck_results_QP" in prediction and \
       prediction["equivcheck_results_PQ"]["is_success"] and prediction["equivcheck_results_QP"]["is_success"]


def beq_plus_success(prediction):
    return "beq_plus" in prediction and prediction["beq_plus"]


def has_key(key, dct):
    for exercise in dct.values():
        for prediction in exercise:
            if key in prediction:
                return True
    return False


def main(path):
    with open(path, 'r') as f:
        results = json.load(f)

    # BEq@8, at least one correct
    beq_total = 0
    beq_plus_total = 0

    for key in results:
        beq_correct = False
        beq_plus_correct = False

        for prediction in results[key]:
            if beq_success(prediction):
                beq_correct = True
            if beq_plus_success(prediction):
                beq_plus_correct = True

        if beq_correct:
            beq_total += 1
        if beq_plus_correct:
            beq_plus_total += 1

    if has_key("equivcheck_results_PQ", results):
        print ("BEq@8:", beq_total / len(results))
        print ("BEq@8:", f"{beq_total}/{len(results)}")

    if has_key("beq_plus", results):
        print ("BEq+@8:", beq_plus_total / len(results))
        print ("BEq+@8:", f"{beq_plus_total}/{len(results)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to results file")
    args = parser.parse_args()

    main(args.path)
