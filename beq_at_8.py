import json


# SPLIT = "nora"
SPLIT = "ra"
# SPLIT = "gtra"

RESULTS = {
 "nora": "result_af_nora/autoformalization_equiv_checked_beq_normal.json",
 "ra":   "result_af_ra/autoformalization_equiv_checked_beq_normal.json",
 "gtra": "result_af_gtra/autoformalization_equiv_checked_beq_normal.json",
}[SPLIT]


def main():
    with open(RESULTS, 'r') as f:
        results = json.load(f)

    # BEq@8, at least one correct
    successes = 0

    for key in results:
        this_run_success = False

        for prediction in results[key]:
            if "equivcheck_results_PQ" in prediction and "equivcheck_results_QP" in prediction and \
             prediction["equivcheck_results_PQ"]["is_success"] and prediction["equivcheck_results_QP"]["is_success"]:
                this_run_success = True

        if this_run_success:
            successes += 1

    print ("BEq@8:", successes / len(results))
    print ("BEq@8:", f"{successes}/{len(results)}")


if __name__ == '__main__':
    main()
