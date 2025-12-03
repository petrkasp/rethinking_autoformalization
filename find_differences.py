import json

SPLIT = "o1"

if SPLIT == "o1":
    T0_FILE = "result_o1_t0.0/autoformalization_equiv_checked_beq_normal.json"
    V480_FILE = "result_o1_v480/autoformalization_equiv_checked_beq_normal.json"
elif SPLIT == "ra":
    T0_FILE = "result_ra_t0.0/autoformalization_equiv_checked_beq_normal.json"
    V480_FILE = "result_ra_v480/autoformalization_equiv_checked_beq_normal.json"

def main():
    with open(T0_FILE, 'r', encoding="utf-8") as f:
        t0_file = json.load(f)
    with open(V480_FILE, 'r', encoding="utf-8") as f:
        v480_file = json.load(f)

    differences = {}

    for key in t0_file:
        t0_item = t0_file[key][0]
        v480_item = v480_file[key][0]

        PQ_different = t0_item["equivcheck_results_PQ"]["is_success"] != v480_item["equivcheck_results_PQ"]["is_success"]
        QP_different = t0_item["equivcheck_results_QP"]["is_success"] != v480_item["equivcheck_results_QP"]["is_success"]

        if PQ_different or QP_different:
            differences[key] = {
                "t0": t0_item,
                "v480": v480_item
            }

    with open(f"differences_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump(differences, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
