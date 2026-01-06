import json

PART_1 = "result_verif_almost/autoformalization_equiv_checked_beq_normal.json"
PART_2 = "result_verif/autoformalization_equiv_checked_beq_normal.json"

def main():
    with open(PART_1, "r") as f:
        part1 = json.load(f)

    with open(PART_2, "r") as f:
        part2 = json.load(f)

    for key in part2:
        part1[key] = part2[key]

    with open("glued_result.json", "w") as f:
        json.dump(part1, f)

if __name__ == "__main__":
    main()
