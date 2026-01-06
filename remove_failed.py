import json


TARGET = "result_verif/autoformalization.json"


def main():
    with open(TARGET, "r") as f:
        results = json.load(f)

    for key in list(results.keys()):
        my_id = int(key.split("_")[-1])

        if my_id <= 3532:
            del results[key]

    with open(TARGET.removesuffix(".json") + "2.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
