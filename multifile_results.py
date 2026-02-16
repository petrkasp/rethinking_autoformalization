import json
import argparse
from beq_at_n import calculate_results


def load_and_aggregate(paths):
    """
    Load multiple result JSON files and aggregate them by concatenating
    the prediction arrays for each exercise key.

    Each file has the format:
        { "exercise_name": [ {prediction_1}, ... ], ... }

    The aggregated result concatenates the arrays across files:
        { "exercise_name": [ pred_from_file1, pred_from_file2, ... ], ... }
    """
    aggregated = {}

    for path in paths:
        with open(path, 'r') as f:
            results = json.load(f)

        for key, predictions in results.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].extend(predictions)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from multiple evaluation runs and calculate metrics."
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to result JSON files to aggregate",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="@N. Number of samples to consider (default: total aggregated count)",
    )
    args = parser.parse_args()

    aggregated = load_and_aggregate(args.paths)
    print(f"Aggregated {len(args.paths)} files, "
          f"{len(aggregated)} exercises, "
          f"{len(list(aggregated.values())[0])} predictions per exercise")
    calculate_results(aggregated, args.n)


if __name__ == "__main__":
    main()
