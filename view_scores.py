from pathlib import Path
import json

if __name__ == "__main__":
    scores_dir = Path("/path/to/scores_*/")
    files = list(scores_dir.rglob("*.json"))

    for file in sorted(files):
        with open(file, "r") as f:
            data = json.load(f)

        print(f"{str(file)[30:]}: {data['score']:.4f}")
