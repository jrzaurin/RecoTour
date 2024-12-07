import json
from pathlib import Path


def find_results_json(directory):
    # Search for results.json in the directory and its subdirectories
    for path in directory.rglob("results.json"):
        return path
    return None


# Get the base results directory
results_dir = Path("results")
binary_results = {}

# Find all directories containing 'binary'
binary_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "binary" in d.name]

# Search for results.json in each directory
for dir_path in binary_dirs:
    results_path = find_results_json(dir_path)

    if results_path:
        try:
            with open(results_path) as f:
                binary_results[dir_path.name] = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading JSON from {dir_path.name}")
    else:
        print(f"No results.json found in {dir_path.name}")

# Print the collected results
for dir_name, results in binary_results.items():
    print(f"\n{dir_name}:")
    print(json.dumps(results, indent=2))
