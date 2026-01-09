import os
import json
import argparse


def check_results(folder):
    files = os.listdir(folder)
    total_files = len([f for f in files if f.endswith('_result.json')])
    correct_files = 0
    for f in files:
        if f.endswith('_result.json'):
            with open(os.path.join(folder, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                if data['is_correct'] == True:
                    correct_files += 1
    print(f"Total files: {total_files}, Correct files: {correct_files}, Accuracy: {correct_files/total_files:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Check results")
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    folder = args.folder
    check_results(folder)


if __name__ == "__main__":
    main()