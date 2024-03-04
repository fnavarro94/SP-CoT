import json
import os
import argparse
import random
from tqdm import tqdm
from src.utils import DEFAULT_TRAIN_PATHS, DEFAULT_DEV_PATHS, DEFAULT_TEST_PATHS


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing")

    # Default parameters
    parser.add_argument("--task", type=str, choices=list(DEFAULT_TRAIN_PATHS.keys()))
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    if args.input_file is None:
        if args.split == "train":
            print(f"Default input file is used: {DEFAULT_TRAIN_PATHS[args.task]}")
            args.input_file = DEFAULT_TRAIN_PATHS[args.task]
        elif args.split == "dev":
            print(f"Default input file is used: {DEFAULT_DEV_PATHS[args.task]}")
            args.input_file = DEFAULT_DEV_PATHS[args.task]
        elif args.split == "test":
            print(f"Default input file is used: {DEFAULT_TEST_PATHS[args.task]}")
            args.input_file = DEFAULT_DEV_PATHS[args.task]
        else:
            raise ValueError(f"Split {args.split} is not supported.")

    if args.save_path is None:
        print(f"Default save path is used: data/{args.task}")
        args.save_path = os.path.join("data", args.task)

    os.makedirs(args.save_path, exist_ok=True)

    return args


def load_json_and_jsonl_file(input_file: str):
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            raw_dataset = json.load(f)
    elif input_file.endswith(".jsonl"):
        raw_dataset = []
        with open(input_file, "r") as f:
            for line in f:
                raw_dataset.append(json.loads(line))
    else:
        raise ValueError(f"File format {input_file} is not supported.")

    return raw_dataset


def main(args: argparse.Namespace):

    dataset = []

    if args.task == "hotpot-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            supporting_facts = item["supporting_facts"]
            context = item["context"]
            supporting_contexts = []
            for fact in supporting_facts:
                for paragraph in context:
                    if fact[0] in paragraph[0] and fact[1] < len(paragraph[1]):
                        supporting_contexts.append(f"{fact[0]} | {paragraph[1][fact[1]].strip()}")
            dataset.append({
                'question': item['question'],
                "answer": item["answer"],
                "evidence": list(set(supporting_contexts)),
                "type": item["type"],
                "level": item["level"]
            })
    elif args.task == "musique-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            paragraphs = item["paragraphs"]
            question_decomposition = item["question_decomposition"]
            QAEs = []
            for qd in question_decomposition:
                assert paragraphs[qd["paragraph_support_idx"]]["idx"] == qd["paragraph_support_idx"]
                assert paragraphs[qd["paragraph_support_idx"]]["is_supporting"]
                QAEs.append({
                    "question": qd["question"],
                    "answer": qd["answer"],
                    "evidence": paragraphs[qd["paragraph_support_idx"]]["paragraph_text"]
                })
            dataset.append({
                "question": item["question"],
                "answer": item["answer"],
                "decomposition": QAEs,
            })
    elif args.task == "cweb-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            aliases = []
            for alias in item["answers"]:
                aliases.extend(alias["aliases"])
            dataset.append({
                "question": item["question"],
                "answer": [a["answer"] for a in item["answers"]] + aliases,
            })
    elif args.task == "wikimh-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            dataset.append({
                "question": item["question"],
                "answer": item["answer"],
                "evidence": ["->".join(e) for e in item["evidences"]],
            })
    elif args.task == "hybrid-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            dataset.append({
                "question": item["question"],
                "answer": item["answer-text"],
            })
    elif args.task == "grail-qa":
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            answers = []
            for a in item["answer"]:
                if "entity_name" in a:
                    answers.append(a["entity_name"])
                elif "answer_argument" in a:
                    answers.append(a["answer_argument"])
                else:
                    raise ValueError(f"Answer format is not supported: {a}")
            dataset.append({
                "question": item["question"],
                "answer": answers,
            })
    elif args.task == "complex-qa":
        with open(args.input_file, "r") as f:
            raw_dataset = f.read().split("\n")
        for line in tqdm(raw_dataset):
            line_tokens = line.split("\t")
            if len(line_tokens) != 2:
                continue
            question = line_tokens[0]
            answers = eval(line_tokens[1])
            dataset.append({
                "question": question,
                "answer": answers,
            })

    elif args.task in ["nq", "webqa", "triviaqa"]:
        raw_dataset = load_json_and_jsonl_file(args.input_file)
        for item in tqdm(raw_dataset):
            dataset.append({
                "question": item["question"],
                "answer": item["answer"],
            })

    else:
        raise ValueError(f"Task {args.task} is not supported.")

    if args.split == "train":
        with open(os.path.join(args.save_path, "processed_train.json"), "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Processed dataset is saved to {os.path.join(args.save_path, 'processed_train.json')}.")

    elif args.split == "dev":
        with open(os.path.join(args.save_path, "processed_dev.json"), "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Processed dataset is saved to {os.path.join(args.save_path, 'processed_dev.json')}.")

    elif args.split == "test":
        random.seed(42)
        selected_dataset = random.sample(dataset, k=min(1000, len(dataset)))
        print(f"Randomly select {len(selected_dataset)} samples from the test set.")
        with open(os.path.join(args.save_path, "processed_test.json"), "w") as f:
            json.dump(selected_dataset, f, indent=4)
        print(f"Processed dataset is saved to {os.path.join(args.save_path, 'processed_test.json')}.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
