import json
import os
from src.utils import load_json_and_jsonl_file, print_now, DEFAULT_PATHS, extract_answer, qa_evaluate
import argparse
from tqdm import tqdm
from copy import deepcopy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-Prompt-CoT")

    # Default parameters
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--task", type=str, default="complex-qa",
                        choices=list(DEFAULT_PATHS.keys()), help="dataset name")

    parser.add_argument("--method", type=str, default="zeroshot-cot",
                        choices=["baseline", "zeroshot", "zeroshot-cot", "genread", "self-prompt", "auto-cot", "self-prompt-cot"],
                        help="method name")

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)

    parser.add_argument(
        # "--model_name", type=str, default="gpt-3.5-turbo-0301", help="model used for response generation.")
        "--model_name", type=str, default="rag", help="model used for response generation.")

    parser.add_argument("--output_file", type=str, default=None, help="output file")
    parser.add_argument("--result_file", type=str, default=None, help="output file")
    parser.add_argument("--on_test_data", action="store_true", help="whether to evaluate on test data")


    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = DEFAULT_PATHS[args.task]
    if args.save_path is None:
        args.save_path = os.path.join(DEFAULT_PATHS[args.task], args.model_name)
    if args.input_file is None:
        args.input_file = os.path.join(args.save_path, f"{args.method}.json")
        if not os.path.exists(args.input_file):
            raise ValueError(f"Default file {args.input_file} does not exist.")
    if args.output_file is None:
        args.output_file = args.input_file.replace(".json", "_evaluate.json")
    # if args.result_file is None:
    #     args.result_file = args.input_file.replace(".json", "_evaluate_result.json")

    print(f"[{print_now(1)}] Task: {args.task}")
    print(f"[{print_now(1)}] Method: {args.method}")
    print(f"[{print_now(1)}] Loading dataset from {args.input_file} ...")

    os.makedirs(args.save_path, exist_ok=True)

    return args


def main(args: argparse.Namespace):
    dataset = load_json_and_jsonl_file(args.input_file)

    results = {
        "correct": [],
        "wrong": []
    }
    all_em = 0
    all_f1 = 0

    output_dataset = []

    if args.on_test_data:
        raw_dataset = load_json_and_jsonl_file(os.path.join(args.dataset_path, "processed_test.json"))
    else:
        raw_dataset = load_json_and_jsonl_file(os.path.join(args.dataset_path, "processed_dev.json"))

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        for raw_item in raw_dataset:
            if raw_item["question"] == question:
                item["answer"] = raw_item["answer"]
                break
        if not isinstance(item["answer"], list):
            item["answer"] = [item["answer"]]
        gold_answers = item["answer"]

        response = item["response"]
        if isinstance(response, list):
            response = " ".join(response).lower()
        else:
            response = response.lower()

        if args.method == "self-prompt-cot":
            if "final answer is: " in response.lower():
                raw_pred = response[response.find("final answer is: ") + len("final answer is: "):]
            elif "final answer is " in response.lower():
                raw_pred = response[response.find("final answer is ") + len("final answer is "):]
            elif "answer is: " in response.lower():
                raw_pred = response[response.find("answer is: ") + len("answer is: "):]
            elif "final answer in just one entity is: " in response.lower():
                raw_pred = response[response.find("final answer in just one entity is: ") + len("final answer in just one entity is: "):]
            elif "answer the question in just one entity: " in response.lower():
                raw_pred = response[response.find("answer the question in just one entity: ") + len("answer the question in just one entity: "):]
            else:
                raw_sentences = response.split("\n")
                answer_sentences = []
                for raw_sentence in raw_sentences:
                    if "answer: " in raw_sentence.lower():
                        answer_sentences.append(raw_sentence.lower())
                if len(answer_sentences) == 0:
                    raw_pred = ""
                else:
                    final_answer_sentence = answer_sentences[-1]
                    raw_pred = final_answer_sentence[final_answer_sentence.find("answer: ") + len("answer: "):]
        elif args.method == "self-prompt":
            if ", because" in response.lower():
                raw_pred = response[:response.find(", because")]
            else:
                raw_sentences = response.split("\n")
                answer_sentences = []
                for raw_sentence in raw_sentences:
                    if "answer: " in raw_sentence.lower():
                        answer_sentences.append(raw_sentence.lower())
                if len(answer_sentences) == 0:
                    raw_pred = response
                else:
                    final_answer_sentence = answer_sentences[-1]
                    raw_pred = final_answer_sentence[final_answer_sentence.find("answer: ") + len("answer: "):]
        else:
            if "answer: " in response.lower():
                raw_pred = response[response.find("answer: ") + len("answer: "):]
            else:
                raw_pred = response
        pred_answers = extract_answer(raw_pred)

        em_score, f1_score = qa_evaluate(gold_answers, pred_answers)
        all_em += em_score
        all_f1 += f1_score

        item["pred_ans"] = pred_answers
        item["em"] = em_score
        item["f1"] = f1_score
        new_item = deepcopy(item)

        output_dataset.append(new_item)

        if em_score == 1.0:
            results["correct"].append(new_item)
        else:
            results["wrong"].append(new_item)

    with open(args.output_file, "w") as f:
        json.dump(output_dataset, f, indent=4)

    # with open(args.result_file, "w") as f:
    #     json.dump(results, f, indent=4)

    print("EM: {:.4f}, F1: {:.4f}".format(all_em / len(dataset), all_f1 / len(dataset)))
    print(f"Correct: {len(results['correct'])}, Wrong: {len(results['wrong'])}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
