import json
import os
from src.utils import load_json_and_jsonl_file, print_now, DEFAULT_PATHS, extract_answer, qa_evaluate, normalize_answer
import argparse
from tqdm import tqdm
from copy import deepcopy
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-Prompt-CoT")

    # Default parameters
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--task", type=str, default="complex-qa",
                        choices=list(DEFAULT_PATHS.keys()), help="dataset name")

    parser.add_argument("--method", type=str, default="zeroshot-cot",
                        # choices=["baseline", "zeroshot", "zeroshot-cot", "genread", "self-prompt", "auto-cot", "self-prompt-cot", "manual-cot"],
                        help="method name")

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--on_test_data", action="store_true", help="whether to run on test data")
    parser.add_argument("--overwrite_input_file", action="store_true", help="whether to overwrite input file")
    parser.add_argument("--extract_answer", action="store_true", help="whether to extract answer from the response")

    parser.add_argument(
        # "--model_name", type=str, default="gpt-3.5-turbo-0301", help="model used for response generation.")
        "--model_name", type=str, default="rag", help="model used for response generation.")

    parser.add_argument("--output_file", type=str, default=None, help="output file")
    parser.add_argument("--result_file", type=str, default=None, help="output file")

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
        if args.overwrite_input_file:
            args.output_file = args.input_file
        else:
            args.output_file = args.input_file.replace(".json", "_evaluate.json")

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
        match = False
        for raw_item in raw_dataset:
            if raw_item["question"] == question:
                if args.task == "wikimh-qa":
                    item["sub_answers"] = [evidence.split("->")[-1] for evidence in raw_item["evidence"]]
                elif args.task == "musique-qa":
                    item["sub_answers"] = [qa["answer"] for qa in raw_item["decomposition"]]
                item["answer"] = raw_item["answer"]
                match = True
                break
        if not match:
            raise ValueError(f"Question {question} does not exist in the raw dataset.")

        if isinstance(item["answer"], list):
            temp_answer = ", ".join(item["answer"])
            item["answer"] = extract_answer(temp_answer)
        else:
            item["answer"] = extract_answer(item["answer"])
        gold_answers = item["answer"]

        if args.extract_answer:
            response = item["response"]
            if isinstance(response, list):
                response = "\n".join(response)
            else:
                response = response

            if args.method in ["self-prompt-cot", "manual-cot"]:
                if "final answer is: " in response.lower():
                    raw_pred = response[response.lower().find("final answer is: ") + len("final answer is: "):]
                elif "final answer is " in response.lower():
                    raw_pred = response[response.lower().find("final answer is ") + len("final answer is "):]
                elif "answer is: " in response.lower():
                    raw_pred = response[response.lower().find("answer is: ") + len("answer is: "):]
                elif "final answer in just one entity is: " in response.lower():
                    raw_pred = response[response.lower().find("final answer in just one entity is: ") + len("final answer in just one entity is: "):]
                elif "answer the question in just one entity: " in response.lower():
                    raw_pred = response[response.lower().find("answer the question in just one entity: ") + len("answer the question in just one entity: "):]
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
                    raw_pred = response[:response.lower().find(", because")]
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
            elif args.method in ["zeroshot-cot", "auto-cot"]:
                if "is: " in response.lower():
                    raw_pred = response[response.lower().find("is: ") + len("is: "):]
                elif "answer: " in response.lower():
                    raw_pred = response[response.lower().find("answer: ") + len("answer: "):]
                else:
                    raw_pred = response
            else:
                if "answer: " in response.lower():
                    raw_pred = response[response.lower().find("answer: ") + len("answer: "):]
                else:
                    raw_pred = response
            pred_answers = extract_answer(raw_pred)
        else:
            response = item["response"]
            pred_answers = item["pred_ans"] if "pred_ans" in item else item["pred_answer"]

        em_score, f1_score = qa_evaluate(gold_answers, pred_answers)
        all_em += em_score
        all_f1 += f1_score

        if "pred_ans" in item:
            item["pred_ans"] = pred_answers
        else:
            item["pred_answer"] = pred_answers
        item["em"] = em_score
        item["f1"] = f1_score

        if "sub_answers" in item:
            if isinstance(response, list):
                response = "\n".join(response)
            if args.task == "zeroshot-cot":
                response = item["cot"]
            correct_cnt = 0
            sub_answers = item["sub_answers"]
            for sub_answer in sub_answers[:-1]:
                if normalize_answer(sub_answer) in normalize_answer(response):
                # if sub_answer in pred_cot:
                    correct_cnt += 1
            accuracy = correct_cnt / len(sub_answers)
            item["sub_answer_acc"] = accuracy

        new_item = deepcopy(item)

        output_dataset.append(new_item)

        if em_score == 1.0:
            results["correct"].append(new_item)
        else:
            results["wrong"].append(new_item)

    # with open(args.output_file, "w") as f:
    #     json.dump(output_dataset, f, indent=4, ensure_ascii=False)

    if "sub_answers" in output_dataset[0]:
        correct_samples = [item for item in output_dataset if item["em"] == 1.0]
        total_sub_answers = sum([len(item["sub_answers"]) for item in correct_samples])
        total_correct_sub_answers = sum([len(item["sub_answers"]) * item["sub_answer_acc"] for item in correct_samples])
        total_sub_answer_acc = total_correct_sub_answers / total_sub_answers
        sub_answer_acc_avg = np.mean([item["sub_answer_acc"] for item in correct_samples])
        print(f"Total sub answers: {total_sub_answers}, Total correct sub answers: {total_correct_sub_answers}, Total sub answer acc: {total_sub_answer_acc}, Sub answer acc avg: {sub_answer_acc_avg}")

    print("EM: {:.1f}, F1: {:.1f}".format(all_em / len(dataset) * 100, all_f1 / len(dataset) * 100))
    print(f"Correct: {len(results['correct'])}, Wrong: {len(results['wrong'])}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
