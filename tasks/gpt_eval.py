import argparse
import json
import random
import string
import os
import datetime

import numpy as np

# from src.utils import *
# from src.api import get_chat_response
from tqdm import tqdm
import asyncio
import openai



def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    CST = datetime.timezone(t_delta, 'CST')
    now = datetime.datetime.now(CST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


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


# 100 RMB
openai.api_key = "sk-TAcdfE1XE6l9aaoLYcbEHQEmOe1oHlnwMHo1JMhbgZCv80We"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

template = {
    "system_prompt": "You are a helpful and precise assistant for checking the quality of the predicted reasoning steps.",
    "prompt_template": "[Question]\n{question}\n\n"
                       "[The Start of the Assistant 1's reasoning steps]\n{cot_1}\n\n[The End of the Assistant 1's reasoning steps]\n\n"
                       "[The Start of the Assistant 2's reasoning steps]\n{cot_2}\n\n[The End of the Assistant 2's reasoning steps]\n\n"
                       "[The Start of the Assistant 3's reasoning steps]\n{cot_3}\n\n[The End of the Assistant 3's reasoning steps]\n\n"
                       "[System]\n{prompt}\n\n",
    "defaults": {
        "prompt": "We would like to request your feedback on the performance of three AI assistants in response to the user question displayed above.\n"
                  "Please rate their responses in terms of the clearness, conciseness, comprehensibility, directness in four separate paragraphs. "
                  "In each paragraph, each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n"
                  "Please first output in each paragraph a single line containing only three values indicating the scores for Assistant 1, 2 and 3, respectively. "
                  "The three scores are separated by a space. In the subsequent line of each paragraph, please provide a comprehensive explanation of your evaluation, avoiding the same score for multiple assistants and ensuring that the order in which the responses were presented does not affect your judgment."
        },
    "description": "Prompt for general questions",
    "category": "general"
}


async def get_chat_response(
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
):

    message = [
        {
            "role": "system",
            "content": template["system_prompt"]
         },
        {"role": "user", "content": prompt}
    ]
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except:
        return "RESPONSE_ERROR"


def get_prompt(question, cot_1, cot_2, cot_3):
    prompt = template["prompt_template"].format(
        question=question,
        cot_1=cot_1,
        cot_2=cot_2,
        cot_3=cot_3,
        prompt=template["defaults"]["prompt"]
    )
    return prompt


async def evaluate_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
):
    while True:
        input_sample = await input_queue.get()
        index = np.arange(3)
        np.random.shuffle(index)
        cots = [input_sample["auto_cot"], input_sample["zeroshot_cot"], input_sample["self_prompt_cot"]]
        cot_names = ["auto_cot", "zeroshot_cot", "self_prompt_cot"]
        prompt = get_prompt(input_sample["question"], cots[index[0]], cots[index[1]], cots[index[2]])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0)

        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "auto_cot": input_sample["auto_cot"],
                "zeroshot_cot": input_sample["zeroshot_cot"],
                "self_prompt_cot": input_sample["self_prompt_cot"],
                "response": response,
            }
        else:
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "cots": {
                    cot_names[index[0]]: cots[index[0]],
                    cot_names[index[1]]: cots[index[1]],
                    cot_names[index[2]]: cots[index[2]],
                },
                "response": response
            }

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-Prompt-CoT")

    # Default parameters
    parser.add_argument("--zeroshot_cot_file", type=str, default=None)
    parser.add_argument("--auto_cot_file", type=str, default=None)
    parser.add_argument("--self_prompt_cot_file", type=str, default=None)
    parser.add_argument("--task", type=str, default="musique-qa")

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name", type=str, default="gpt-4-0613", help="model used for response generation.")
    parser.add_argument(
        "--answer_tokens", type=int, default=1000, help="maximum length of output tokens by model for zero-shot")
    parser.add_argument("--limit_dataset_size", type=int, default=-1, help="limit dataset size for debugging")
    parser.add_argument("--max_concurrency", type=int, default=1, help="limit dataset size for debugging")
    parser.add_argument("--flag", type=str, default="", help="flag for output file")
    parser.add_argument("--sleep_every_n_requests", type=int, default=3500, help="flag for output file")
    parser.add_argument("--num_demos", type=int, default=8, help="Number of demos to use")
    parser.add_argument("--use_cache", action="store_true", help="use cached results")
    parser.add_argument("--on_test_data", action="store_true", help="use cached results")

    args = parser.parse_args()

    if args.dataset_path is None:
        print(f"[{print_now(1)}] Using default dataset path for {args.task} ...")
        args.dataset_path = DEFAULT_PATHS[args.task]
    else:
        print(f"[{print_now(1)}] Using dataset path {args.dataset_path} ...")

    args.save_path = os.path.join(DEFAULT_PATHS[args.task], args.model_name)

    os.makedirs(args.save_path, exist_ok=True)

    return args


async def main(args: argparse.Namespace):

    fix_seed(args.random_seed)
    print(f"[{print_now(1)}] Initializing model: {args.model_name} on {args.task} ...")

    print(f"[{print_now(1)}] Loading zeroshot-cot from {args.zeroshot_cot_file} ...")
    zeroshotcot_dataset = load_json_and_jsonl_file(args.zeroshot_cot_file)
    zeroshotcot_dataset = [example for example in zeroshotcot_dataset if example["em"] == 1]
    print(f"[{print_now(1)}] Loading auto-cot from {args.auto_cot_file} ...")
    autocot_dataset = load_json_and_jsonl_file(args.auto_cot_file)
    autocot_dataset = [example for example in autocot_dataset if example["em"] == 1]
    print(f"[{print_now(1)}] Loading self-prompt-cot from {args.self_prompt_cot_file} ...")
    selfpromptcot_dataset = load_json_and_jsonl_file(args.self_prompt_cot_file)
    selfpromptcot_dataset = [example for example in selfpromptcot_dataset if example["em"] == 1]

    output_dir = args.save_path
    if args.flag != "":
        output_jsonl_path = os.path.join(output_dir, f"gpt4_evaluate-{args.flag}.jsonl")
    else:
        output_jsonl_path = os.path.join(output_dir, f"gpt4_evaluate.jsonl")

    if args.use_cache and os.path.exists(output_jsonl_path.replace(".jsonl", ".json")):
        with open(output_jsonl_path.replace(".jsonl", ".json"), "r") as f:
            existing_dataset = json.load(f)
    else:
        existing_dataset = []

    dataset = []
    for spc_sample in selfpromptcot_dataset:
        pass_flag = True
        atc_sample = None
        zsc_sample = None

        for sample in zeroshotcot_dataset:
            if sample["question"] == spc_sample["question"]:
                pass_flag = False
                zsc_sample = sample
                break
        if pass_flag:
            continue
        pass_flag = True
        for sample in autocot_dataset:
            if sample["question"] == spc_sample["question"]:
                pass_flag = False
                atc_sample = sample
                break
        if pass_flag:
            continue

        for exist_sample in existing_dataset:
            if exist_sample["question"] == spc_sample["question"]:
                pass_flag = True
                break
        if pass_flag:
            continue

        dataset.append({
            "question": spc_sample["question"],
            "answer": spc_sample["answer"],
            "auto_cot": " ".join(atc_sample["response"]),
            "zeroshot_cot": zsc_sample["cot"],
            "self_prompt_cot": "Let's think step by step: "+" ".join(spc_sample["response"]),
        })
    with open("eval_musique.json", "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    # raise Exception("stop")

    # print(f"[{print_now(1)}] Total number of samples: {len(dataset)}")
    random.seed(args.random_seed)
    if args.limit_dataset_size > len(existing_dataset):
        num_samples = min(args.limit_dataset_size-len(existing_dataset), len(dataset))
        if num_samples > 0:
            dataset = random.sample(dataset, num_samples)
        else:
            dataset = []
    elif args.limit_dataset_size > 0:
        dataset = []
    print(f"[{print_now(1)}] Total number of samples: {len(dataset)}")

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    tasks: List[asyncio.Task] = []
    for _ in range(args.max_concurrency):
        task = asyncio.create_task(evaluate_worker(
            model=args.model_name,
            input_queue=input_queue,
            output_queue=output_queue,
            max_tokens=1024,
        ))

        tasks.append(task)

    total = len(dataset)
    sleep_count = 0
    for data in dataset:
        question = data["question"]
        raw_answers = data["gold_ans"] if "gold_ans" in data else data["answer"]
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        # gold_answers = extract_answer(", ".join(raw_answers))

        input_sample = {
            "question": question,
            "answer": raw_answers,
            "zeroshot_cot": data["zeroshot_cot"],
            "auto_cot": data["auto_cot"],
            "self_prompt_cot": data["self_prompt_cot"],
            "total": total
        }

        sleep_count += 1
        input_queue.put_nowait(input_sample)
        if sleep_count >= min(args.sleep_every_n_requests, 3500):
            await asyncio.sleep(60)
            sleep_count = 0

    await input_queue.join()

    outputs = []
    for _ in range(output_queue.qsize()):
        outputs.append(await output_queue.get())

    failed_data = []
    succeeded_data = []
    #
    for output in outputs:
        if output["response"] == "RESPONSE_ERROR":
            failed_data.append(output)
        else:
            succeeded_data.append(output)
    print(f"[{print_now(1)}] Succeeded: {len(succeeded_data)} Failed: {len(failed_data)}")

    # write to file
    with open(output_jsonl_path, "w") as f:
        for output in existing_dataset + succeeded_data:
            f.write(json.dumps(output) + "\n")

    # transfer output file to json format
    with open(output_jsonl_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(existing_dataset + succeeded_data, f, indent=4)

    print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    sleep_count = 0
    for failed_sample in failed_data:
        failed_sample["total"] = len(failed_data)
        input_queue.put_nowait(failed_sample)
        sleep_count += 1
        if sleep_count >= min(args.sleep_every_n_requests, 3500):
            await asyncio.sleep(60)
            sleep_count = 0
    print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    await input_queue.join()
    print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")

    new_outputs = []
    for _ in range(output_queue.qsize()):
        new_outputs.append(await output_queue.get())
    print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")

    failed_data = []
    for output in new_outputs:
        if output["response"] == "RESPONSE_ERROR":
            failed_data.append(output)
        else:
            succeeded_data.append(output)
    print(f"[{print_now(1)}] Succeeded: {len(succeeded_data)} Failed: {len(failed_data)}")


    # write to file
    with open(output_jsonl_path, "w") as f:
        for output in existing_dataset + succeeded_data:
            f.write(json.dumps(output) + "\n")

    eval_dataset = []
    for output_sample in existing_dataset + succeeded_data:
        response = output_sample["response"]
        if isinstance(response, str):
            results = response.split("\n\n")
        else:
            results = response
            output_sample["response"] = response.split("\n\n")
        cot_names = list(output_sample["cots"].keys())
        result_names = ["clearness", "conciseness", "completeness", "directness"]
        success_flag = True
        if len(results) != 4:
            success_flag = False
        if not success_flag:
            continue

        for i in range(4):
            results_split = results[i].split("\n")
            if len(results_split) < 1:
                success_flag = False
                break
            scores = [score for score in results_split[0].split(" ")]
            try:
                scores = [float(score) for score in scores]
            except:
                success_flag = False
                break
            if len(scores) != 3:
                success_flag = False
                break

            explanation = results[i].split("\n")[1] if len(results[i].split("\n")) > 1 else ""
            output_sample[f"{result_names[i]}"] = {
                cot_names[0]: scores[0],
                cot_names[1]: scores[1],
                cot_names[2]: scores[2],
                "explanation": explanation
            }

        if success_flag:
            output_sample["overall"] = {
                cot_names[0]: np.mean(
                    [output_sample[result_name][cot_names[0]] for result_name in result_names]),
                cot_names[1]: np.mean(
                    [output_sample[result_name][cot_names[1]] for result_name in result_names]),
                cot_names[2]: np.mean(
                    [output_sample[result_name][cot_names[2]] for result_name in result_names]),
            }
            eval_dataset.append(output_sample)
    print(f"[{print_now(1)}] Evaluated: {len(eval_dataset)} Failed: {len(succeeded_data) - len(eval_dataset)}")

    mean_zsc = []
    mean_atc = []
    mean_spc = []
    clearness_zsc = []
    clearness_atc = []
    clearness_spc = []
    conciseness_zsc = []
    conciseness_atc = []
    conciseness_spc = []
    completeness_zsc = []
    completeness_atc = []
    completeness_spc = []
    directness_zsc = []
    directness_atc = []
    directness_spc = []

    output_dataset = []
    for output in eval_dataset:
        output_dataset.append(output)
        mean_zsc.append(output["overall"]["zeroshot_cot"])
        mean_atc.append(output["overall"]["auto_cot"])
        mean_spc.append(output["overall"]["self_prompt_cot"])
        clearness_zsc.append(output["clearness"]["zeroshot_cot"])
        clearness_atc.append(output["clearness"]["auto_cot"])
        clearness_spc.append(output["clearness"]["self_prompt_cot"])
        conciseness_zsc.append(output["conciseness"]["zeroshot_cot"])
        conciseness_atc.append(output["conciseness"]["auto_cot"])
        conciseness_spc.append(output["conciseness"]["self_prompt_cot"])
        completeness_zsc.append(output["completeness"]["zeroshot_cot"])
        completeness_atc.append(output["completeness"]["auto_cot"])
        completeness_spc.append(output["completeness"]["self_prompt_cot"])
        directness_zsc.append(output["directness"]["zeroshot_cot"])
        directness_atc.append(output["directness"]["auto_cot"])
        directness_spc.append(output["directness"]["self_prompt_cot"])

    print(f"[{print_now(1)}] Total number of samples: {len(output_dataset)}")
    print(f"[{print_now(1)}] zsc mean: {np.mean(mean_zsc)}, clearness: {np.mean(clearness_zsc)}, conciseness: {np.mean(conciseness_zsc)}, completeness: {np.mean(completeness_zsc)}, directness: {np.mean(directness_zsc)}")
    print(f"[{print_now(1)}] atc mean: {np.mean(mean_atc)}, clearness: {np.mean(clearness_atc)}, conciseness: {np.mean(conciseness_atc)}, completeness: {np.mean(completeness_atc)}, directness: {np.mean(directness_atc)}")
    print(f"[{print_now(1)}] spc mean: {np.mean(mean_spc)}, clearness: {np.mean(clearness_spc)}, conciseness: {np.mean(conciseness_spc)}, completeness: {np.mean(completeness_spc)}, directness: {np.mean(directness_spc)}")



    # transfer output file to json format
    with open(output_jsonl_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(output_dataset, f, indent=4, ensure_ascii=False)

    print(f"[{print_now(1)}] Saved to {output_jsonl_path}")


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
    print(f"[{print_now(1)}] Task Done!")








