import argparse
import json
import string
import os
from src.utils import *
from src.api import get_chat_response
from tqdm import tqdm
import asyncio


def get_prompt_decompose(question: str, demos: List[Dict] = None):
    deco_trigger = "Decompose this question into direct and concise sub-questions (denote the answer of Q1 as #1):\n"
    prompt = ""
    if demos is None:
        prompt += f"Question: {question}\nDecompose the question into sub-questions:\n"
    else:
        for demo in demos:
            prompt += f"Question: {demo['question']}\n{deco_trigger}{demo['decomposition']}\n"
            # for i, qa in enumerate(demo["hops"]):
            #     prompt += f"{i + 1}. {qa['question']}\n"
            # prompt += "\n"

        prompt += f"Question: {question}\n{deco_trigger}"

    return prompt

def get_prompt_extract_entity(question: str):
    prompt = f"Extract the key named entity in the following question: {question}\nAnswer in just one entity:"
    return prompt

def get_prompt_generate_passage(entity: str):
    prompt = f"Generate a wikipedia passage about {entity}: "
    return prompt

def get_prompt_generate_answer(question: str, passage: str):
    prompt = f"Refer to the passage below and answer the following question with just one entity. \n\n" \
             f"Passage: {passage}\n\n" \
             f"Question: {question}\n\n" \
             f"Answer in just one entity: "
    return prompt



def get_prompt_conclude(question: str, sub_qas: List[Dict]):
    prompt = f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "
    return prompt


async def stage_1_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
        num_demos: int = 8,
):
    while True:
        input_sample = await input_queue.get()
        prompt = get_prompt_decompose(input_sample["question"], input_sample["demos"][: num_demos])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)
        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "demos": input_sample["demos"],
                "response": response,
            }
        else:
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                # "pred_answer": pred_answers,
                "prompt": prompt.split("\n"),
                "response": response.split("\n"),
            }
        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


async def stage_2_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        passage_tokens: int,
        answer_tokens: int,
):
    while True:
        input_sample = await input_queue.get()
        sub_questions = input_sample["response"]
        output_sample = {
            "question": input_sample["question"],
            "answer": input_sample["answer"],
            "decomposition": sub_questions,
            "sub_qas": [],
            "response": "SUCCESS",
        }
        temp_answers = []
        for i, sub_question in enumerate(sub_questions):
            if i > 0 and "#1" in sub_question:
                sub_question = sub_question.replace("#1", temp_answers[0])
            if i > 1 and "#2" in sub_question:
                sub_question = sub_question.replace("#2", temp_answers[1])
            if i > 2 and "#3" in sub_question:
                sub_question = sub_question.replace("#3", temp_answers[2])

            prompt_entity = get_prompt_extract_entity(sub_question)
            entity = await get_chat_response(model=model, prompt=prompt_entity, max_tokens=answer_tokens)
            if entity == "RESPONSE_ERROR":
                output_sample["response"] = "RESPONSE_ERROR"
                temp_answers.append("RESPONSE_ERROR")
            else:
                prompt_passage = get_prompt_generate_passage(entity)
                passage = await get_chat_response(model=model, prompt=prompt_passage, max_tokens=passage_tokens)
                if passage == "RESPONSE_ERROR":
                    output_sample["response"] = "RESPONSE_ERROR"
                    temp_answers.append("RESPONSE_ERROR")
                else:
                    prompt_answer = get_prompt_generate_answer(sub_question, passage)
                    answer = await get_chat_response(model=model, prompt=prompt_answer, max_tokens=answer_tokens)
                    if answer == "RESPONSE_ERROR":
                        output_sample["response"] = "RESPONSE_ERROR"
                        temp_answers.append("RESPONSE_ERROR")
                    else:
                        output_sample["sub_qas"].append(
                            {
                                "question": sub_question,
                                "entity": entity,
                                "passage": passage,
                                "answer": answer,
                            }
                        )
                        temp_answers.append(answer)

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()





def build_prompt_auto_cot(question: str, demos: List[Dict]):
    prompt = ""
    for demo in demos:
        prompt += f"Question: {demo['question']}\n" \
                  f"Answer: {demo['cot']}\n" \
                  f"Therefore, the answer (in just one entity) is: {demo['raw_pred']}\n"
    prompt += f"Question: {question}\nAnswer:"

    return prompt


async def auto_cot_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        prompt = build_prompt_auto_cot(input_sample["question"], input_sample["demos"])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)
        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "demos": input_sample["demos"],
                "response": response,
            }
        else:
            answer_prompt = "therefore, the answer (in just one entity) is: "
            if answer_prompt in response.lower():
                raw_pred = response[response.lower().find(answer_prompt) + len(answer_prompt):]
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
            pred_answers = extract_answer(raw_pred)
            gold_answers = input_sample["answer"]
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "pred_answer": pred_answers,
                "prompt": prompt.split("\n"),
                "response": response.split("\n"),
            }
            if do_eval:
                max_em, max_f1 = qa_evaluate(pred_answers, gold_answers)
                output_sample["em"] = max_em
                output_sample["f1"] = max_f1

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-Prompt-CoT")

    # Default parameters
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--task", type=str, default="cweb-qa",
                        choices=list(DEFAULT_PATHS.keys()), help="dataset name")

    parser.add_argument("--method", type=str, default="genread",
                        choices=[
                            "zeroshot",
                            "zeroshot-cot",
                            "genread",
                            "self-prompt",
                            "auto-cot",
                            "self-prompt-cot",
                            "stage_1",
                            "stage_2"
                        ],
                        help="method name")

    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--do_eval", action="store_true", help="do evaluation")
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--model_name", type=str, default="gpt-3.5-turbo-0301", help="model used for response generation.")

    parser.add_argument(
        "--answer_tokens", type=int, default=10, help="maximum length of output tokens by model for zero-shot")

    parser.add_argument(
        "--cot_tokens", type=int, default=256, help="maximum length of output tokens by model for CoT")
    parser.add_argument(
        "--passage_tokens", type=int, default=256, help="maximum length of output tokens by model for CoT")
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

    if args.input_file is None:
        if args.on_test_data:
            args.input_file = os.path.join(args.dataset_path, "processed_test.json")
        else:
            args.input_file = os.path.join(args.dataset_path, "processed_dev.json")
        if not os.path.exists(args.input_file):
            raise ValueError(f"Default file {args.input_file} does not exist.")

    print(f"[{print_now(1)}] Loading dataset from {args.input_file} ...")

    os.makedirs(args.save_path, exist_ok=True)

    return args


async def main(args: argparse.Namespace):

    fix_seed(args.random_seed)
    print(f"[{print_now(1)}] Initializing model: {args.model_name} on {args.task} ...")

    raw_dataset = load_json_and_jsonl_file(args.input_file)
    if args.limit_dataset_size > 0:
        raw_dataset = raw_dataset[:args.limit_dataset_size]

    output_dir = args.save_path
    if args.flag != "":
        output_jsonl_path = os.path.join(output_dir, f"{args.method}-{args.flag}.jsonl")
    else:
        output_jsonl_path = os.path.join(output_dir, f"{args.method}.jsonl")

    if args.use_cache:
        with open(output_jsonl_path.replace(".jsonl", ".json"), "r") as f:
            existing_dataset = json.load(f)
    else:
        existing_dataset = []

    dataset = []
    for raw_sample in raw_dataset:
        duplicate_flag = False
        for sample in existing_dataset:
            if sample["question"] == raw_sample["question"]:
                duplicate_flag = True
                break
        if duplicate_flag:
            continue
        dataset.append(raw_sample)

    print(f"[{print_now(1)}] Total number of samples: {len(dataset)}")

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    tasks: List[asyncio.Task] = []
    for _ in range(args.max_concurrency):
        if args.method == "stage_1":
            task = asyncio.create_task(stage_1_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=1024,
                do_eval=args.do_eval,
                num_demos=args.num_demos
            ))
        elif args.method == "stage_2":
            task = asyncio.create_task(stage_2_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                passage_tokens=1024,
                answer_tokens=20
            ))

        else:
            raise NotImplementedError

        tasks.append(task)

    total = len(dataset)
    sleep_count = 0
    for data in dataset:
        question = data["question"]
        raw_answers = data["gold_ans"] if "gold_ans" in data else data["answer"]
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        # gold_answers = extract_answer(", ".join(raw_answers))
        gold_answers = raw_answers

        input_sample = {
            "question": question,
            "answer": gold_answers,
            "total": total
        }
        if args.method in ["stage_1", "auto-cot", "self-prompt"]:
            input_sample["demos"] = data["demos"]

        # if we use pre-generated step 1 file of gen-read, we skip the passage generation step
        if "passage" in data:
            input_sample["passage"] = data["passage"]
        # if we use pre-generated step 1 file of zeroshot-cot
        if "cot" in data:
            input_sample["cot"] = data["cot"]
        if "response" in data:
            input_sample["response"] = data["response"]

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

    # print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    # sleep_count = 0
    # for failed_sample in failed_data:
    #     failed_sample["total"] = len(failed_data)
    #     input_queue.put_nowait(failed_sample)
    #     sleep_count += 1
    #     if sleep_count >= min(args.sleep_every_n_requests, 3500):
    #         await asyncio.sleep(60)
    #         sleep_count = 0
    # print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    # await input_queue.join()
    # print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    #
    # new_outputs = []
    # for _ in range(output_queue.qsize()):
    #     new_outputs.append(await output_queue.get())
    # print(f"Next loop, len(failed_data): {len(failed_data)}, input_queue.qsize(): {input_queue.qsize()}, output_queue.qsize(): {output_queue.qsize()}")
    #
    # failed_data = []
    # for output in new_outputs:
    #     if output["response"] == "RESPONSE_ERROR":
    #         failed_data.append(output)
    #     else:
    #         succeeded_data.append(output)
    # print(f"[{print_now(1)}] Succeeded: {len(succeeded_data)} Failed: {len(failed_data)}")

    # total_em = 0
    # total_f1 = 0
    # output_dataset = []
    # if args.do_eval:
    #     for output in existing_dataset + succeeded_data:
    #         if not isinstance(output["answer"], list):
    #             output["answer"] = [output["answer"]]
    #         em, f1 = qa_evaluate(output["answer"], output["pred_answer"])
    #         output["em"] = em
    #         output["f1"] = f1
    #         output_dataset.append(output)
    #         total_em += output["em"]
    #         total_f1 += output["f1"]
    #     print(f"[{print_now(1)}] Total number of samples: {len(output_dataset)}")
    #     print(f"[{print_now(1)}] Total EM: {total_em / len(output_dataset)}")
    #     print(f"[{print_now(1)}] Total F1: {total_f1 / len(output_dataset)}")

    # # write to file
    # with open(output_jsonl_path, "w") as f:
    #     for output in output_dataset:
    #         f.write(json.dumps(output) + "\n")
    #
    # # transfer output file to json format
    # with open(output_jsonl_path.replace(".jsonl", ".json"), "w") as f:
    #     json.dump(output_dataset, f, indent=4, ensure_ascii=False)

    print(f"[{print_now(1)}] Saved to {output_jsonl_path}")


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
    print(f"[{print_now(1)}] Task Done!")








