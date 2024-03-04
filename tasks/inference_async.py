import argparse
import json
import string
import os
from src.utils import *
from src.api import get_chat_response
from tqdm import tqdm
import asyncio


def get_prompt_zeroshot(question: str, contexts: List[Dict] = None):
    if contexts is None:
        prompt = f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "
    else:
        prompt = ""
        for context in contexts:
            prompt += f"Title: {context['title']}\nContext: {context['context']}\n\n"
        prompt += f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "

    return prompt


def get_prompt_cot(question: str, cot: str = None):
    if cot is None:
        prompt = f"Question: {question}\nAnswer: Let's think step by step. "
    else:
        prompt = f"Answer the following question with just one entity:\n" \
                 f"Question: {question}\nAnswer: Let's think step by step. {cot}\n" \
                 f"Therefore, the answer (short phrase) is: "
    return prompt


def get_prompt_genread(question: str, passage: str = None):
    if passage is None:
        prompt = f"Generate a background document from Wikipedia to answer the given question.\n\n" \
                 f"Question: {question}\n"
    else:
        prompt = f"Refer to the passage below and answer the following question with just one entity. \n\n" \
                 f"Passage: {passage} \n\n" \
                 f"Question: {question} \n\n" \
                 f"The answer is"
    return prompt


async def zeroshot_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        if "contexts" in input_sample:
            prompt = get_prompt_zeroshot(input_sample["question"], contexts=input_sample["contexts"])
        else:
            prompt = get_prompt_zeroshot(input_sample["question"])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)
        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "contexts": input_sample["contexts"] if "contexts" in input_sample else [],
                "response": response,
            }
        else:
            pred_answers = extract_answer(response)
            gold_answers = input_sample["answer"]
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "pred_answer": pred_answers,
                "contexts": input_sample["contexts"] if "contexts" in input_sample else [],
                "prompt": prompt,
                "response": response,
            }
            if do_eval:
                max_em, max_f1 = qa_evaluate(pred_answers, gold_answers)
                output_sample["em"] = max_em
                output_sample["f1"] = max_f1

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


async def zeroshot_cot_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        cot_tokens: int,
        answer_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        if "cot" in input_sample:
            cot = input_sample["cot"]
        else:
            cot_prompt = get_prompt_cot(input_sample["question"])
            cot = await get_chat_response(model=model, prompt=cot_prompt, max_tokens=cot_tokens)

        if cot == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "response": cot,
            }
        else:
            answer_prompt = get_prompt_cot(input_sample["question"], clean_passage(cot))
            raw_pred = await get_chat_response(model=model, prompt=answer_prompt, max_tokens=answer_tokens)
            if raw_pred == "RESPONSE_ERROR":
                output_sample = {
                    "question": input_sample["question"],
                    "answer": input_sample["answer"],
                    "response": raw_pred,
                }
            else:
                pred_answers = extract_answer(raw_pred)
                gold_answers = input_sample["answer"]
                output_sample = {
                    "question": input_sample["question"],
                    "answer": input_sample["answer"],
                    "cot": cot,
                    "pred_answer": pred_answers,
                    "prompt": answer_prompt,
                    "response": raw_pred,
                }
                if do_eval:
                    max_em, max_f1 = qa_evaluate(pred_answers, gold_answers)
                    output_sample["em"] = max_em
                    output_sample["f1"] = max_f1

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


async def genread_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        passage_tokens: int,
        answer_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        if "passage" in input_sample:
            passage = input_sample["passage"]
        else:
            passage_prompt = get_prompt_genread(input_sample["question"])
            passage = await get_chat_response(model=model, prompt=passage_prompt, max_tokens=passage_tokens)

        if passage == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "response": passage,
            }
        else:
            answer_prompt = get_prompt_genread(input_sample["question"], clean_passage(passage))
            raw_pred = await get_chat_response(model=model, prompt=answer_prompt, max_tokens=answer_tokens)
            if raw_pred == "RESPONSE_ERROR":
                output_sample = {
                    "question": input_sample["question"],
                    "answer": input_sample["answer"],
                    "response": raw_pred,
                }
            else:
                pred_answers = extract_answer(raw_pred)
                gold_answers = input_sample["answer"]
                output_sample = {
                    "question": input_sample["question"],
                    "answer": input_sample["answer"],
                    "pred_answer": pred_answers,
                    "prompt": answer_prompt,
                    "response": raw_pred,
                    "passage": passage,
                }
                if do_eval:
                    max_em, max_f1 = qa_evaluate(pred_answers, gold_answers)
                    output_sample["em"] = max_em
                    output_sample["f1"] = max_f1

        output_queue.put_nowait(output_sample)
        print(f"[{print_now(1)}] Progress: {output_queue.qsize()}/{input_sample['total']}", end="\r", flush=True)
        input_queue.task_done()


def build_prompt_self_prompt_cot(question: str, demos: List[Dict] = None, num_demos: int = 8):
    prompt = ""
    if demos is not None:
        for demo in demos[:num_demos]:
            prompt += f"Question: {demo['question']}\nAnswer: {demo['cot']}\n"
        # prompt += f"Question: {question}\nAnswer: Let's think step by step and answer the question with just one entity.\n"
        prompt += f"Question: {question}\nAnswer: Let's think step by step:\n"
    # elif cot is not None:
    #     prompt += f"Question: {question}\n"
    #     prompt += f"Let's think step by step and answer the question with just one entity.\n" \
    #               f"{cot}\n" \
    #               f"Therefore, answer the question with just one entity: "
    else:
        raise ValueError("Either demos or cot must be provided.")

    return prompt


async def self_prompt_cot_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
        num_demos: int = 8
):
    while True:
        input_sample = await input_queue.get()
        prompt = build_prompt_self_prompt_cot(input_sample["question"], demos=input_sample["demos"], num_demos=num_demos)
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)

        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "demos": input_sample["demos"],
                "response": response,
            }
        else:
            answer_prompt = "the final answer in just one entity is: "
            answer_prompt_2 = "the final answer is: "

            # if "answer the question in just one entity: " in response:
            if answer_prompt in response:
                raw_pred = response[response.find(answer_prompt) + len(answer_prompt):]
            elif answer_prompt_2 in response:
                raw_pred = response[response.find(answer_prompt_2) + len(answer_prompt_2):]
            else:
                answer_prompt = "answer in just one entity: "
                raw_sentences = response.split("\n")
                answer_sentences = []
                for raw_sentence in raw_sentences:
                    if answer_prompt in raw_sentence.lower():
                        answer_sentences.append(raw_sentence.lower())
                if len(answer_sentences) == 0:
                    raw_pred = ""
                else:
                    final_answer_sentence = answer_sentences[-1]
                    raw_pred = final_answer_sentence[final_answer_sentence.find(answer_prompt) + len(answer_prompt):]

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


def get_prompt_manual_cot(question: str, demos: List[dict]):
    prompt = ""
    for demo in demos:
        cot = '\n'.join(demo['cot'])
        prompt += f"Question: {demo['question']}\n" \
                    f"Answer: Let's think step by step:\n{cot}\n" \
                    f"Therefore, the final answer in just one entity is: {demo['answer']}\n\n"

    prompt += f"Question: {question}\nAnswer: Let's think step by step:\n"
    return prompt

async def manual_cot_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        prompt = get_prompt_manual_cot(input_sample["question"], demos=input_sample["demos"])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)

        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "demos": input_sample["demos"],
                "response": response,
            }
        else:
            answer_prompt = "the final answer in just one entity is: "
            answer_prompt_2 = "the final answer is: "

            # if "answer the question in just one entity: " in response:
            if answer_prompt in response:
                raw_pred = response[response.find(answer_prompt) + len(answer_prompt):]
            elif answer_prompt_2 in response:
                raw_pred = response[response.find(answer_prompt_2) + len(answer_prompt_2):]
            else:
                answer_prompt = "answer in just one entity: "
                raw_sentences = response.split("\n")
                answer_sentences = []
                for raw_sentence in raw_sentences:
                    if answer_prompt in raw_sentence.lower():
                        answer_sentences.append(raw_sentence.lower())
                if len(answer_sentences) == 0:
                    raw_pred = ""
                else:
                    final_answer_sentence = answer_sentences[-1]
                    raw_pred = final_answer_sentence[final_answer_sentence.find(answer_prompt) + len(answer_prompt):]

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


def build_prompt_self_prompt(question: str, demos: List[Dict]):
    prompt = ""
    for demo in demos:
        prompt += f"Question: {demo['question']}\n"
        prompt += f"Answer: {demo['answer']}\n"
    prompt += f"Question: {question}\nAnswer:"

    return prompt


async def self_prompt_worker(
        model: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        max_tokens: int,
        do_eval: bool = False,
):
    while True:
        input_sample = await input_queue.get()
        prompt = build_prompt_self_prompt(input_sample["question"], input_sample["demos"])
        response = await get_chat_response(model=model, prompt=prompt, max_tokens=max_tokens)
        if response == "RESPONSE_ERROR":
            output_sample = {
                "question": input_sample["question"],
                "answer": input_sample["answer"],
                "demos": input_sample["demos"],
                "response": response,
            }
        else:
            if ", because" in response.lower():
                raw_pred = response[:response.find(", because")]
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
                            "manual-cot"
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
    parser.add_argument("--on_test_data", action="store_true", help="use test data")
    parser.add_argument("--on_train_data", action="store_true", help="use test data")
    parser.add_argument("--manual_cot_path", type=str, default="demos/manual-cot.json", help="use cached results")
    parser.add_argument("--retrieval", action="store_true", help="use retrieval")

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
        elif args.on_train_data:
            args.input_file = os.path.join(args.dataset_path, "processed_train.json")
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
        if args.method == "self-prompt-cot":
            task = asyncio.create_task(self_prompt_cot_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=1024,
                do_eval=args.do_eval,
                num_demos=args.num_demos
            ))
        elif args.method == "zeroshot":
            task = asyncio.create_task(zeroshot_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=args.answer_tokens,
                do_eval=args.do_eval
            ))
        elif args.method == "zeroshot-cot":
            task = asyncio.create_task(zeroshot_cot_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                cot_tokens=args.cot_tokens,
                answer_tokens=args.answer_tokens,
                do_eval=args.do_eval
            ))
        elif args.method == "genread":
            task = asyncio.create_task(genread_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                passage_tokens=args.passage_tokens,
                answer_tokens=args.answer_tokens,
                do_eval=args.do_eval
            ))
        elif args.method == "auto-cot":
            task = asyncio.create_task(auto_cot_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=args.passage_tokens,
                do_eval=args.do_eval
            ))
        elif args.method == "self-prompt":
            task = asyncio.create_task(self_prompt_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=args.passage_tokens,
                do_eval=args.do_eval
            ))
        elif args.method == "manual-cot":
            task = asyncio.create_task(manual_cot_worker(
                model=args.model_name,
                input_queue=input_queue,
                output_queue=output_queue,
                max_tokens=args.passage_tokens,
                do_eval=args.do_eval
            ))
        else:
            raise NotImplementedError

        tasks.append(task)

    total = len(dataset)
    sleep_count = 0
    manual_cot = None
    if args.method == "manual-cot":
        with open(args.manual_cot_path, "r") as f:
            manual_cot = json.load(f)

    for data in dataset:
        question = data["question"]
        raw_answers = data["gold_ans"] if "gold_ans" in data else data["answer"]
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        gold_answers = extract_answer(", ".join(raw_answers))

        input_sample = {
            "question": question,
            "answer": gold_answers,
            "total": total
        }
        if args.method in ["self-prompt-cot", "auto-cot", "self-prompt"]:
            input_sample["demos"] = data["demos"]
        elif args.method == "manual-cot":
            input_sample["demos"] = manual_cot

        if "contexts" in data:
            input_sample["contexts"] = data["contexts"]

        # if we use pre-generated step 1 file of gen-read, we skip the passage generation step
        if "passage" in data:
            input_sample["passage"] = data["passage"]
        # if we use pre-generated step 1 file of zeroshot-cot
        if "cot" in data:
            input_sample["cot"] = data["cot"]

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

    total_em = 0
    total_f1 = 0
    output_dataset = []
    if args.do_eval:
        for output in existing_dataset + succeeded_data:
            if not isinstance(output["answer"], list):
                output["answer"] = [output["answer"]]
            em, f1 = qa_evaluate(output["answer"], output["pred_answer"])
            output["em"] = em
            output["f1"] = f1
            output_dataset.append(output)
            total_em += output["em"]
            total_f1 += output["f1"]
        print(f"[{print_now(1)}] Total number of samples: {len(output_dataset)}")
        print(f"[{print_now(1)}] Total EM: {total_em / len(output_dataset)}")
        print(f"[{print_now(1)}] Total F1: {total_f1 / len(output_dataset)}")

    # write to file
    with open(output_jsonl_path, "w") as f:
        for output in output_dataset:
            f.write(json.dumps(output) + "\n")

    # transfer output file to json format
    with open(output_jsonl_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(output_dataset, f, indent=4, ensure_ascii=False)

    print(f"[{print_now(1)}] Saved to {output_jsonl_path}")


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
    print(f"[{print_now(1)}] Task Done!")








