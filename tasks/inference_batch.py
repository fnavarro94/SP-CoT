import argparse
import json
import string
import os
from src.utils import *
# from src.api import ChatGPT, CompleteGPT
from src.models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


os.environ["HF_DATASETS_CACHE"] = "/data2/.cache/huggingface/datasets"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-Prompt-CoT")

    # Default parameters
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--task", type=str, default="complex-qa",
                        choices=list(DEFAULT_PATHS.keys()), help="dataset name")

    parser.add_argument("--method", type=str, default="baseline",
                        choices=[
                            "baseline",
                            "zeroshot",
                            "retrieval_only",
                            "zeroshot-cot",
                            "genread",
                            "self-prompt",
                            "auto-cot",
                            "self-prompt-cot",
                            "manual-cot"
                        ],
                        help="method name")

    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--do_eval", action="store_true", help="do evaluation")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_demos", type=int, default=8)

    parser.add_argument(
        # "--model_name", type=str, default="gpt-3.5-turbo-0301", help="model used for response generation.")
        "--model_name", type=str, default="dpr", help="model used for response generation.")

    parser.add_argument(
        "--answer_tokens", type=int, default=20, help="maximum length of output tokens by model for zero-shot")

    parser.add_argument(
        "--cot_tokens", type=int, default=256, help="maximum length of output tokens by model for CoT")
    parser.add_argument(
        "--passage_tokens", type=int, default=256, help="maximum length of output tokens by model for CoT")
    parser.add_argument("--limit_dataset_size", type=int, default=-1, help="limit dataset size for debugging")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for inference")
    parser.add_argument("--on_test_data", action="store_true", help="use test data for inference")
    parser.add_argument("--on_train_data", action="store_true", help="use train data for inference")
    parser.add_argument("--manual_cot_path", type=str, default="demos/manual-cot.json", help="path to manual cot file")
    parser.add_argument("--flag", type=str, default="", help="flag for output file")

    args = parser.parse_args()

    args.dataset_path = DEFAULT_PATHS[args.task]
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


def get_prompt_zeroshot(question: str, model: str = "default", contexts: List[Dict] = None):
    question = question.replace(" ?", "?")

    prompt = ""
    if contexts:
        if "vicuna" in model or "alpaca" in model or "wizard" in model:
            for context in contexts:
                prompt += f"Title: {context['title']}\nPassage: {context['context']}\n"
        else:
            for context in contexts:
                prompt += f"Title: {context['title']}\nContext: {context['context']}\n"

    if model == "falcon-7b":
        # prompt = f"Question: {question}\nAnswer (in one entity):" # 0.175
        # prompt = f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "  # 0.025
        prompt += f"Question: {question}\nAnswer in just one entity:" # 0.175
    elif model == "gpt-neoxt-chat":
        prompt += f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "
    # elif model == "vicuna-13b":
    #     prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
    #              "Write a response that appropriately completes the request.\n\n" \
    #              f"### Instruction:\nAnswer the question with just one entity\n\n" \
    #              f"### Input:\n{question}\n\n" \
    #              "### Response:\n"
    elif "vicuna" in model or "alpaca" in model or "wizard" in model:
        prompt += f"Question: {question}\nAnswer with just one entity:"

    else:
        prompt += f"Answer the following question with just one entity:\nQuestion: {question}\nAnswer: "

    return prompt


def get_prompt_baseline(question: str):
    prompt = f"Question: {question}\nAnswer: "
    return prompt


def get_prompt_cot(question: str, cot: str = None):
    if cot is None:
        prompt = f"Question: {question}\nAnswer: Let's think step by step."
    else:
        prompt = f"Answer the following question with just one entity:\n" \
                 f"Question: {question}\nAnswer: Let's think step by step. {cot}\n" \
                 f"Therefore, the answer (short phrase) is: "
    return prompt



def get_prompt_genread(question: str, passage: str = None):
    if passage is None:
        prompt = f"Generate a background document from Wikipedia to answer the given question.\n" \
                 f"Question: {question}\n"
    else:
        prompt = f"Refer to the passage below and answer the following question with just one entity. \n" \
                 f"Passage: {passage} \n" \
                 f"Question: {question} \n" \
                 f"The answer is"
    return prompt


def build_prompt_self_prompt_cot(question: str, demos: List[Dict], num_demos: int = 8, model: str = "default"):
    prompt = ""

    if model == "gpt-neoxt-chat":
        # for demo in demos[:num_demos]:
        #     raw_question = demo['question'].replace(" ?", "?")
        #     cot = "\n".join(demo['cot'].split("\n")[1:])
        #     prompt += f"<human>: Question: {raw_question}\n" \
        #               f"Answer the question step by step:\n" \
        #               f"<bot>: {cot}\n"
        # prompt += f"<human>: Question: {question.replace(' ?', '?')}\n" \
        #           f"Answer the question step by step:"
        # prompt = prompt[9:]

        # for demo in demos[:num_demos]:
        #     raw_question = demo['question'].replace(" ?", "?")
        #     cot = "\n".join(demo['cot'].split("\n")[1:])
        #     prompt += f"Question: {raw_question}\n" \
        #               f"Answer the question step by step:\n{cot}\n"
        # prompt += f"Question: {question.replace(' ?', '?')}\n" \
        #           f"Answer the question step by step:\n"

        for demo in demos[:num_demos]:
            raw_question = demo['question'].replace(" ?", "?")
            prompt += f"Question: {raw_question}\nAnswer: {demo['cot']}\n"
        prompt += f"Question: {question.replace(' ?', '?')}\nAnswer: Let's think step by step:\n"


    else:
        for demo in demos[:num_demos]:
            raw_question = demo['question'].replace(" ?", "?")
            prompt += f"Question: {raw_question}\nAnswer: {demo['cot']}\n"
        prompt += f"Question: {question.replace(' ?', '?')}\nAnswer: Let's think step by step:\n"
    return prompt


def build_prompt_self_prompt(question: str, demos: List[Dict], num_demos: int = 8):
    prompt = ""
    for demo in demos[:num_demos]:
        prompt += f"Question: {demo['question']}\n"
        prompt += f"Answer: {demo['answer']}\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def build_prompt_auto_cot(question: str, demos: List[Dict]):
    prompt = ""
    for demo in demos:
        raw_question = demo['question'].replace(" ?", "?")
        prompt += f"Question: {raw_question}\n"
        answer = demo['answer'] if "answer" in demo else demo["pred_ans"]
        if isinstance(answer, list):
            answer = answer[0]
        cot = demo["cot"]
        prompt += f"Answer: {cot}. The answer is: {answer}\n\n"
    raw_question = question.replace(" ?", "?")
    prompt += f"Question: {raw_question}\nAnswer:"
    return prompt

def build_prompt_manual_cot(question: str, demos: List[Dict]):
    prompt = ""
    for demo in demos:
        cot = '\n'.join(demo['cot'])
        raw_question = demo['question'].replace(" ?", "?")
        prompt += f"Question: {raw_question}\n" \
                    f"Answer: Let's think step by step:\n{cot}\n" \
                    f"Therefore, the final answer in just one entity is: {demo['answer']}\n\n"
    raw_question = question.replace(" ?", "?")
    prompt += f"Question: {raw_question}\nAnswer: Let's think step by step:\n"
    return prompt


def main(args: argparse.Namespace):

    fix_seed(args.random_seed)

    print(f"[{print_now(1)}] Initializing model: {args.model_name} on {args.task} ...")
    # if args.model_name in [
    #     "gpt-3.5-turbo-0301"
    # ]:
    #     model = ChatGPT(model_name=args.model_name)
    # elif args.model_name in [
    #     "text-davinci-002"
    # ]:
    #     model = CompleteGPT(model_name=args.model_name)
    if args.model_name in [
        "gpt-neoxt-chat"
    ]:
        model = GPTNeoXTChat(model_name=args.model_name)
    elif args.model_name in [
        "gpt-neox"
    ]:
        model = GPTNeoX(model_name=args.model_name)
    elif args.model_name in [
        "flan-ul2", "flan-t5-xxl"
    ]:
        model = Flan_UL2_20B(model_name=args.model_name)
    elif args.model_name in [
        "t5-11b-ssm", "t5-3b-ssm", "t5-large-ssm"
    ]:
        model = T5_SSM(model_name=args.model_name)
    elif args.model_name in [
        "alpaca-7b", "alpaca-13b", "vicuna-13b"
    ]:
        model = Alpaca(model_name=args.model_name)
    elif args.model_name in [
        "falcon-7b"
    ]:
        model = Falcon(model_name=args.model_name)
    elif args.model_name in [
        "wizard-13b"
    ]:
        model = Wizard(model_name=args.model_name)

    elif args.model_name == "rag_token":
        model = RAG_Token()
    elif args.model_name == "rag_sequence":
        model = RAG_Sequence()
    elif args.model_name.lower() == "dpr":
        model = DPR()
    elif args.model_name.lower() == "realm":
        model = REALM()
    else:
        raise ValueError("model_name is invalid.")

    dataset = load_json_and_jsonl_file(args.input_file)

    if args.limit_dataset_size > 0:
        dataset = dataset[:args.limit_dataset_size]

    total = 0
    total_em = 0
    total_f1 = 0

    output_dir = args.save_path

    if args.flag != "":
        output_jsonl_path = os.path.join(output_dir, f"{args.method}-{args.flag}.jsonl")
    else:
        output_jsonl_path = os.path.join(output_dir, f"{args.method}.jsonl")

    output_file = open(output_jsonl_path, "w")
    output_dataset = []

    dataloader = setup_data_loader(
        data=dataset,
        seed=args.random_seed,
        batch_size=args.batch_size
    )

    manual_cot = None
    if args.method == "manual-cot":
        with open(args.manual_cot_path, "r") as f:
            manual_cot = json.load(f)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Running inference, Model={args.model_name}, Method={args.method}, Task={args.task} ...")):
        question_batch = batch["question"]
        total += len(question_batch)
        raw_answers_batch = batch["gold_ans"] if "gold_ans" in batch else batch["answer"]
        answer_batch = [extract_answer(raw_answer) for raw_answer in raw_answers_batch]
        output_sample_batch = []

        if args.method == "self-prompt-cot":
            prompt_batch = []
            for i in range(len(question_batch)):
                demos = [{"question": demo["question"][i], "cot": demo["cot"][i], "answer": demo["answer"][i]} for demo in batch["demos"]]
                prompt_batch.append(build_prompt_self_prompt_cot(question_batch[i], demos, num_demos=args.num_demos, model=args.model_name))

            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    use_temp=False,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            elif args.model_name in ["alpaca-13b", "falcon-7b", "vicuna-13b", "wizard-13b"]:
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            else:
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.cot_tokens)

            # response_batch = [response.split("\n\n")[0] for response in response_batch]
            for i in range(len(response_batch)):
                response = response_batch[i]
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
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i].split("\n"),
                    "response": response_batch[i].split("\n")
                }
                output_sample_batch.append(output_sample)

        elif args.method == "zeroshot":
            prompt_batch = []
            for i in range(len(question_batch)):
                if "contexts" in batch:
                    contexts = [{
                        "context": sample["context"][i],
                        "title": sample["title"][i],
                    }
                        for sample in batch["contexts"]
                    ]
                    prompt_batch.append(get_prompt_zeroshot(question_batch[i], model=args.model_name, contexts=contexts))



            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.answer_tokens, use_temp=True)
            elif args.model_name in ["alpaca-13b", "falcon-7b", "vicuna-13b", "wizard-13b"]:
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.answer_tokens,
                    ignore_double_newline=True
                )

            else:
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.answer_tokens)

            for i in range(len(response_batch)):
                response = response_batch[i]
                raw_pred = response

                if len(raw_pred) != 0:
                    if raw_pred[0] == "\n":
                        raw_pred = raw_pred[1:]

                    if "answer: " in raw_pred.lower():
                        raw_pred = raw_pred[raw_pred.lower().find("answer: ") + len("answer: "):]
                        raw_pred = raw_pred.split("\n")[0].strip()

                    if "\n" in raw_pred:
                        raw_pred = raw_pred.split("\n")[0].strip()

                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i].split("\n"),
                    "response": response.split("\n")
                }
                output_sample_batch.append(output_sample)

        elif args.method == "zeroshot-cot":
            cot_prompt_batch = [get_prompt_cot(question) for question in question_batch]
            cot_batch = model.get_response_batch(cot_prompt_batch, max_tokens=args.cot_tokens)
            cot_batch = [clean_passage(cot) for cot in cot_batch]
            answer_prompt_batch = [get_prompt_cot(question, cot) for question, cot in zip(question_batch, cot_batch)]
            response_batch = model.get_response_batch(answer_prompt_batch, max_tokens=args.answer_tokens)
            for i in range(len(response_batch)):
                response = response_batch[i]
                raw_pred = response
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": answer_prompt_batch[i].split("\n"),
                    "response": response.split("\n")
                }
                output_sample_batch.append(output_sample)

        elif args.method == "self-prompt":
            prompt_batch = []
            for i in range(len(question_batch)):
                demos = [{"question": demo["question"][i], "answer": demo["answer"][i]} for demo in batch["demos"]]
                prompt_batch.append(build_prompt_self_prompt(question_batch[i], demos))
            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.passage_tokens, use_temp=False)
            else:
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.passage_tokens)
            for i in range(len(response_batch)):
                response = response_batch[i]
                if ", because" in response.lower():
                    raw_pred = response[:response.find(", because")]
                else:
                    raw_sentences = response.split("\n")
                    answer_sentences = []
                    for raw_sentence in raw_sentences:
                        if "answer: " in raw_sentence.lower():
                            answer_sentences.append(raw_sentence)
                    if len(answer_sentences) == 0:
                        raw_pred = response
                    else:
                        final_answer_sentence = answer_sentences[-1]
                        raw_pred = final_answer_sentence[final_answer_sentence.find("answer: ") + len("answer: "):]
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i].split("\n"),
                    "response": response
                }
                output_sample_batch.append(output_sample)

        elif args.method == "genread":
            passage_prompt_batch = [get_prompt_genread(question) for question in question_batch]
            if args.model_name == "gpt-neoxt-chat":
                passage_batch = model.get_response_batch(passage_prompt_batch, max_tokens=args.passage_tokens, use_temp=False)
            else:
                passage_batch = model.get_response_batch(passage_prompt_batch, max_tokens=args.passage_tokens)
            answer_prompt_batch = [get_prompt_genread(question, clean_passage(passage)) for question, passage in zip(question_batch, passage_batch)]
            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(answer_prompt_batch, max_tokens=args.answer_tokens, use_temp=False)
            else:
                response_batch = model.get_response_batch(answer_prompt_batch, max_tokens=args.answer_tokens)
            for i in range(len(response_batch)):
                response = response_batch[i]
                raw_pred = response
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "passage": passage_batch[i],
                    "prompt": answer_prompt_batch[i],
                    "response": response
                }
                output_sample_batch.append(output_sample)

        elif args.method == "auto-cot":
            prompt_batch = []
            for i in range(len(question_batch)):
                demos = [{"question": demo["question"][i], "cot": demo["cot"][i], "answer": demo["answer"][i]} for demo in batch["demos"]]
                prompt_batch.append(build_prompt_auto_cot(question_batch[i], demos))

            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    use_temp=False,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            elif args.model_name in ["alpaca-13b", "falcon-7b", "vicuna-13b", "wizard-13b"]:
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            else:
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.cot_tokens)

            for i in range(len(response_batch)):
                response = response_batch[i]
                if "the answer is: " in response.lower():
                    raw_pred = response[response.lower().find("the answer is: ") + len("the answer is: "):]
                else:
                    raw_sentences = response.split("\n")
                    answer_sentences = []
                    for raw_sentence in raw_sentences:
                        if "answer: " in raw_sentence.lower():
                            answer_sentences.append(raw_sentence)
                    if len(answer_sentences) == 0:
                        raw_pred = ""
                    else:
                        final_answer_sentence = answer_sentences[-1]
                        raw_pred = final_answer_sentence[final_answer_sentence.lower().find("answer: ") + len("answer: "):]
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i],
                    "response": response
                }
                output_sample_batch.append(output_sample)

        elif args.method == "manual-cot":
            prompt_batch = []
            for i in range(len(question_batch)):
                prompt_batch.append(build_prompt_manual_cot(question_batch[i], manual_cot))
            if args.model_name == "gpt-neoxt-chat":
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    use_temp=False,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            elif args.model_name in ["alpaca-13b", "falcon-7b", "vicuna-13b", "wizard-13b"]:
                response_batch = model.get_response_batch(
                    prompt_batch,
                    max_tokens=args.cot_tokens,
                    replace_prompt=False,
                    demo_idx=args.num_demos
                )
            else:
                response_batch = model.get_response_batch(prompt_batch, max_tokens=args.cot_tokens)

            for i in range(len(response_batch)):
                response = response_batch[i]
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
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i].split("\n"),
                    "response": response_batch[i].split("\n")
                }
                output_sample_batch.append(output_sample)

        elif args.method == "baseline":
            if args.model_name.lower() in ["dpr", "realm"]:
                prompt_batch = question_batch
            else:
                prompt_batch = [get_prompt_baseline(question) for question in question_batch]
            response_batch = model.get_response_batch(prompt_batch, max_tokens=args.answer_tokens)
            for i in range(len(response_batch)):
                response = response_batch[i]
                raw_pred = response
                pred_answers = extract_answer(raw_pred)
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "pred_ans": pred_answers,
                    "prompt": prompt_batch[i],
                    "response": response
                }
                output_sample_batch.append(output_sample)


        elif args.method == "retrieval_only":
            if args.model_name.lower() in ["dpr", "realm"]:
                prompt_batch = question_batch
            else:
                prompt_batch = [get_prompt_baseline(question) for question in question_batch]
            response_batch = model.retrieve_docs_batch(prompt_batch, max_tokens=args.answer_tokens)
            for i in range(len(response_batch)):
                response = response_batch[i]
                output_sample = {
                    "question": question_batch[i],
                    "answer": answer_batch[i],
                    "contexts": response
                }
                output_sample_batch.append(output_sample)



        else:
            raise NotImplementedError(f"Method {args.method} is not implemented.")

        # evaluate
        if args.do_eval and args.method != "retrieval_only":
            for i in range(len(output_sample_batch)):
                gold_answers = output_sample_batch[i]["answer"]
                pred_answers = output_sample_batch[i]["pred_ans"]
            # cweb-qa has multiple gold answers
                max_em, max_f1 = qa_evaluate(gold_answers, pred_answers)
                output_sample_batch[i]["em"] = max_em
                output_sample_batch[i]["f1"] = max_f1

                total_em += max_em
                total_f1 += max_f1

            if (batch_idx+1) % 40 == 0:
                print(f"[{print_now(1)}] Batch: {batch_idx}, EM: {total_em / total}, F1: {total_f1 / total}")

        # write to file
        for output_sample in output_sample_batch:
            output_file.write(json.dumps(output_sample) + "\n")
        output_dataset.extend(output_sample_batch)
        # limit dataset size
        if 0 < args.limit_dataset_size <= total:
            break

    if args.do_eval and args.method != "retrieval_only":
        print(f"[{print_now(1)}] Total {total}, Overall EM: {total_em / total}, F1: {total_f1 / total}")
    else:
        print(f"[{print_now(1)}] Finished inference on {total} samples.")

    # transfer output file to json format
    output_file.close()

    with open(output_jsonl_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(output_dataset, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    print(f"[{print_now(1)}] Task Done!")








