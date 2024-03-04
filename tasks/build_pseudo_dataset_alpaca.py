import json
import os
import random
import re
from string import punctuation
from copy import deepcopy
from tqdm import tqdm
import pickle
from src.utils import *
from src.models import LocalModel, Alpaca


class QuestionHop(object):
    def __init__(self, term, qae_pair):
        self.term = term
        self.context = qae_pair["context"] if "context" in qae_pair else ""
        self.question = qae_pair["question"]
        self.answer = qae_pair["answer"]
        self.norm_term = normalize_answer(term)
        self.norm_answer = normalize_answer(qae_pair["answer"])
        self.norm_question = normalize_answer(qae_pair["question"])
        self.evidence = qae_pair["explanation"]
        self.next_hop = None
        self.prev_hop_1 = None
        self.prev_hop_2 = None

    def set_next_hop(self, next_hop):
        self.next_hop = next_hop

    def set_prev_hop_1(self, prev_hop):
        self.prev_hop_1 = prev_hop

    def set_prev_hop_2(self, prev_hop):
        self.prev_hop_2 = prev_hop


def load_examples(data_dir):
    first_hop_examples = []
    second_hop_examples = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r") as f:
                dataset = json.load(f)
                for one_hop in dataset["first_hops"]:
                    example = dataset["first_hops"][one_hop]

                    first_hop_examples.append(example)
                for two_hop in dataset["second_hops"]:
                    terms = dataset["second_hops"][two_hop]["hops"]
                    if normalize_answer(terms[0]) in normalize_answer(terms[1]):
                        print(f"Skipping {terms[0]} because it is a subset of {terms[1]}")
                    elif normalize_answer(terms[1]) in normalize_answer(terms[0]):
                        print(f"Skipping {terms[1]} because it is a subset of {terms[0]}")
                    else:
                        example = dataset["second_hops"][two_hop]
                        second_hop_examples.append(example)

    print(f"Number of one-hop examples: {len(first_hop_examples)}")
    print(f"Number of two-hop examples: {len(second_hop_examples)}")
    return first_hop_examples, second_hop_examples


def build_multihop_datasets(first_hop_examples, second_hop_examples):
    type_1_dataset = []
    type_2_dataset = []
    type_3_dataset = []
    type_4_dataset = []
    type_5_dataset = []
    type_6_dataset = []

    existing_question_pairs = {}
    for second_hop_term in tqdm(second_hop_examples, desc="Constructing 2-hop questions"):
        for term_2_qae_pair in second_hop_term["qae_pairs"]:

            # invalid named entities extracted by spacy
            if normalize_answer(term_2_qae_pair["answer"]) in ["days", "weeks", "months", "years", "decades",
                                                               "centuries"]:
                continue
            if normalize_answer(term_2_qae_pair["answer"]) in normalize_answer(term_2_qae_pair["question"]):
                continue

            term_2_qae_pair["question"] = term_2_qae_pair["question"].split("\n")[0].replace("Question:", "").strip()

            hop_2 = QuestionHop(
                term=second_hop_term["term"],
                qae_pair=term_2_qae_pair,
            )
            if hop_2.norm_question not in existing_question_pairs:
                existing_question_pairs[hop_2.norm_question] = []

            # find the default first hop
            for first_hop_term in first_hop_examples:
                if first_hop_term["term"] == second_hop_term["hops"][0]:
                    for term_1_qae_pair in first_hop_term["qae_pairs"]:
                        # invalid named entities extracted by spacy
                        if normalize_answer(term_1_qae_pair["answer"]) in ["days", "weeks", "months", "years",
                                                                           "decades", "centuries"]:
                            continue

                        term_1_qae_pair["question"] = term_1_qae_pair["question"].split("\n")[0].replace("Question:",
                                                                                                         "").strip()
                        question = term_1_qae_pair["question"]
                        answer = term_1_qae_pair["answer"]
                        if normalize_answer(answer) in normalize_answer(question):
                            continue
                        if remove_punc(answer).lower() in remove_punc(hop_2.question).lower():
                            hop_1 = QuestionHop(
                                term=first_hop_term["term"],
                                qae_pair=term_1_qae_pair,
                            )
                            if hop_1.norm_question in existing_question_pairs[hop_2.norm_question]:
                                continue
                            else:
                                existing_question_pairs[hop_2.norm_question].append(hop_1.norm_question)

                            temp_hop_2 = deepcopy(hop_2)
                            hop_1.set_next_hop(temp_hop_2)
                            temp_hop_2.set_prev_hop_1(hop_1)
                            type_1_dataset.append(deepcopy(temp_hop_2))

    last_hop_1_prev_question = ""
    for i in tqdm(range(len(type_1_dataset)), desc="Constructing type 2,3,4,6 datasets"):
        hop_1 = type_1_dataset[i]
        if last_hop_1_prev_question != hop_1.prev_hop_1.question:
            last_hop_1_prev_question = hop_1.prev_hop_1.question
            new_hop_1_prev_flag = True
        else:
            new_hop_1_prev_flag = False

        for j in range(len(type_1_dataset)):
            if i == j:
                continue
            hop_2 = type_1_dataset[j]

            if new_hop_1_prev_flag:
                # Case 1: hop_1.prev -> (hop_2.prev -> hop_2)
                hop_1_prev = deepcopy(hop_1.prev_hop_1)
                hop_2_prev = deepcopy(hop_2.prev_hop_1)
                if include_without_punc(hop_1_prev.answer, hop_2_prev.question):
                    case_1_flag = True
                    if hop_1_prev.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                        case_1_flag = False
                    # avoid cyclic graph
                    if hop_2.norm_answer in hop_2_prev.norm_question:
                        case_1_flag = False
                    if hop_2.norm_answer in hop_1_prev.norm_question:
                        case_1_flag = False
                    if hop_2_prev.norm_answer in hop_1_prev.norm_question:
                        case_1_flag = False
                    if hop_1_prev.norm_answer in hop_2.norm_question:
                        case_1_flag = False

                    if case_1_flag:
                        hop_1_prev.set_next_hop(hop_2_prev)
                        hop_2_prev.set_prev_hop_1(hop_1_prev)
                        temp_hop_2 = deepcopy(hop_2)
                        hop_2_prev.set_next_hop(temp_hop_2)
                        temp_hop_2.set_prev_hop_1(hop_2_prev)
                        type_2_dataset.append(deepcopy(temp_hop_2))

                # Case 2: hop_1.prev -> (hop_2 <- hop_2.prev)
                if include_without_punc(hop_1_prev.answer, hop_2.question):
                    hop_1_prev = deepcopy(hop_1.prev_hop_1)
                    hop_2_prev = deepcopy(hop_2.prev_hop_1)
                    case_2_flag = True
                    if hop_1_prev.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                        case_2_flag = False
                    # final answer cannot be in the previous questions
                    if hop_2.norm_answer in hop_2_prev.norm_question:
                        case_2_flag = False
                    if hop_2.norm_answer in hop_1_prev.norm_question:
                        case_2_flag = False
                    # avoid cyclic graph
                    if hop_2_prev.norm_answer in hop_1_prev.norm_question:
                        case_2_flag = False
                    if hop_1_prev.norm_answer in hop_2_prev.norm_question:
                        case_2_flag = False
                    # two previous hops answer cannot be the same
                    if hop_1_prev.norm_answer in hop_2_prev.norm_answer:
                        case_2_flag = False
                    if hop_2_prev.norm_answer in hop_1_prev.norm_answer:
                        case_2_flag = False
                    if case_2_flag:
                        temp_hop_2 = deepcopy(hop_2)
                        temp_hop_2.set_prev_hop_2(hop_1_prev)
                        hop_1_prev.set_next_hop(temp_hop_2)
                        type_3_dataset.append(deepcopy(temp_hop_2))

            # Case 3: (hop_1.prev -> hop_1) -> (hop_2.prev -> hop_2)
            hop_1_prev = deepcopy(hop_1.prev_hop_1)
            hop_2_prev = deepcopy(hop_2.prev_hop_1)
            if include_without_punc(hop_1.answer, hop_2_prev.question):
                case_3_flag = True
                if hop_1_prev.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                    case_3_flag = False
                if hop_1.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                    case_3_flag = False
                # avoid cyclic graph
                if hop_2.norm_answer in hop_2_prev.norm_question:
                    case_3_flag = False
                if hop_2.norm_answer in hop_1_prev.norm_question:
                    case_3_flag = False
                if hop_2.norm_answer in hop_1.norm_question:
                    case_3_flag = False
                if hop_2_prev.norm_answer in hop_1_prev.norm_question:
                    case_3_flag = False
                if hop_2_prev.norm_answer in hop_1.norm_question:
                    case_3_flag = False
                if hop_1.norm_answer in hop_2.norm_question:
                    case_3_flag = False
                if hop_1.norm_answer in hop_1_prev.norm_question:
                    case_3_flag = False
                if hop_1_prev.norm_answer in hop_2_prev.norm_question:
                    case_3_flag = False
                if hop_1_prev.norm_answer in hop_2.norm_question:
                    case_3_flag = False

                if case_3_flag:
                    temp_hop_1 = deepcopy(hop_1)
                    temp_hop_1.set_next_hop(hop_2_prev)
                    hop_2_prev.set_prev_hop_1(temp_hop_1)
                    type_2_dataset.append(deepcopy(hop_2_prev))
                    temp_hop_2 = deepcopy(hop_2)
                    hop_2_prev.set_next_hop(temp_hop_2)
                    temp_hop_2.set_prev_hop_1(hop_2_prev)
                    type_4_dataset.append(deepcopy(temp_hop_2))
                    temp_hop_1.prev_hop_1 = None
                    hop_2_prev.set_prev_hop_1(temp_hop_1)
                    temp_hop_2.set_prev_hop_1(hop_2_prev)
                    type_2_dataset.append(deepcopy(temp_hop_2))

            # Case 4: (hop_1.prev -> hop_1) -> (hop_2 <- hop_2.prev)
            hop_1_prev = deepcopy(hop_1.prev_hop_1)
            hop_2_prev = deepcopy(hop_2.prev_hop_1)
            if include_without_punc(hop_1.answer, hop_2.question):
                case_4_flag = True
                if hop_1_prev.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                    case_4_flag = False
                if hop_1.norm_term in [hop_2_prev.norm_term, hop_2.norm_term]:
                    case_4_flag = False
                # avoid cyclic graph
                if hop_2.norm_answer in hop_2_prev.norm_question:
                    case_4_flag = False
                if hop_2.norm_answer in hop_1_prev.norm_question:
                    case_4_flag = False
                if hop_2.norm_answer in hop_1.norm_question:
                    case_4_flag = False
                if hop_2_prev.norm_answer in hop_1_prev.norm_question:
                    case_4_flag = False
                if hop_2_prev.norm_answer in hop_1.norm_question:
                    case_4_flag = False
                if hop_1.norm_answer in hop_2_prev.norm_question:
                    case_4_flag = False
                if hop_1.norm_answer in hop_2_prev.norm_answer:
                    case_4_flag = False
                if hop_2_prev.norm_answer in hop_1.norm_answer:
                    case_4_flag = False
                if hop_1.norm_answer in hop_1_prev.norm_question:
                    case_4_flag = False
                if hop_1_prev.norm_answer in hop_2_prev.norm_question:
                    case_4_flag = False
                if hop_1_prev.norm_answer in hop_2.norm_question:
                    case_4_flag = False

                if case_4_flag:
                    temp_hop_2 = deepcopy(hop_2)
                    temp_hop_1 = deepcopy(hop_1)
                    temp_hop_1.set_next_hop(temp_hop_2)
                    temp_hop_2.set_prev_hop_2(temp_hop_1)
                    type_6_dataset.append(deepcopy(temp_hop_2))
                    temp_hop_2.set_prev_hop_1(temp_hop_1)
                    temp_hop_2.prev_hop_2 = None
                    type_2_dataset.append(deepcopy(temp_hop_2))

    for type_3_sample in tqdm(type_3_dataset, desc="Constructing type 5 dataset"):
        hop_3 = deepcopy(type_3_sample)
        existing_hop_1_prev_questions = []
        existing_hop_1_questions = []
        for type_1_sample in type_1_dataset:
            hop_1 = deepcopy(type_1_sample)
            # Case 5 (hop_3.prev_1 | hop_3.perv_2 -> hop_3) -> hop_1.prev
            if include_without_punc(hop_3.answer, hop_1.prev_hop_1.question):
                case_5_flag = True
                if type_1_sample.prev_hop_1.norm_question in existing_hop_1_prev_questions:
                    case_5_flag = False
                if hop_1.prev_hop_1.norm_term in [hop_3.norm_term, hop_3.prev_hop_1.norm_term,
                                                  hop_3.prev_hop_2.norm_term]:
                    case_5_flag = False
                if hop_1.prev_hop_1.norm_answer in hop_3.norm_question:
                    case_5_flag = False
                if hop_1.prev_hop_1.norm_answer in hop_3.prev_hop_1.norm_question:
                    case_5_flag = False
                if hop_1.prev_hop_1.norm_answer in hop_3.prev_hop_2.norm_question:
                    case_5_flag = False
                if hop_3.prev_hop_1.norm_answer in hop_1.prev_hop_1.norm_question:
                    case_5_flag = False
                if hop_3.prev_hop_2.norm_answer in hop_1.prev_hop_1.norm_question:
                    case_5_flag = False

                if case_5_flag:
                    existing_hop_1_prev_questions.append(hop_1.prev_hop_1.norm_question)
                    temp_hop_3 = deepcopy(hop_3)
                    temp_hop_1 = deepcopy(hop_1)
                    temp_hop_3.set_next_hop(hop_1.prev_hop_1)
                    temp_hop_1.prev_hop_1.prev_hop_1 = deepcopy(temp_hop_3)
                    temp_hop_1.prev_hop_1.next_hop = None
                    type_5_dataset.append(deepcopy(temp_hop_1.prev_hop_1))

            # Case 6: (hop_3.prev_1 | hop_3.perv_2 -> hop_3) -> hop_1
            if include_without_punc(hop_3.answer, hop_1.question):
                case_6_flag = True
                if type_1_sample.norm_question in existing_hop_1_questions:
                    case_6_flag = False
                if hop_1.norm_term in [hop_3.norm_term, hop_3.prev_hop_1.norm_term, hop_3.prev_hop_2.norm_term]:
                    case_6_flag = False
                if hop_1.norm_answer in hop_3.norm_question:
                    case_6_flag = False
                if hop_1.norm_answer in hop_3.prev_hop_1.norm_question:
                    case_6_flag = False
                if hop_1.norm_answer in hop_3.prev_hop_2.norm_question:
                    case_6_flag = False
                if hop_3.prev_hop_1.norm_answer in hop_1.norm_question:
                    case_6_flag = False
                if hop_3.prev_hop_2.norm_answer in hop_1.norm_question:
                    case_6_flag = False

                if case_6_flag:
                    existing_hop_1_questions.append(hop_1.norm_question)
                    temp_hop_3 = deepcopy(hop_3)
                    temp_hop_1 = deepcopy(hop_1)
                    temp_hop_3.set_next_hop(temp_hop_1)
                    temp_hop_1.prev_hop_1 = deepcopy(temp_hop_3)
                    type_5_dataset.append(deepcopy(temp_hop_1))

    all_dataset = {
        "type_1": type_1_dataset,
        "type_2": type_2_dataset,
        "type_3": type_3_dataset,
        "type_4": type_4_dataset,
        "type_5": type_5_dataset,
        "type_6": type_6_dataset
    }

    return all_dataset


def filter_dataset(dataset, dataset_type: str, filter_type: str, max_duplicate_degree: int = 0):
    filtered_dataset = []
    existing_pairs = []
    assert filter_type in ["question", "answer"]
    for data in tqdm(dataset, desc="Filtering {} dataset".format(dataset_type)):
        third_question = ""
        third_answer = ""
        fourth_answer = ""
        fourth_question = ""
        if dataset_type == "type_1":
            first_question: str = data.prev_hop_1.norm_question
            second_question: str = data.norm_question
            first_answer: str = data.prev_hop_1.norm_answer
            second_answer: str = data.norm_answer
        elif dataset_type == "type_2":
            first_question: str = data.prev_hop_1.prev_hop_1.question
            second_question: str = data.prev_hop_1.question
            third_question: str = data.question
            first_answer: str = data.prev_hop_1.prev_hop_1.answer
            second_answer: str = data.prev_hop_1.answer
            third_answer: str = data.answer
        elif dataset_type == "type_3":
            first_question: str = data.prev_hop_1.norm_question
            second_question: str = data.prev_hop_2.norm_question
            third_question: str = data.norm_question
            first_answer: str = data.prev_hop_1.norm_answer
            second_answer: str = data.prev_hop_2.norm_answer
            third_answer: str = data.norm_answer
            question_validate = third_question.replace(first_answer, "")
            if second_answer not in question_validate:
                continue
        elif dataset_type == "type_4":
            first_question: str = data.prev_hop_1.prev_hop_1.prev_hop_1.norm_question
            second_question: str = data.prev_hop_1.prev_hop_1.norm_question
            third_question: str = data.prev_hop_1.norm_question
            fourth_question: str = data.norm_question
            first_answer: str = data.prev_hop_1.prev_hop_1.prev_hop_1.norm_answer
            second_answer: str = data.prev_hop_1.prev_hop_1.norm_answer
            third_answer: str = data.prev_hop_1.norm_answer
            fourth_answer: str = data.norm_answer
        elif dataset_type == "type_5":
            first_question: str = data.prev_hop_1.prev_hop_1.norm_question
            second_question: str = data.prev_hop_1.prev_hop_2.norm_question
            third_question: str = data.prev_hop_1.norm_question
            fourth_question: str = data.norm_question
            first_answer: str = data.prev_hop_1.prev_hop_1.norm_answer
            second_answer: str = data.prev_hop_1.prev_hop_2.norm_answer
            third_answer: str = data.prev_hop_1.norm_answer
            fourth_answer: str = data.norm_answer
            question_validate = third_question.replace(first_answer, "")
            if second_answer not in question_validate:
                continue
        elif dataset_type == "type_6":
            first_question: str = data.prev_hop_2.prev_hop_1.norm_question
            second_question: str = data.prev_hop_2.norm_question
            third_question: str = data.prev_hop_1.norm_question
            fourth_question: str = data.norm_question
            first_answer: str = data.prev_hop_2.prev_hop_1.norm_answer
            second_answer: str = data.prev_hop_2.norm_answer
            third_answer: str = data.prev_hop_1.norm_answer
            fourth_answer: str = data.norm_answer
            question_validate = fourth_question.replace(second_answer, "")
            if third_answer not in question_validate:
                continue
        else:
            first_question = data.question
            first_answer = data.answer

        filter_items = []
        if filter_type == "question":
            if dataset_type == "type_0":
                filter_items = [first_question]
            elif dataset_type == "type_1":
                filter_items = [first_question, second_question]
            elif dataset_type in ["type_2", "type_3"]:
                filter_items = [first_question, second_question, third_question]
            else:
                filter_items = [first_question, second_question, third_question, fourth_question]
        elif filter_type == "answer":
            if dataset_type == "type_0":
                filter_items = [first_answer]
            elif dataset_type == "type_1":
                filter_items = [first_answer, second_answer]
            elif dataset_type in ["type_2", "type_3"]:
                filter_items = [first_answer, second_answer, third_answer]
            else:
                filter_items = [first_answer, second_answer, third_answer, fourth_answer]

        duplicate_degree = 0
        for existing_pair in existing_pairs:
            current_degree = 0
            for filter_item in filter_items:
                if filter_item in existing_pair:
                    current_degree += 1
            if current_degree > duplicate_degree:
                duplicate_degree = current_degree

        if duplicate_degree <= max_duplicate_degree:
            filtered_dataset.append(data)
            existing_pairs.append(filter_items)

    print(f"Length of original dataset: {len(dataset)}")
    print(f"Filtered {len(dataset) - len(filtered_dataset)} out of {len(dataset)} examples")
    print(f"Length of filtered dataset: {len(filtered_dataset)}")

    return filtered_dataset


def export_to_dict(dataset, dataset_type: str):
    dict_dataset = []
    for data in dataset:
        hop_1, hop_2, hop_3, hop_4 = None, None, None, None
        if dataset_type == "type_1":
            hop_1 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_2 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        elif dataset_type == "type_2":
            hop_1 = {
                "context": data.prev_hop_1.prev_hop_1.context,
                "question": data.prev_hop_1.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.prev_hop_1.answer),
                "evidence": data.prev_hop_1.prev_hop_1.evidence,
            }
            hop_2 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_3 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        elif dataset_type == "type_3":
            hop_1 = {
                "context": data.prev_hop_2.context,
                "question": data.prev_hop_2.question,
                "answer": remove_punc(data.prev_hop_2.answer),
                "evidence": data.prev_hop_2.evidence,
            }
            hop_2 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_3 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        elif dataset_type == "type_4":
            hop_1 = {
                "context": data.prev_hop_1.prev_hop_1.prev_hop_1.context,
                "question": data.prev_hop_1.prev_hop_1.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.prev_hop_1.prev_hop_1.answer),
                "evidence": data.prev_hop_1.prev_hop_1.prev_hop_1.evidence,
            }
            hop_2 = {
                "context": data.prev_hop_1.prev_hop_1.context,
                "question": data.prev_hop_1.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.prev_hop_1.answer),
                "evidence": data.prev_hop_1.prev_hop_1.evidence,
            }
            hop_3 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_4 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        elif dataset_type == "type_5":
            hop_1 = {
                "context": data.prev_hop_1.prev_hop_1.context,
                "question": data.prev_hop_1.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.prev_hop_1.answer),
                "evidence": data.prev_hop_1.prev_hop_1.evidence,
            }
            hop_2 = {
                "context": data.prev_hop_1.prev_hop_2.context,
                "question": data.prev_hop_1.prev_hop_2.question,
                "answer": remove_punc(data.prev_hop_1.prev_hop_2.answer),
                "evidence": data.prev_hop_1.prev_hop_2.evidence,
            }
            hop_3 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_4 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        elif dataset_type == "type_6":
            hop_1 = {
                "context": data.prev_hop_2.prev_hop_1.context,
                "question": data.prev_hop_2.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_2.prev_hop_1.answer),
                "evidence": data.prev_hop_2.prev_hop_1.evidence,
            }
            hop_2 = {
                "context": data.prev_hop_2.context,
                "question": data.prev_hop_2.question,
                "answer": remove_punc(data.prev_hop_2.answer),
                "evidence": data.prev_hop_2.evidence,
            }
            hop_3 = {
                "context": data.prev_hop_1.context,
                "question": data.prev_hop_1.question,
                "answer": remove_punc(data.prev_hop_1.answer),
                "evidence": data.prev_hop_1.evidence,
            }
            hop_4 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }
        else:
            hop_1 = {
                "context": data.context,
                "question": data.question,
                "answer": remove_punc(data.answer),
                "evidence": data.evidence,
            }

        if hop_4:
            answer = hop_4["answer"]
            hops = [hop_1, hop_2, hop_3, hop_4]
        elif hop_3:
            answer = hop_3["answer"]
            hops = [hop_1, hop_2, hop_3]
        elif hop_2:
            answer = hop_2["answer"]
            hops = [hop_1, hop_2]
        else:
            answer = hop_1["answer"]
            hops = [hop_1]

        dict_dataset.append({
            "question": "",
            "answer": answer,
            "hop_type": dataset_type,
            "hops": hops,
        })
    return dict_dataset


def build_demo_question_set(question_set: List[str], hop_type:str):
    if hop_type == "type_1":
        question_set[1] = question_set[1].replace("#1", f"[{question_set[0]}]")
    elif hop_type == "type_2":
        for i in range(2):
            question_set[i+1] = question_set[i+1].replace(f"#{i+1}", f"[{question_set[i]}]")
    elif hop_type == "type_3":
        for i in range(2):
            question_set[2] = question_set[2].replace(f"#{i+1}", f"[{question_set[i]}]")
    elif hop_type == "type_4":
        for i in range(3):
            question_set[i+1] = question_set[i+1].replace(f"#{i+1}", f"[{question_set[i]}]")
    elif hop_type == "type_5":
        for i in range(2):
            question_set[2] = question_set[2].replace(f"#{i+1}", f"[{question_set[i]}]")
        question_set[3] = question_set[3].replace("#3", f"[{question_set[2]}]")
    elif hop_type == "type_6":
        question_set[1] = question_set[1].replace("#1", f"[{question_set[0]}]")
        for i in [2, 3]:
            question_set[3] = question_set[3].replace(f"#{i}", f"[{question_set[i-1]}]")
    else:
        raise ValueError("Invalid hop type!")

    return question_set


def include_then_replace(question, answer, replace):
    if f" {answer} " in question:
        question = question.replace(f" {answer} ", f" [{replace}] ")
    elif f" {answer}," in question:
        question = question.replace(f" {answer},", f" [{replace}],")
    elif f" {answer}?" in question:
        question = question.replace(f" {answer}?", f" [{replace}]?")
    else:
        question = ""
    return question


def prepare_raw_question(questions: List[str], answers: List[str], hop_type: str):
    raw_question = questions[-1]
    if hop_type == "type_1":
        raw_question = include_then_replace(raw_question, answers[0], questions[0])
    elif hop_type == "type_2":
        for i in range(2):
            questions[i+1] = include_then_replace(questions[i+1], answers[i], questions[i])
            raw_question = questions[-1]
    elif hop_type == "type_3":
        for i in range(2):
            raw_question = include_then_replace(raw_question, answers[i], questions[i])
    elif hop_type == "type_4":
        for i in range(3):
            questions[i+1] = include_then_replace(questions[i+1], answers[i], questions[i])
        raw_question = questions[-1]
    elif hop_type == "type_5":
        raw_question = questions[2]
        for i in range(2):
            raw_question = include_then_replace(raw_question, answers[i], questions[i])
        raw_question = include_then_replace(questions[3], answers[2], raw_question)
    elif hop_type == "type_6":
        questions[1] = include_then_replace(questions[1], answers[0], questions[0])
        for i in [1, 2]:
            raw_question = include_then_replace(raw_question, answers[i], questions[i])
    else:
        raise ValueError("Invalid hop type!")

    return raw_question


def gen_question_with_demos(
        dataset: List[Dict],
        demo_set: List[Dict],
        hop_type: str,
        model: LocalModel,
        num_demos: int = 4,
        max_retry: int = 1
):
    output_dataset = []

    random.seed(42)
    for data in tqdm(dataset, desc=f"Generating {hop_type} questions..."):
        prompt = ""
        demos = random.sample(demo_set, k=min(num_demos, len(demo_set)))
        for demo in demos:
            question_set = [h["question"] for h in demo["hops"]]
            question_set = build_demo_question_set(question_set, hop_type)
            raw_question = question_set[-1].replace(" ?", "?")
            if not raw_question.endswith("?"):
                raw_question += "?"
            prompt += f"Raw question: {raw_question}\n" \
                      f"Replace the sentence within [] with a relative clause and make the raw question into a natural question: {demo['question']}\n\n"

        answers = [h["answer"] for h in data["hops"]]
        questions = [h["question"] for h in data["hops"]]

        raw_question = prepare_raw_question(questions, answers, hop_type)
        if raw_question == "":
            continue
        raw_question = raw_question.replace(" ?", "?")
        if not raw_question.endswith("?"):
            raw_question += "?"
        prompt += f"Raw question: {raw_question}\n" \
                  f"Replace the sentence within [] with a relative clause and make the raw question into a natural question:"

        pass_flag = False
        num_retry = 0
        response = ""
        while not pass_flag:
            response = model.get_response(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1 + num_retry*0.1,
                plain_text=True,
            )
            if "\n" in response:
                response = response.split("\n")[0]

            response = response.replace("Question:", "").strip()
            response = response.replace(", according to the passage", "").strip()

            pass_flag = True
            num_retry += 1

            for answer in answers:
                if f" {answer} ".lower() in response.lower():
                    pass_flag = False
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    break
                elif f" {answer},".lower() in response.lower():
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    pass_flag = False
                    break
                elif f" {answer}?".lower() in response.lower():
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    pass_flag = False
                    break

            if "[" in response or "]" in response:
                print(f"[{print_now(1)}] Warning: response contains brackets")
                pass_flag = False

            for hop in data["hops"]:
                if hop["question"] in response:
                    print(f"[{print_now(1)}] Warning: response contains hop question")
                    pass_flag = False
                    break

            if num_retry > max_retry:
                print(f"[{print_now(1)}] Warning: max retry reached")
                pass_flag = False
                break

        if pass_flag:
            output_dataset.append({
                "question": response,
                "answer": data["answer"],
                "hop_type": data["hop_type"],
                "hops": data["hops"]
            })

    return output_dataset


def generate_yes_no_questions(
        dataset: List[Dict],
        demo_set: List[Dict],
        yesno_demos: List[Dict],
        hop_type: str,
        model: LocalModel,
        num_demos: int = 4,
        max_retry: int = 1
):
    output_dataset = []
    random.seed(42)
    for data in tqdm(dataset, desc=f"Generating {hop_type} yes no questions..."):
        prompt = ""
        yesno_demo = random.sample(yesno_demos, k=4)
        for demo in yesno_demo:
            raw_question = demo["question"].replace(" ?", "?")
            if not raw_question.endswith("?"):
                raw_question += "?"
            prompt += f"Question: {raw_question}\nAnswer: {demo['answer']}\n"
            raw_question = demo['reformulated_question'].replace(" ?", "?")
            if not raw_question.endswith("?"):
                raw_question += "?"
            prompt += f"Reform the question to a general interrogative sentence that can be answered with {demo['reformulated_answer']}: {demo['reformulated_question']}\n\n"

        last_hop_question = data["hops"][-1]["question"]
        last_hop_answer = data["hops"][-1]["answer"]
        last_hop_question = last_hop_question.replace(" ?", "?")
        if not last_hop_question.endswith("?"):
            last_hop_question += "?"
        prompt += f"Question: {last_hop_question}\nAnswer: {last_hop_answer}\n"
        prompt += f"Reform the question to a general interrogative sentence that can be answered with {yesno_demo[-1]['reformulated_answer']}:"
        yesno_question = model.get_response(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1,
                plain_text=True,
            )

        yesno_question = yesno_question.split("\n")[0].replace("Question:", "").strip()

        prompt = ""
        demos = random.sample(demo_set, k=min(num_demos, len(demo_set)))
        for demo in demos:
            question_set = [h["question"] for h in demo["hops"]]
            question_set = build_demo_question_set(question_set, hop_type)
            raw_question = question_set[-1].replace(" ?", "?")
            if not raw_question.endswith("?"):
                raw_question += "?"
            target_question = demo["question"].replace(" ?", "?")
            if not target_question.endswith("?"):
                target_question += "?"
            prompt += f"Raw question: {raw_question}\n" \
                      f"Replace the sentence within [] with a relative clause and make the raw question into a natural question: {demo['question']}\n\n"

        answers = [h["answer"] for h in data["hops"]]
        questions = [h["question"] for h in data["hops"]]
        answers[-1] = yesno_demo[-1]["reformulated_answer"]
        questions[-1] = yesno_question
        data["hops"][-1]["question"] = yesno_question
        data["hops"][-1]["answer"] = yesno_demo[-1]["reformulated_answer"]

        raw_question = prepare_raw_question(questions, answers, hop_type)
        if raw_question == "":
            continue
        raw_question = raw_question.replace(" ?", "?")
        if not raw_question.endswith("?"):
            raw_question += "?"
        prompt += f"Raw question: {raw_question}\n" \
                  f"Replace the sentence within [] with a relative clause and make the raw question into a natural question:"

        pass_flag = False
        num_retry = 0
        response = ""
        while not pass_flag:
            response = model.get_response(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1 + num_retry*0.1,
                plain_text=True,
            )
            pass_flag = True
            num_retry += 1

            if "\n" in response:
                response = response.split("\n")[0]
            response = response.replace("Question:", "").strip()
            response = response.replace(", according to the passage", "").strip()

            for answer in answers:
                if f" {answer} ".lower() in response.lower():
                    pass_flag = False
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    break
                elif f" {answer},".lower() in response.lower():
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    pass_flag = False
                    break
                elif f" {answer}?".lower() in response.lower():
                    print(f"[{print_now(1)}] Warning: response contains answer")
                    pass_flag = False
                    break

            if "[" in response or "]" in response:
                print(f"[{print_now(1)}] Warning: response contains brackets")
                pass_flag = False

            for hop in data["hops"]:
                if hop["question"] in response:
                    print(f"[{print_now(1)}] Warning: response contains hop question")
                    pass_flag = False
                    break

            if num_retry > max_retry:
                print(f"[{print_now(1)}] Warning: max retry reached")
                pass_flag = False
                break

        if pass_flag:
            output_dataset.append({
                "question": response,
                "answer": answers[-1],
                "hop_type": data["hop_type"],
                "hops": data["hops"]
            })

    return output_dataset


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../data/self-prompt-cot/gpt-neoxt-chat/",
                        help="Directory of generated data")
    parser.add_argument("--output_dir", type=str, default="../data/self-prompt-cot/",
                        help="Directory of output dataset")
    parser.add_argument("--demo_path", type=str, default="../demos/multihop_demos.json", help="Path of demo dataset")
    parser.add_argument("--yesno_demo_path", type=str, default="../demos/yes_no_demos.json", help="Path of demo dataset")
    parser.add_argument("--output_composition_path", type=str, default=None)
    parser.add_argument("--output_filtered_path", type=str, default=None)
    parser.add_argument("--output_completion_path", type=str, default=None)
    parser.add_argument("--output_yesno_path", type=str, default=None)
    parser.add_argument("--output_merge_path", type=str, default=None)
    parser.add_argument("--do_yesno", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")

    args = parser.parse_args()

    if args.output_composition_path is None:
        args.output_composition_path = os.path.join(args.output_dir, "all_datasets_alpaca.pkl")

    if args.output_filtered_path is None:
        args.output_filtered_path = os.path.join(args.output_dir, "pseudo_dataset_raw_alpaca.json")

    if args.output_completion_path is None:
        args.output_completion_path = os.path.join(args.output_dir, "pseudo_dataset_complete_alpaca.json")

    if args.output_yesno_path is None:
        args.output_yesno_path = os.path.join(args.output_dir, "pseudo_dataset_yesno_alpaca.json")

    if args.output_merge_path is None:
        args.output_merge_path = os.path.join(args.output_dir, "pseudo_dataset_alpaca.json")

    # get the base dir of output_dataset
    os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == "__main__":

    args = parse_argument()

    # if os.path.exists(args.output_composition_path) and not args.overwrite_output:
    #     print("Loading dataset from cache ...")
    #     multi_hop_datasets = pickle.load(open(args.output_composition_path, "rb"))
    # else:
    #     print("Building multi-hop dataset...")
    #     first_hop_examples, second_hop_examples = load_examples(args.data_dir)
    #     multi_hop_datasets = build_multihop_datasets(first_hop_examples, second_hop_examples)
    #
    #     with open(args.output_composition_path, "wb") as f:
    #         pickle.dump(multi_hop_datasets, f)
    #
    # if os.path.exists(args.output_filtered_path) and not args.overwrite_output:
    #     print("Loading filtered datasets from cache ...")
    #     filtered_datasets = json.load(open(args.output_filtered_path, "r"))
    #
    # else:
    #     print("Filtering and Exporting datasets to json ...")
    #     max_duplicate_degrees = [0, 0, 1, 1, 1, 1]
    #     filtered_datasets = {}
    #     for i, dataset_type in enumerate(multi_hop_datasets):
    #         filtered_dataset = filter_dataset(
    #             dataset=multi_hop_datasets[dataset_type],
    #             dataset_type=dataset_type,
    #             filter_type="answer",
    #             max_duplicate_degree=max_duplicate_degrees[i]
    #         )
    #         dict_dataset = export_to_dict(filtered_dataset, dataset_type)
    #         filtered_datasets[dataset_type] = dict_dataset
    #
    #     total_examples = 0
    #     for dataset_type in filtered_datasets:
    #         total_examples += len(filtered_datasets[dataset_type])
    #         print(f"{dataset_type} dataset: {len(filtered_datasets[dataset_type])} / {len(multi_hop_datasets[dataset_type])}")
    #
    #     print(f"Total number of examples: {total_examples}")
    #
    #     with open(args.output_filtered_path, "w") as f:
    #         json.dump(filtered_datasets, f, indent=4)
    #     print(f"Exported datasets to json at {args.output_filtered_path}")
    #
    # if not os.path.exists(args.output_yesno_path) or args.overwrite_output:
    #     filtered_datasets = json.load(open(args.output_filtered_path, "r"))
    #     demo_set = json.load(open(args.demo_path, "r"))
    #     yesno_demo_set = json.load(open(args.yesno_demo_path, "r"))
    #     yesno_dataset = {}
    #     model = Alpaca("alpaca-13b")
    #     for hop_type in filtered_datasets:
    #         yesno_dataset[hop_type] = {}
    #         yes_dataset = random.sample(filtered_datasets[hop_type], k=int(len(filtered_datasets[hop_type]) * 0.1))
    #         yesno_dataset[hop_type]["yes"] = generate_yes_no_questions(
    #             dataset=yes_dataset,
    #             demo_set=demo_set[hop_type],
    #             yesno_demos=yesno_demo_set["yes"],
    #             hop_type=hop_type,
    #             num_demos=4,
    #             model=model
    #         )
    #         no_dataset = random.sample(filtered_datasets[hop_type], k=int(len(filtered_datasets[hop_type]) * 0.1))
    #         yesno_dataset[hop_type]["no"] = generate_yes_no_questions(
    #             dataset=no_dataset,
    #             demo_set=demo_set[hop_type],
    #             yesno_demos=yesno_demo_set["no"],
    #             hop_type=hop_type,
    #             num_demos=4,
    #             model=model
    #         )
    #         print(f"Length of {hop_type} yesno dataset: {len(yesno_dataset[hop_type]['yes']) + len(yesno_dataset[hop_type]['no'])}")
    #         print(f"Length of {hop_type} original dataset: {len(filtered_datasets[hop_type]) * 0.2}")
    #         print(f"Success rate: {(len(yesno_dataset[hop_type]['yes']) + len(yesno_dataset[hop_type]['no'])) / (len(filtered_datasets[hop_type]) * 0.2)}")
    #
    #     with open(args.output_yesno_path, "w") as f:
    #         json.dump(yesno_dataset, f, indent=4, ensure_ascii=False)

    if not os.path.exists(args.output_completion_path) or args.overwrite_output:
        print(f"Generating completion dataset...")
        model = Alpaca("alpaca-13b")
        print("Loaded model.")

        demo_set = json.load(open(args.demo_path, "r"))
        filtered_datasets = json.load(open(args.output_filtered_path, "r"))
        complete_dataset = {}
        for hop_type in filtered_datasets:
            complete_dataset[hop_type] = gen_question_with_demos(
                dataset=filtered_datasets[hop_type],
                demo_set=demo_set[hop_type],
                hop_type=hop_type,
                num_demos=4,
                model=model)
            print(f"Length of {hop_type} complete dataset: {len(complete_dataset[hop_type])}")
            print(f"Length of {hop_type} original dataset: {len(filtered_datasets[hop_type])}")
            print(f"Success rate: {len(complete_dataset[hop_type]) / len(filtered_datasets[hop_type])}")

        with open(args.output_completion_path, "w") as f:
            json.dump(complete_dataset, f, indent=4)

    if not os.path.exists(args.output_merge_path) or args.overwrite_output:
        filtered_datasets = json.load(open(args.output_completion_path, "r"))
        yesno_datasets = json.load(open(args.output_yesno_path, "r"))
        merged_datasets = {}
        for hop_type in filtered_datasets:
            raw_dataset = filtered_datasets[hop_type] + yesno_datasets[hop_type]["yes"] + yesno_datasets[hop_type]["no"]
            raw_length = len(raw_dataset)
            new_dataset = [x for x in raw_dataset if x["question"] != ""]  # remove empty questions
            new_length = len(new_dataset)
            merged_datasets[hop_type] = new_dataset
            random.shuffle(merged_datasets[hop_type])
            print(f"Length of {hop_type} merged dataset: {new_length}")
            print(f"Length of {hop_type} original dataset: {raw_length}")
            print(f"Success rate: {new_length / raw_length}")

        with open(args.output_merge_path, "w") as f:
            json.dump(merged_datasets, f, indent=4, ensure_ascii=False)


