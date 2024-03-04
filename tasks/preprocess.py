import json
import os
import argparse
from tqdm import tqdm


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_wiki_pages(wiki_path):
    wiki_pages = {}
    for file in tqdm(os.listdir(wiki_path), desc="Loading wiki-pages"):
        if file.endswith(".jsonl"):
            file_path = os.path.join(wiki_path, file)
            wiki_data = load_jsonl(file_path)
            for page in wiki_data:
                wiki_pages[page["title"]] = page
    return wiki_pages


def get_evidence(claim_data, wiki_pages):
    evidence_texts = []
    evidence = claim_data["evidence"]

    for evidence_set in evidence:
        set_texts = []
        for item in evidence_set["content"]:
            item_tokens = item.split("_")
            page_id = item_tokens[0]
            page = wiki_pages.get(page_id)
            if not page:
                continue
            evidence_type = item_tokens[1]
            evidence_key = "_".join(item_tokens[1:])
            if evidence_type == "sentence":
                set_texts.append(page[evidence_key])
            elif evidence_type == "section":
                set_texts.append(page[evidence_key]["value"])
            else:
                for key in page["order"]:
                    if key.startswith("sentence"):
                        continue
                    elif key.startswith("table"):
                        found_flag = False
                        for row in page[key]["table"]:
                            for cell in row:
                                if cell["id"] == evidence_key:
                                    set_texts.append(cell["value"])
                                    found_flag = True
                                    break
                            if found_flag:
                                break

                    elif key.startswith("list"):
                        for list_item in page[key]["list"]:
                            if list_item["id"] == evidence_key:
                                set_texts.append(list_item["value"])
                                break
        evidence_texts.append(set_texts)

    return evidence_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="dataset name: [feverous]"
                        )

    args = parser.parse_args()

    if args.dataset == "feverous":

        # Set the paths to your dataset and wiki-pages folders
        dataset_path = "data/feverous"
        wiki_pages_path = "/data2/feverous/FeverousWikiv1"

        # Load the dataset and wiki-pages
        train_data = load_jsonl(os.path.join(dataset_path, "train.jsonl"))
        dev_data = load_jsonl(os.path.join(dataset_path, "dev.jsonl"))
        wiki_pages = load_wiki_pages(wiki_pages_path)

        dataset = []
        # Get the evidence for each claim
        for claim in tqdm(dev_data, desc="Getting dev evidence"):
            if claim["label"] == "NOT ENOUGH INFO":
                continue
            evidence = get_evidence(claim, wiki_pages)
            dataset.append({
                "id": claim["id"],
                "claim": claim["claim"],
                "label": claim["label"],
                "evidence": evidence,
                "challenge": claim["challenge"]
            })

        # Save the dataset
        with open(os.path.join(dataset_path, "dev_evidence.jsonl"), "w") as file:
            for claim in dataset:
                file.write(json.dumps(claim) + "\n")

        with open(os.path.join(dataset_path, "dev_evidence.json"), "w") as file:
            json.dump(dataset, file, indent=4)

        dataset = []
        # Get the evidence for each claim
        for claim in tqdm(train_data, desc="Getting train evidence"):
            if claim["label"] == "NOT ENOUGH INFO":
                continue
            evidence = get_evidence(claim, wiki_pages)
            dataset.append({
                "id": claim["id"],
                "claim": claim["claim"],
                "label": claim["label"],
                "evidence": evidence,
                "challenge": claim["challenge"]
            })

        # Save the dataset
        with open(os.path.join(dataset_path, "train_evidence.jsonl"), "w") as file:
            for claim in dataset:
                file.write(json.dumps(claim) + "\n")

        with open(os.path.join(dataset_path, "train_evidence.json"), "w") as file:
            json.dump(dataset, file, indent=4)
