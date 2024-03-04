import transformers
from torch.utils.data import Dataset
import json
import logging
# import datasets



class Seq2SeqDataset(Dataset):
    def __init__(self, data_path):
        super(Seq2SeqDataset, self).__init__()

        with open(data_path, 'r') as f:
            raw_dataset = json.load(f)

        self.sources = []
        self.targets = []
        for key in raw_dataset:
            for sample in raw_dataset[key]:
                input_text = f"Question: {sample['question']}\nLet's think step by step:\n"
                target_text = ""
                for j, hop in enumerate(sample['hops']):
                    # target_text += f"Step {j + 1}: {hop['question']}\n" \
                                   # f"Knowledge: {hop['context']}\n" \
                                   # f"Answer: {hop['answer']}, \n"
                    target_text += f"Step {j + 1}: {hop['question']}\n" \
                                   f"Answer: {hop['answer']}\n"
                target_text += f"Therefore, the final answer is: {sample['answer']}"
                self.sources.append(input_text)
                self.targets.append(target_text)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        return self.sources[item], self.targets[item]


class Seq2SeqCollator(object):
    def __init__(self, tokenizer, instruction_length=128, output_length=384):
        self.tokenizer = tokenizer
        self.instruction_length = instruction_length
        self.output_length = output_length

    def __call__(self, batch):
        sources = [ex[0] for ex in batch]
        targets = [ex[1] for ex in batch]

        inputs = self.tokenizer(
            sources,
            max_length=self.instruction_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        labels = self.tokenizer(
            targets,
            max_length=self.output_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).input_ids

        inputs['labels'] = labels

        return inputs

