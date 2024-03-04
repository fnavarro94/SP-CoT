from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GPTNeoXForCausalLM,
    GPTNeoXConfig,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRReader,
    DPRReaderTokenizer,
    RagConfig,
    RagRetriever,
    RagTokenizer,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    T5ForConditionalGeneration,
    RealmTokenizer,
    RealmRetriever,
    RealmForOpenQA,
    LlamaTokenizer,
)
import transformers
import torch
import re
from typing import List




MODEL_DICT = {
    "gpt-neoxt-chat": "togethercomputer/GPT-NeoXT-Chat-Base-20B",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "dpr": {
        "question": "facebook/dpr-question_encoder-single-nq-base",
        "ctx": "facebook/dpr-ctx_encoder-single-nq-base",
        "reader": "facebook/dpr-reader-single-nq-base",
    },
    "rag-token": "facebook/rag-token-nq",
    "rag-sequence": "facebook/rag-sequence-nq",
    "flan-ul2": "google/flan-ul2",
    "flan-t5-xxl": "google/flan-t5-xxl",
    "t5-11b-ssm": "google/t5-11b-ssm-nq",
    "t5-3b-ssm": "google/t5-3b-ssm-nq",
    "t5-large-ssm": "google/t5-large-ssm-nq",
    "alpaca-7b": "chavinlo/alpaca-native",
    "alpaca-13b": "chavinlo/gpt4-x-alpaca",
    "vicuna-13b": "TheBloke/wizard-vicuna-13B-HF",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
    "wizard-13b": "TheBloke/wizardLM-13B-1.0-fp16"
}


class LocalModel(object):
    def __init__(self):
        self.tokenizer = None
        self.model_name = None
        self.model = None

    def get_response(self, *args, **kwargs):
        raise NotImplementedError

    def get_response_batch(self, *args, **kwargs):
        raise NotImplementedError


class GPTNeoXTChat(LocalModel):
    def __init__(self, model_name: str):
        super(GPTNeoXTChat, self).__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name], padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, text: str, use_temp: bool = True):
        if use_temp:
            prompt = f"<human>: {text}\n<bot>:"
        else:
            prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        return inputs, prompt

    def build_inputs_batch(self, texts: List[str], use_temp: bool = True):
        prompts = []
        for text in texts:
            if use_temp:
                prompt = f"<human>: {text}\n<bot>:"
            else:
                prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True)["input_ids"].to(self.model.device)
        return inputs, prompts

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.01,
            use_temp: bool = True,
            demo_idx: int = 0,
            replace_prompt: bool = True,
    ):
        input_ids, input_prompt = self.build_inputs(prompt, use_temp=use_temp)

        if temperature == 0:
            temperature = 0.01

        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if use_temp:
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if replace_prompt:
                raw_response = raw_response.replace(input_prompt, "")
            # find the answer between "<bot>:" and "<human>: " with regrex expression
            match = re.search(r".*?(?=<human>|\n\n|$)", raw_response, re.DOTALL)
            if match:
                response = match.group(0).strip()
            else:
                response = raw_response.split("\n\n")[demo_idx].strip()
        else:
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if replace_prompt:
                raw_response = raw_response.replace(input_prompt, "")
            response = raw_response.split("\n\n")[demo_idx]

        return response.strip()

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.01,
            use_temp: bool = True,
            demo_idx: int = 0,
            replace_prompt: bool = True,
    ):
        if temperature == 0:
            temperature = 0.01

        input_ids, input_prompts = self.build_inputs_batch(prompt, use_temp=use_temp)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for i in range(len(output_strs)):
            # find the answer between "<bot>:" and "\n" with regrex expression
            if use_temp:
                raw_response = output_strs[i]
                if replace_prompt:
                    raw_response = raw_response.replace(input_prompts[i], "")
                # find the answer between "<bot>:" and "<human>: " with regrex expression
                match = re.search(r".*?(?=<human>|\n\n|$)", raw_response, re.DOTALL)
                if match:
                    response = match.group(0).strip()
                else:
                    response = raw_response.split("\n\n")[demo_idx].strip()
                responses.append(response)
            else:
                raw_response = output_strs[i]
                if replace_prompt:
                    raw_response = raw_response.replace(input_prompts[i], "")
                responses.append(raw_response.split("\n\n")[demo_idx].strip())

        return responses


class Alpaca(LocalModel):
    def __init__(self, model_name: str):
        super(Alpaca, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_DICT[model_name])
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, instruction: str, input_text: str = None, plain_text: bool = True):
        if plain_text:
            prompt = f"{instruction}"
        else:
            if input_text is None:
                prompt = "Below is an instruction that describes a task. " \
                         "Write a response that appropriately completes the request.\n\n" \
                         f"### Instruction:\n{instruction}\n\n" \
                         "### Response:\n"
            else:
                prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
                         "Write a response that appropriately completes the request.\n\n" \
                         f"### Instruction:\n{instruction}\n\n" \
                         f"### Input:\n{input_text}\n\n" \
                         "### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        return inputs, prompt

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True)["input_ids"].to(self.model.device)
        return inputs, prompts

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.01,
            extra_text: str = None,
            plain_text: bool = False,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False,
    ):
        if temperature == 0.0:
            temperature = 0.01
        input_ids, input_text = self.build_inputs(
            prompt,
            input_text=extra_text,
            plain_text=plain_text,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_str
        if replace_prompt:
            response = response.replace(input_text, "")

        if ignore_double_newline:
            response = response.replace("\n\n", "\n")
        else:
            response = response.split("\n\n")[demo_idx]
        return response.strip()

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.01,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False
    ):
        input_ids, input_prompts = self.build_inputs_batch(prompt)
        if temperature == 0.0:
            temperature = 0.01

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for i in range(len(output_strs)):
            response = output_strs[i]
            if replace_prompt:
                response = response.replace(input_prompts[i], "")
            if ignore_double_newline:
                response = response.replace("\n\n", "\n")
            else:
                response = response.split("\n\n")[demo_idx]
            responses.append(response)
        return responses


class Falcon(LocalModel):
    def __init__(self, model_name: str):
        super(Falcon, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name], padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=MODEL_DICT[model_name],
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

    def build_inputs(self, text: str):
        prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt")
        return inputs

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.1,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False
    ):
        if temperature == 0.0:
            temperature = 0.01
        outputs = self.pipeline(
            prompt,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_str = outputs[0]["generated_text"]
        response = output_str
        if replace_prompt:
            response = response.replace(prompt, "").strip()
        if ignore_double_newline:
            response = response.replace("\n\n", "\n")
        else:
            response = response.split("\n\n")[demo_idx]
        return response

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.1,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False
    ):
        if temperature == 0:
            temperature = 0.01

        outputs = self.pipeline(
            prompt,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        responses = []
        for idx, batch_output in enumerate(outputs):
            response = batch_output[0]["generated_text"]
            if replace_prompt:
                response = response.replace(prompt[idx], "").strip()
            if ignore_double_newline:
                response = response.replace("\n\n", "\n")
            else:
                response = response.split("\n\n")[demo_idx]
            responses.append(response)
        return responses


class Wizard(LocalModel):
    def __init__(self, model_name: str):
        super(Wizard, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, instruction: str, plain_text: bool = True):
        if plain_text:
            prompt = f"{instruction}"
        else:
            prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                     "The assistant gives helpful, detailed, and polite answers to the user's questions.\n" \
                     f"USER: {instruction}\n" \
                     "ASSISTANT: "

        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        return inputs, prompt

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True)["input_ids"].to(self.model.device)
        return inputs, prompts

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.01,
            plain_text: bool = True,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False,
    ):
        if temperature == 0.0:
            temperature = 0.01
        input_ids, input_text = self.build_inputs(
            prompt,
            plain_text=plain_text,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_str
        if replace_prompt:
            response = response.replace(input_text, "")

        if ignore_double_newline:
            response = response.replace("\n\n", "\n")
        else:
            response = response.split("\n\n")[demo_idx]
        return response.strip()

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.01,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False
    ):
        input_ids, input_prompts = self.build_inputs_batch(prompt)
        if temperature == 0.0:
            temperature = 0.01

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for i in range(len(output_strs)):
            response = output_strs[i]
            if replace_prompt:
                response = response.replace(input_prompts[i], "")
            if ignore_double_newline:
                response = response.replace("\n\n", "\n")
            else:
                response = response.split("\n\n")[demo_idx]
            responses.append(response)
        return responses





















class GPTNeoX(LocalModel):
    def __init__(self, model_name: str):
        super(GPTNeoX, self).__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name], padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, text: str):
        prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        return inputs, prompt

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True)["input_ids"].to(self.model.device)
        return inputs, prompts

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.01,
            demo_idx: int = 0,
            replace_prompt: bool = True,
    ):
        input_ids, input_prompt = self.build_inputs(prompt)

        if temperature == 0:
            temperature = 0.01

        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if replace_prompt:
            raw_response = raw_response.replace(input_prompt, "")
        response = raw_response.split("\n\n")[demo_idx]

        return response.strip()

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.01,
            demo_idx: int = 0,
            replace_prompt: bool = True,
    ):
        if temperature == 0:
            temperature = 0.01

        input_ids, input_prompts = self.build_inputs_batch(prompt)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for i in range(len(output_strs)):
            raw_response = output_strs[i]
            if replace_prompt:
                raw_response = raw_response.replace(input_prompts[i], "")
            responses.append(raw_response.split("\n\n")[demo_idx].strip())

        return responses




class Vicuna(LocalModel):
    def __init__(self, model_name: str):
        super(Vicuna, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_DICT[model_name])
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, instruction: str, input_text: str = None, plain_text: bool = True):
        if plain_text:
            prompt = f"{instruction}"
        else:
            if input_text is None:
                prompt = "Below is an instruction that describes a task. " \
                         "Write a response that appropriately completes the request.\n\n" \
                         f"### Instruction:\n{instruction}\n\n" \
                         "### Response:"
            else:
                prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
                         "Write a response that appropriately completes the request.\n\n" \
                         f"### Instruction:\n{instruction}\n\n" \
                         f"### Input:\n{input_text}\n\n" \
                         "### Response:"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        return inputs, prompt

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True)["input_ids"].to(self.model.device)
        return inputs, prompts

    def get_response(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.01,
            extra_text: str = None,
            plain_text: bool = False,
            demo_idx: int = 0,
            replace_prompt: bool = True,
            ignore_double_newline: bool = False,
    ):
        if temperature == 0.0:
            temperature = 0.01
        input_ids, input_text = self.build_inputs(
            prompt,
            input_text=extra_text,
            plain_text=plain_text,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_str
        if replace_prompt:
            response = response.replace(input_text, "")

        if ignore_double_newline:
            response = response.replace("\n\n", "\n")
        else:
            response = response.split("\n\n")[demo_idx]

        print("="*100)
        print(input_text)
        print("="*100)
        print(response)
        print("="*100)

        return response.strip()

    def get_response_batch(
            self,
            prompt: List[str],
            max_tokens: int,
            temperature: float = 0.01,
            demo_idx: int = 0,
            replace_prompt: bool = True,
    ):
        input_ids, input_prompts = self.build_inputs_batch(prompt)
        if temperature == 0.0:
            temperature = 0.01

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=0.9,
            top_k=60,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for i in range(len(output_strs)):
            response = output_strs[i]
            if replace_prompt:
                response = response.replace(input_prompts[i], "")
            response = response.split("\n\n")[demo_idx]
            responses.append(response)
        return responses




class DPR(LocalModel):
    def __init__(self, device: str = "cuda:0"):
        super(DPR, self).__init__()
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
        self.reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(device)
        self.tokenizer = RagTokenizer(question_encoder=self.question_tokenizer, generator=self.reader_tokenizer)
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=False)
        self.retriever.question_encoder_tokenizer = self.question_tokenizer
        self.retriever.generator_tokenizer = self.reader_tokenizer
        self.reader.eval()
        self.context_encoder.eval()
        self.question_encoder.eval()
        self.device = device

    def build_batch_inputs(self, questions: List[str]):
        inputs = self.tokenizer.prepare_seq2seq_batch(
            questions,
            truncation=True,
            padding="longest",
            max_length=128,
            return_tensors="pt"
        )

        return inputs

    def retrieve_docs_batch(self, prompt: List[str], max_tokens: int, n_retrieved_docs: int = 5):
        inputs = self.build_batch_inputs(prompt)
        question_enc_outputs = self.question_encoder(
            input_ids=inputs["input_ids"].to(self.question_encoder.device),
            attention_mask=inputs["attention_mask"].to(self.question_encoder.device),
            return_dict=True
        )
        question_encoder_last_hidden_state = question_enc_outputs[0]
        retriever_outputs = self.retriever(
            inputs["input_ids"],
            question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
            n_docs=n_retrieved_docs,
            return_tensors="pt"
        )

        context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
            retriever_outputs["context_input_ids"],
            retriever_outputs["context_attention_mask"],
            retriever_outputs["retrieved_doc_embeds"],
            retriever_outputs["doc_ids"],
        )

        pq_texts = [self.context_tokenizer.decode(context_input_ids[i], skip_special_tokens=True) for i in
                    range(len(context_input_ids))]

        results = []
        for pq_text in pq_texts:
            text_tokens = pq_text.split("/")
            results.append({
                "context": text_tokens[1].strip(),
                "title": text_tokens[0].strip(),
            })

        outputs = []
        for i in range(len(prompt)):
            outputs.append(results[i*n_retrieved_docs: (i+1)*n_retrieved_docs])

        return outputs




    def get_response_batch(self, prompt: List[str], max_tokens: int, n_retrieved_docs: int = 5):
        inputs = self.build_batch_inputs(prompt)
        question_enc_outputs = self.question_encoder(
            input_ids=inputs["input_ids"].to(self.question_encoder.device),
            attention_mask=inputs["attention_mask"].to(self.question_encoder.device),
            return_dict=True
        )
        question_encoder_last_hidden_state = question_enc_outputs[0]
        retriever_outputs = self.retriever(
            inputs["input_ids"],
            question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
            n_docs=n_retrieved_docs,
            return_tensors="pt"
        )

        context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
            retriever_outputs["context_input_ids"],
            retriever_outputs["context_attention_mask"],
            retriever_outputs["retrieved_doc_embeds"],
            retriever_outputs["doc_ids"],
        )

        pq_texts = [self.context_tokenizer.decode(context_input_ids[i], skip_special_tokens=True) for i in
                    range(len(context_input_ids))]
        context_texts = []
        question_texts = []
        titles = []
        for pq_text in pq_texts:
            text_tokens = pq_text.split("/")
            context_texts.append(text_tokens[1].strip())
            titles.append(text_tokens[0].strip())
            question_texts.append(text_tokens[-1].strip())

        reader_inputs = self.reader_tokenizer(
            questions=question_texts,
            titles=titles,
            texts=context_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        )

        reader_outputs = self.reader(
            input_ids=reader_inputs["input_ids"].to("cuda:0"),
            attention_mask=reader_inputs["attention_mask"].to("cuda:0"),
            output_hidden_states=True,
            return_dict=True
        )

        start_logits = reader_outputs.start_logits
        end_logits = reader_outputs.end_logits

        batch_size = len(prompt)
        outputs = []
        for i in range(batch_size):
            start_logits_local = start_logits[i*n_retrieved_docs: (i+1)*n_retrieved_docs]
            end_logits_local = end_logits[i*n_retrieved_docs: (i+1)*n_retrieved_docs]
            answer_start = torch.argmax(start_logits_local, dim=1)
            answer_end = torch.argmax(end_logits_local, dim=1)
            input_ids_local = reader_inputs["input_ids"][i*n_retrieved_docs: (i+1)*n_retrieved_docs]
            pred_ans = []
            for j in range(n_retrieved_docs):
                if answer_end[j] < answer_start[j]:
                    continue
                if answer_end[j] - answer_start[j] + 1 > max_tokens:
                    continue
                if answer_start[j] == 0 or answer_end[j] == 0:
                    continue
                answer = self.reader_tokenizer.decode(input_ids_local[j][answer_start[j]:answer_end[j] + 1])
                pred_ans.append(answer)
            outputs.append("/".join(list(set(pred_ans))))
        return outputs


class RAG_Token(LocalModel):
    def __init__(self, device: str = "cuda:0"):
        super(RAG_Token, self).__init__()
        # self.retriever = RagRetriever.from_pretrained(
        #     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
        # )
        config = RagConfig.from_pretrained("facebook/rag-token-nq")
        config.question_encoder.max_length = 64
        config.generator.max_length = 16
        config.max_length = 80

        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=False, device=device)
        self.retriever.config = config
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.retriever).to(device)

    def build_inputs_batch(self, texts: List[str]):
        inputs = self.tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt", padding=True).to(self.model.device)
        return inputs["input_ids"]

    def get_response_batch(self, prompt: List[str], max_tokens: int, n_retrieved_docs: int = 10, num_beams: int = 5):
        outputs = self.model.generate(
            input_ids=self.build_inputs_batch(prompt),
            n_docs=n_retrieved_docs,
            num_beams=num_beams,
            num_return_sequences=1,
            max_length=max_tokens
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_strs


class RAG_Sequence(LocalModel):
    def __init__(self, device: str = "cuda:0"):
        super(RAG_Sequence, self).__init__()

        config = RagConfig.from_pretrained("facebook/rag-sequence-nq")
        config.question_encoder.max_length = 64
        config.generator.max_length = 64
        config.max_length = 128

        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=False, device=device)
        self.retriever.config = config
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=self.retriever).to(device)

    def build_inputs_batch(self, texts: List[str]):
        inputs = self.tokenizer.prepare_seq2seq_batch(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=128
        ).to(self.model.device)
        return inputs["input_ids"]

    def get_response_batch(self, prompt: List[str], max_tokens: int, n_retrieved_docs: int = 10, num_beams: int = 5):
        outputs = self.model.generate(
            input_ids=self.build_inputs_batch(prompt),
            n_docs=n_retrieved_docs,
            num_beams=num_beams,
            num_return_sequences=1,
            max_length=max_tokens
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_strs


class Flan_UL2_20B(LocalModel):
    def __init__(self, model_name):
        super(Flan_UL2_20B, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name], truncation_side="left")
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.bfloat16)

    def build_inputs(self, text: str):
        prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.model.device)
        return inputs

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.model.device)
        return inputs

    def get_response(self, prompt: str, max_tokens: int):
        outputs = self.model.generate(
            input_ids=self.build_inputs(prompt),
            max_length=max_tokens)

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = [output.strip() for output in output_str]
        return response

    def get_response_batch(self, prompt: List[str], max_tokens: int):
        outputs = self.model.generate(
            input_ids=self.build_inputs_batch(prompt),
            max_length=max_tokens,
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [output_str.strip() for output_str in output_strs]
        return responses


class Flan_T5(LocalModel):
    def __init__(self, model_name):
        super(Flan_T5, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name], truncation_side="left")
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16)

    def build_inputs(self, text: str):
        prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.model.device)
        return inputs

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.model.device)
        return inputs

    def get_response(self, prompt: str, max_tokens: int, temperature: float = 1.0):
        outputs = self.model.generate(
            input_ids=self.build_inputs(prompt),
            temperature=temperature,
            do_sample=True,
            max_length=max_tokens)

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_str.strip()
        print("=" * 100)
        print("Prompt: ", prompt)
        print("=" * 100)
        print("Response: ", response)
        print("=" * 100)

        return response

    def get_response_batch(self, prompt: List[str], max_tokens: int):
        outputs = self.model.generate(
            input_ids=self.build_inputs_batch(prompt),
            max_length=max_tokens,
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [output_str.strip() for output_str in output_strs]
        return responses




class T5_SSM(LocalModel):
    def __init__(self, model_name):
        super(T5_SSM, self).__init__()
        self.model_name = MODEL_DICT[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DICT[model_name], device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)

    def build_inputs(self, text: str):
        prompt = f"{text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        return inputs

    def build_inputs_batch(self, texts: List[str]):
        prompts = []
        for text in texts:
            prompt = f"{text}"
            prompts.append(prompt)
        inputs = self.tokenizer(text=prompts, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        return inputs

    def get_response(self, prompt: str, max_tokens: int):
        outputs = self.model.generate(
            input_ids=self.build_inputs(prompt),
            max_length=max_tokens
        )

        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = [output.strip() for output in output_str]
        return response

    def get_response_batch(self, prompt: List[str], max_tokens: int):
        outputs = self.model.generate(
            input_ids=self.build_inputs_batch(prompt),
            max_length=max_tokens,
        )
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [output_str.strip() for output_str in output_strs]
        return responses


class REALM(LocalModel):
    def __init__(self, device: str = "cuda:0"):
        super(REALM, self).__init__()
        self.tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
        self.retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
        self.model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=self.retriever).to(device)
        self.model.eval()
        self.device = device

    def build_batch_inputs(self, questions: List[str]):
        inputs = self.tokenizer(
            questions,
            truncation=True,
            padding="longest",
            max_length=128,
            return_tensors="pt"
        )

        return inputs

    def get_response_batch(self, prompts: List[str], max_tokens: int, n_retrieved_docs: int = 5):
        inputs = self.build_batch_inputs(prompts)
        outputs = []

        for idx in range(len(inputs["input_ids"])):
            input_ids = inputs["input_ids"][idx].unsqueeze(0).to(self.device)
            attention_mask = inputs["attention_mask"][idx].unsqueeze(0).to(self.device)
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            raw_answer = self.tokenizer.decode(model_output.predicted_answer_ids, skip_special_tokens=True)
            outputs.append(raw_answer)
        return outputs
