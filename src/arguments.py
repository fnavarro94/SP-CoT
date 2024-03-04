import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # Required parameters
    parser.add_argument("--openai_api_key", type=str, required=True, help="openai api key")

    # Default parameters
    parser.add_argument("--task", type=str, default="open-domain-qa",
                        choices=["open-domain-qa", "fact-verification"],
                        help="task name in [open-domain-qa, fact-verification]")

    parser.add_argument("--dataset", type=str, default="hotpot-qa",
                        help="dataset name")

    parser.add_argument("--dataset_path", type=str, default="data/hotpot-qa/dev_evidence.json")

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--max_num_worker", type=int, default=6, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3.5-turbo-0301", help="model used for response generation.")

    parser.add_argument(
        "--max_tokens", type=int, default=5, help="maximum length of output tokens by model for zero-shot")


















    parser.add_argument(
        "--model", type=str, default="gpt3-xl", help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=[
            "zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    args = parser.parse_args()

    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args
