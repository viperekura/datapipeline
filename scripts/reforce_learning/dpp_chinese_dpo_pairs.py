from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    return {
        "query": input_dict["prompt"],
        "chosen": input_dict["chosen"],
        "rejected": input_dict["rejected"],
    }


if __name__ == "__main__":
    dataset = load_dataset("wenbopan/Chinese-dpo-pairs")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="Chinese-dpo-pairs",
        process_func=process_func,
    )
