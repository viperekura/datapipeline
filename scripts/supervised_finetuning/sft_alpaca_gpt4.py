from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    instruction = input_dict["instruction"]
    inp = input_dict.get("input", "")
    if inp:
        query = instruction + "\n" + inp
    else:
        query = instruction
    return {"query": query, "response": input_dict["output"]}


if __name__ == "__main__":
    dataset = load_dataset("llm-wizard/alpaca-gpt4-data")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="alpaca-gpt4-data",
        process_func=process_func,
    )
