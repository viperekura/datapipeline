from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    conversations = input_dict["conversations"]
    examples = []
    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i]["value"]
        assistant_msg = conversations[i + 1]["value"]
        examples.append({"query": user_msg, "response": assistant_msg})
    return examples


if __name__ == "__main__":
    dataset = load_dataset("teknium/OpenHermes-2.5")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="OpenHermes-2.5",
        process_func=process_func,
    )
