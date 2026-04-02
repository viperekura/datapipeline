from datasets import load_dataset, concatenate_datasets
from pipeline import export_dataset, TextNormalizer

normalizer = TextNormalizer()


def process_func(input_dict: dict):
    query = input_dict["prompt"] if input_dict["prompt"] else ""
    resp = input_dict["response"] if input_dict["response"] else ""
    return {
        "query": normalizer.normalize(query),
        "response": normalizer.normalize(resp),
    }


if __name__ == "__main__":
    all_data = [
        "stem_zh",
        "infinity-instruct",
        "firefly",
        "magpie",
        "dpsk-r1-distil",
        "coig-cqia",
        "disc-law",
        "neo_sft_phase2",
        "chinese-medical",
        "chinese-reasoning-distil",
        "psycho-10k-dpsk-r1",
        "sof-c-zh",
        "industryinstruction",
        "Chinese-QA-AFAF",
    ]

    dataset_list = []
    for subset in all_data:
        ds = load_dataset("Mxode/Chinese-Instruct", name=subset)
        dataset_list.append(ds["train"])

    combined_dataset = concatenate_datasets(dataset_list)
    export_dataset(
        dataset=combined_dataset,
        output_dir="./dataset",
        output_prefix="chinese-instruct-sft",
        process_func=process_func,
    )
