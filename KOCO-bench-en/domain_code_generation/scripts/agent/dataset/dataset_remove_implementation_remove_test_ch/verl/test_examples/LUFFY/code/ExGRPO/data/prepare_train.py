import argparse
from datasets import load_dataset
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Prepare training data from HuggingFace dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load from HuggingFace")
    parser.add_argument("--output_file", type=str, default="dataset.parquet", help="Output parquet file name")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split="train")

    print(dataset[0])

    ret_dict = []
    for item in dataset:
        ret_dict.append(item)

    train_df = pd.DataFrame(ret_dict)
    train_df.to_parquet(args.output_file)
    print(f"Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()