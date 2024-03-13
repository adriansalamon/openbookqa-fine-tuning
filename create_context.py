import argparse
import os
from sentence_transformers import SentenceTransformer
import pickle
from annoy import AnnoyIndex
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default="OpenBookQA-V1-Sep2018",
    type=str,
    help="Directory where the source data is stored",
)
parser.add_argument(
    "--out_dir",
    default="OpenBookQA-V1-Sep2018",
    type=str,
    help="Directory where the annotated data with context will be stored",
)


def read_data(folder, file):
    data = pd.read_json(
        os.path.join(folder, "Data", "Main", f"{file}.jsonl"), lines=True
    )
    data["answers"] = data["question"].apply(
        lambda x: [y["text"] for y in x["choices"]]
    )
    data["question"] = data["question"].apply(lambda x: x["stem"])

    embeddings = model.encode(data["question"], show_progress_bar=True)

    data["context"] = None
    for i, embedding in enumerate(embeddings):
        closest = index.get_nns_by_vector(embedding, 25)
        data.at[i, "context"] = [sentence_to_idx[idx] for idx in closest]

    return data, embeddings


if __name__ == "__main__":
    args = parser.parse_args()
    folder = args.data_dir

    assert os.path.exists(folder), f"Data folder {folder} does not exist"
    os.makedirs(args.out_dir, exist_ok=True)

    file = os.path.join(folder, "Data", "Main", "openbook.txt")
    facts = [line.strip()[1:-1] for line in open(file, "r")]

    print("Loading model")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Encoding fact sentences")
    embeddings = model.encode(facts, show_progress_bar=True)

    print("Creating search index")

    dim = embeddings.shape[1]
    index = AnnoyIndex(dim, "angular")
    sentence_to_idx = {}
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
        sentence_to_idx[i] = facts[i]

    index.build(1024)

    print("Annotating data with context")

    for file in ["train", "dev", "test"]:
        data, embeddings = read_data(folder, file)
        path = f"{args.out_dir}/{file}_with_context.pkl"
        pickle.dump(data, open(path, "wb"))
        print(f"{file} with context written to {path}")

    print(
        "Done, you can now run train.py with the --use_knowledge flag set to True to use the context in the training process."
    )
