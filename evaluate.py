import torch
import argparse
from train import eval, get_data, encode_data
from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="google-t5/t5-small", type=str, help="Base model name"
)
parser.add_argument(
    "--checkpoint", default=None, type=str, help="Path to the checkpoint to load"
)
parser.add_argument(
    "--use_knowledge",
    default=False,
    type=bool,
    help="Use the knowledge base from the dataset",
)
parser.add_argument(
    "--data_dir",
    default="OpenBookQA-V1-Sep2018",
    type=str,
    help="Directory where the data is stored",
)


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_data, train_labels = get_data(args.data_dir, "train", args.use_knowledge)
    val_data, val_labels = get_data(args.data_dir, "dev", args.use_knowledge)
    test_data, test_labels = get_data(args.data_dir, "test", args.use_knowledge)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)

    train_enc = encode_data(tokenizer, train_data, train_labels, device)
    val_enc = encode_data(tokenizer, val_data, val_labels, device)
    test_enc = encode_data(tokenizer, test_data, test_labels, device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Evaluating model")

    train_acc = eval(model, train_enc, batch_size=8)
    print(f"Train set accuracy: {train_acc}")

    val_acc = eval(model, val_enc, batch_size=8)
    print(f"Validation set accuracy: {val_acc}")

    test_acc = eval(model, test_enc, batch_size=8)
    print(f"Test set accuracy: {test_acc}")
