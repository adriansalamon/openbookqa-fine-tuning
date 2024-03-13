import argparse
import os
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default="OpenBookQA-V1-Sep2018",
    type=str,
    help="Directory where the data is stored",
)
parser.add_argument(
    "--model", default="google-t5/t5-small", type=str, help="Model name"
)
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument(
    "--use_knowledge",
    default=False,
    type=bool,
    help="Use the knowledge base from the dataset",
)

# Magic numbers
max_length = 512
label_length = 2
checkpoint_folder = "checkpoints"
log_folder = "logs"


def format_question(question, answers, context=[]):
    context = " .".join(context)
    text = f"question: {question} choice A: {answers[0]} choice B: {answers[1]} choice C: {answers[2]} choice D: {answers[3]}"
    if context != "":
        text += f" context: {context}"
    return text


def get_data(dir, file, use_knowledge=False):
    data = None
    if use_knowledge:
        data = pd.read_pickle(f"{dir}/{file}_with_context.pkl")
        data["formatted"] = data.apply(
            lambda x: format_question(x["question"], x["answers"], x["context"]), axis=1
        )
    else:
        path = os.path.join(dir, "Data", "Main", f"{file}.jsonl")
        print("Reading data from", path)
        data = pd.read_json(path, lines=True)
        data["answers"] = data["question"].apply(
            lambda x: [y["text"] for y in x["choices"]]
        )
        data["question"] = data["question"].apply(lambda x: x["stem"])
        data["formatted"] = data.apply(
            lambda x: format_question(x["question"], x["answers"]), axis=1
        )

    return data["formatted"].tolist(), data["answerKey"].tolist()


def encode_data(tokenizer, inputs, labels, device="cpu"):
    inputs_enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    labels_enc = tokenizer(
        labels,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=label_length,
    ).to(device)

    labels = labels_enc["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    inputs_enc["label"] = labels

    return inputs_enc


def eval(model, inputs, batch_size=8):
    model.eval()

    batches = len(inputs["input_ids"]) // batch_size

    loop = tqdm(total=batches, position=0, leave=False)
    total_correct = 0
    for i in range(0, len(inputs["input_ids"]), batch_size):
        batch = inputs[i : i + batch_size]
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=2,
        )

        preds = outputs[:, 1:]  # Ignore the first token
        true = batch["label"][:, :1]

        correct = (preds == true).sum()
        total_correct += correct
        loop.set_description(f"Accuracy: {correct / len(batch['input_ids'])}")
        loop.update(1)

    return total_correct / len(inputs["input_ids"])


def train(model, train_data, val_data, optimizer, args):
    batch_size = args.batch_size
    epochs = args.epochs

    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    start_time = time.time()
    log_file = open(f"{log_folder}/{start_time}_log.txt", "w")

    log_file.write(f"Start time: {start_time}\n")
    log_file.write(
        f"Params: model: {args.model}, knowledge_base: {args.use_knowledge}, batch_size: {batch_size}, epochs: {epochs}, learning_rate: {args.lr}\n"
    )

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(train_data["input_ids"]))
        train_data["input_ids"] = train_data["input_ids"][shuffle]
        train_data["attention_mask"] = train_data["attention_mask"][shuffle]
        train_data["label"] = train_data["label"][shuffle]

        train_batches = len(train_data["input_ids"]) // batch_size
        loop = tqdm(total=train_batches, position=0, leave=False)

        for i in range(0, len(train_data["input_ids"]), batch_size):
            batch = train_data[i : i + batch_size]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Tqdm print loss
            loop.set_description(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
            loop.update(1)
            log_file.write(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item():.4f}\n")
            log_file.flush()

        train_acc = eval(model, train_data, batch_size)
        val_acc = eval(model, val_data, batch_size)
        print(
            f"Epoch {epoch + 1} - Train Accuracy: {train_acc:.4f} - Val Accuracy: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"{checkpoint_folder}/{start_time}_train_{epoch}_{val_acc:.4f}.pth",
            )

        log_file.write(
            f"DONE, Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}\n"
        )

    log_file.close()


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        assert False, f"Directory {args.data_dir} does not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, train_labels = get_data(args.data_dir, "train", args.use_knowledge)
    val_data, val_labels = get_data(args.data_dir, "dev", args.use_knowledge)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)

    train_enc = encode_data(tokenizer, train_data, train_labels, device)
    val_enc = encode_data(tokenizer, val_data, val_labels, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training")

    train(model, train_enc, val_enc, optimizer, args)

    print("Training done")
