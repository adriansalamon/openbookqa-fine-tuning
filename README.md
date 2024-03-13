## Simple fine-tuning T5 for OpenbookQA

This repository contains the code for fine-tuning T5 for OpenbookQA. It is based
on the [Huggingface Transformers](https://huggingface.co/docs/transformers/en/index) library,
and quite crude. Done as a project for the CSE447 course at the University of Washington.
Don't expect it to be maintained or stable in any way, but please feel free to use it
as a starting point or reference for your own work. With the T5-base model,
it can be trained on a single GPU, and manages to get an accuracy of 0.68 on
the validation set after 5 epochs.

It can be used to fine-tune T5 either with the knowledge-base or without it.
Using the knowledge base, we use sentance embeddings for all sentances in the
knowledge base to find the closest sentances to the question. This is done using
the `all-mpnet-base-v2` model, and cosine distances.

To run experiments:

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the OpenbookQA dataset:

```bash
python download_data.py
```

Annotate the dataset with context from the knowledge base:

```bash
python create_context.py --data_dir=OpenBookQA-V1-Sep2018 --out_dir=annotated
```

### With knowledge base

Train a model:

```bash
python train.py --data_dir=annotated --use_knowledge=True \
                --model="google-t5/t5-small" --lr=1e-4 --batch_size=8 \
                --epochs=1
```

This will create a log file in `./logs` directory and save checkpoints in the `./checkpoints` directory.

To evaluate the model, run the following:

```bash
python evaluate.py --data_dir=annotated --use_knowledge=True \
                   --model="google-t5/t5-small" \
                   --checkpoint=checkpoints/xxxxxx.xxxxx_train_x_0.xxxxx.pth
```

### Without knowledge base

To train a model without the knowledge base, run the following

```bash
python train.py --data_dir=OpenBookQA-V1-Sep2018 \
                --model="google-t5/t5-small" \
                --lr=1e-4 --batch_size=8 --epochs=1
```

And to evaluate:

```bash
python evaluate.py --data_dir=OpenBookQA-V1-Sep2018 \
                   --model="google-t5/t5-small" \
                   --checkpoint=checkpoints/xxxxxx.xxxxx_train_x_0.xxxxx.pth
```
