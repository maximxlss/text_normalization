import csv
import random
from statistics import mean
import numpy as np
from transformers import (
    T5Config,
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import textdistance
from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)


# ru_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
MAX_LENGTH = 768  # affects vram consumption (shouldn't really affect quality)
MIN_CHARS = 32
BATCH_SIZE = 52

tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer.json")

# need to do this or an error occures, not sure why
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = 0

configuration = T5Config(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    d_ff=1024,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.0,
    feed_forward_proj="gated-gelu",
    decoder_start_token_id=0,
)

model = T5ForConditionalGeneration(configuration)

# model = T5ForConditionalGeneration.from_pretrained("./v4_large/checkpoint-84252")

configuration = TrainingArguments(
    output_dir="text-normalization-ru-terrible",
    optim="adamw_torch",
    tf32=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=21.65,
    save_strategy="steps",
    save_steps=0.005,
    save_total_limit=2,
    logging_steps=0.005,
    learning_rate=1e-8,
    lr_scheduler_type="inverse_sqrt",
    warmup_ratio=0.01,
    evaluation_strategy="steps",
    logging_first_step=True,
    include_inputs_for_metrics=True,
    remove_unused_columns=False,
)

sentence_dataset = load_dataset(
    "csv",
    data_files="ru_train_preprocessed.csv",
    split="train",
    converters={0: str, 1: str},
)  # 761435 sentences, longest sentence is 9622 bytes (lol)
extra_dataset = load_dataset(
    "csv",
    data_files="ru_train_extras.csv",
    split="train",
    converters={0: str, 1: str},
)
dataset: Dataset = concatenate_datasets([sentence_dataset, extra_dataset])


def compute_length(exs):
    n = min(max(len(exs["input_ids"]), len(exs["labels"])), MAX_LENGTH)
    return {"length": n}


dataset = dataset.map(compute_length)

dataset = dataset.train_test_split(500, seed=42)

dataset = DatasetDict(
    {k: _dataset.flatten_indices() for k, _dataset in dataset.items()}
)

# dataset = dataset.flatten_indices()

dataset = dataset.sort("length", reverse=True)


def random_pad(a, b):
    alphabet = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

    needs = MIN_CHARS - max(len(a), len(b))
    if needs <= 0:
        return a, b

    left = needs // 2
    right = needs - left

    if left > 0:
        left = "".join(random.choices(alphabet, k=left - 1)) + " "
    else:
        left = ""
    right = " " + "".join(random.choices(alphabet, k=right - 1))

    return left + a + right, left + b + right


def preprocess(exs):
    process_keys = ["input_ids", "labels"]

    # sometimes remove the ending symbol, or the network will become sensetive to dots
    if random.random() < 0.33:
        for k in process_keys:
            if isinstance(exs[k], str):
                exs[k] = exs[k][:-1]
            else:
                for i in range(len(exs[k])):
                    exs[k][i] = exs[k][i][:-1]

    for i in range(len(exs["input_ids"])):
        exs["input_ids"][i], exs["labels"][i] = random_pad(
            exs["input_ids"][i], exs["labels"][i]
        )

    return {
        k: tokenizer(
            exs[k],
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="np",
        ).input_ids
        for k in ["input_ids", "labels"]
    }


dataset.set_transform(preprocess)

# trim too short sentences
# trim_to = int(len(dataset["train"]) * 0.8)
# dataset["train"] = dataset["train"].select(range(trim_to))


def compute_metrics(prediction: EvalPrediction):
    labels = prediction.label_ids
    inputs = prediction.inputs
    predictions = np.argmax(prediction.predictions[0], axis=-1)

    for i in range(len(labels)):
        labels[i][labels[i] < 0] = 0
        inputs[i][inputs[i] < 0] = 0
        predictions[i][predictions[i] < 0] = 0

    inputs_decoded = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions_decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    with open(
        "evaluation.csv", "w", encoding="UTF-8", errors="replace", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["inputs", "labels", "predictions"])
        writer.writerows(zip(inputs_decoded, labels_decoded, predictions_decoded))

    distances = [
        textdistance.levenshtein(p, l)
        for p, l in zip(predictions_decoded, labels_decoded)
    ]

    mean_distance = mean(distances)
    max_distance = max(distances)

    return {"mean_distance": mean_distance, "max_distance": max_distance}


class MyTrainer(Trainer):
    def _get_train_sampler(self):
        return None

    def _load_optimizer_and_scheduler(self, checkpoint):
        return None


trainer = MyTrainer(
    model=model,
    args=configuration,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)


trainer.train(
    resume_from_checkpoint=True,
)

trainer.save_state()

trainer.push_to_hub()
