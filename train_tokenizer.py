from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import BPEDecoder
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, concatenate_datasets, Dataset


VOCAB_SIZE = 5120

tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
tokenizer.pre_tokenizer = Sequence([Whitespace()])  # , Digits(individual_digits=True)])
tokenizer.normalizer = Lowercase()
tokenizer.decoder = BPEDecoder()
tokenizer.post_processor = TemplateProcessing(
    single="$0 </s>",
    pair="$A </s> $B:1 </s>:1",
    special_tokens=[("<pad>", 0), ("</s>", 1)],
)
tokenizer.enable_padding(pad_token="<pad>")

sentence_dataset = load_dataset(
    "csv",
    data_files="ru_train_preprocessed.csv",
    split="train",
    converters={0: str, 1: str},
)  # 761435 sentences, longest sentence is 9660 bytes (lol)
extra_dataset = load_dataset(
    "csv",
    data_files="ru_train_extras.csv",
    split="train",
    converters={0: str, 1: str},
)
dataset: Dataset = concatenate_datasets([sentence_dataset, extra_dataset])

ru_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

alphabet = [chr(i) for i in range(256)] + list(ru_alphabet)

trainer = BpeTrainer(
    special_tokens=["<pad>", "</s>", "<unk>"],
    vocab_size=VOCAB_SIZE,
    initial_alphabet=alphabet,
    limit_alphabet=len(alphabet) + 30,
    end_of_word_suffix="</w>",
    show_progress=True,
)

tokenizer.train_from_iterator(
    (
        s["input_ids"] + s["labels"]
        for s in dataset.iter(1)
        if s["input_ids"] != s["labels"]
    ),
    trainer,
)

tokenizer.save("./tokenizer.json")
