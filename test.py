from pathlib import Path
import time
from typing import NamedTuple
from transformers import (
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
)


path = Path("./text-normalization-ru-terrible")

if not (path / "pytorch_model.bin").exists():
    try:
        path = max(path.glob("checkpoint-*"))
    except ValueError:
        path = next(path.glob("*"))


try:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
except ValueError:
    tokenizer = ByT5Tokenizer()

t = time.time()
model = T5ForConditionalGeneration.from_pretrained(path)
print(f"[{time.time() - t}s.] Loaded {path}")

model.resize_token_embeddings(tokenizer.vocab_size + 100)

# gpu = torch.device("cuda")

# model.to(gpu, torch.float32)


class Generation(NamedTuple):
    out: str
    time_tokenizer: float
    time_model: float


def generate(text):
    time_start = time.time()
    inp_ids = tokenizer(
        text,
        return_tensors="pt",
    ).input_ids  # .to(gpu)
    print(inp_ids)
    time_tokenizer = time.time() - time_start
    # out_ids = np.argmax(
    #     model.forward(inp_ids, decoder_input_ids=inp_ids).logits.detach().numpy(),
    #     axis=-1,
    # )
    out_ids = model.generate(inp_ids, max_new_tokens=128)
    #     input_ids=inp_ids,
    #     max_new_tokens=128,
    #     do_sample=False,
    # )
    print(out_ids)
    time_model = time.time() - (time_start + time_tokenizer)
    out = tokenizer.batch_decode(out_ids)[0]
    time_tokenizer += time.time() - (time_start + time_model + time_tokenizer)
    # out = tokenizer.encode(text)
    # print(out)
    # time_tokenizer, time_model = time.time() - time_start, 0
    # out = tokenizer.tokenize(text)
    return Generation(out, time_tokenizer, time_model)


out = generate("РСФСР, 24 июля, Aternos.")
print(f"[tok: {out.time_tokenizer:.3f}s. model: {out.time_model:.3f}s.] {out.out}")

while True:
    inp = input("> ")
    out = generate(inp)
    print(f"[tok: {out.time_tokenizer:.3f}s. model: {out.time_model:.3f}s.] {out.out}")
