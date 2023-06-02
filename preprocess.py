import csv
from typing import NamedTuple, Tuple
from functools import cache


REPLACE = {
    "_trans ": "",
    "_trans": "",
    "_latin": "",
    " sil": "",
    " б ": " бэ ",
    " г ": " гэ ",
    " д ": " дэ ",
    " е ": " йэ ",
    " ё ": " йо ",
    " ж ": " жэ ",
    " з ": " зэ ",
    " к ": " ка ",
    " л ": " эл ",
    " м ": " эм ",
    " н ": " эн ",
    " п ": " пэ ",
    " р ": " эр ",
    " т ": " тэ ",
    " ф ": " фэ ",
    " х ": " ха ",
    " ц ": " цэ ",
    " ч ": " чэ ",
    " ш ": " шэ ",
    " щ ": " ще ",
    " ъ ": " твердый знак ",
    " ь ": " мягкий знак ",
    " ю ": " йю ",
    " я ": " йя ",
    " a ": " эй ",
    " b ": " би ",
    " c ": " си ",
    " d ": " ди ",
    " e ": " и ",
    " f ": " эф ",
    " g ": " джи ",
    " h ": " эйч ",
    " i ": " ай ",
    " j ": " джей ",
    " k ": " кей ",
    " l ": " эль ",
    " m ": " эм ",
    " n ": " эн ",
    " o ": " оу ",
    " p ": " пи ",
    " q ": " ку ",
    " r ": " эр ",
    " s ": " эс ",
    " t ": " ти ",
    " u ": " йю ",
    " v ": " ви ",
    " w ": " дабл йю ",
    " x ": " икс ",
    " y ": " игрик ",
    " z ": " зэт ",
}

KINDS = {
    "DATE",
    "CARDINAL",
    "DIGIT",
    "TIME",
    "LETTERS",
    "FRACTION",
    "PUNCT",
    "ORDINAL",
    "TELEPHONE",
    "MEASURE",
    "DECIMAL",
    "MONEY",
    "ELECTRONIC",
    "VERBATIM",
    "PLAIN",
}


class Token(NamedTuple):
    kind: str
    before: str
    after: str

    def __repr__(self):
        return f'{self.kind}["{self.before}"->"{self.after}"]'


def should_have_space_inbetween(a: Token, b: Token) -> bool:
    return not (
        a.before in {"(", "«"}
        or a.kind == "VERBATIM"
        or b.kind == "VERBATIM"
        or b.kind == "PUNCT"
        and b.before not in {"(", "—", "«"}
    )


class Sentence:
    def __init__(self, id: int, tokens: list[Token]):
        self.id = id
        self._tokens = tokens
        self.before = ""
        self.after = ""
        if len(self._tokens) > 0:
            self.before = self._tokens[0].before
            self.after = self._tokens[0].after
            for prev, cur in zip(self._tokens, self._tokens[1:]):
                if should_have_space_inbetween(prev, cur):
                    self.before += " "
                    self.after += " "
                self.before += cur.before
                self.after += cur.after

        self.len_bytes = max(len(self.before.encode()), len(self.after.encode()))

    @property
    def tokens(self):
        return self._tokens

    def __repr__(self):
        return " ".join(map(repr, self._tokens))


sentences: list[Sentence] = []
token_set: set[Token] = set()
kinds = set()

with open("ru_train.csv", encoding="UTF-8") as f:
    table = csv.reader(f)
    next(table)

    cur_id = 0
    tokens = []
    for line in table:
        id = int(line[0])
        if id != cur_id:
            sentences.append(Sentence(id, tokens))
            cur_id = id
            tokens = []

        kind = line[2]
        before = line[3]
        after = line[4]

        after = " " + after + " "

        for x, y in REPLACE.items():
            after = after.replace(x, y)

        after = after.strip()

        tokens.append(Token(kind, before, after))

        if before != after:
            token_set.add(Token("", before, after))


print(
    f"num sentences: {len(sentences)}\n"
    f"longest sentence length: {max(s.len_bytes for s in sentences)} bytes\n"
    f"num different translating tokens: {len(token_set)}"
)

with open("ru_train_preprocessed.csv", "w", encoding="UTF-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["input_ids", "labels"])
    writer.writerows((s.before, s.after) for s in sentences)

# with open("ru_train_tokens.csv", "w", encoding="UTF-8", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["input_ids", "labels"])
#     writer.writerows((tok.before, tok.after) for tok in token_set)
