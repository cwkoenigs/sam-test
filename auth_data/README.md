# Synthetic authentication transcripts

Programmatically generated agent/customer call transcripts for training a
`bert-base-uncased` chunk classifier that detects how a caller was authenticated.
Everything here is produced by `generate_synthetic_auth_data.py` from a fixed seed;
no real customer data is involved (names, addresses, companies and numbers are all
randomly assembled from generic word lists).

## Files

| File | Conversations | Chunks | ID_AUTH | OTHER_AUTH | NO_AUTH |
|------|--------------:|-------:|--------:|-----------:|--------:|
| `train.jsonl` | 8,000 | 20,373 | 6,400 | 1,600 | 12,373 |
| `validation.jsonl` | 1,000 | 2,513 | 800 | 200 | 1,513 |
| `test.jsonl` | 1,000 | 2,539 | 800 | 200 | 1,539 |
| **total** | **10,000** | **25,425** | **8,000** | **2,000** | **15,425** |

`stats.json` holds the same numbers plus token statistics and the generation
parameters. Splits are made at the **conversation** level (80 / 10 / 10),
stratified by authentication path, so no conversation leaks across splits.

## Conversation design

Every conversation contains exactly one authentication exchange:

* **ID path (80% of conversations)**. The agent asks for an account / member /
  policy number and the customer utters a 9–12 digit identifier. The identifier
  is rendered the way speech-to-text would emit it: raw digits (`472913860`),
  grouped (`472 913 860`), spelled out (`four seven two ... double one`), or a
  mix with disfluencies. Agents sometimes read it back or ask for the last four.
* **No-ID path (20% of conversations)**. The customer explicitly says they do not
  have their ID number (for example *"i don't have my member id with me right
  now"*). The agent then verifies **full name, address, and date of birth**,
  each spoken in varied formats (`march third nineteen eighty two`,
  `3/3/1982`, `oh three oh three ...`, spelled surnames, spoken zip codes).

Around the authentication exchange each call has a greeting, a stated problem
(billing, password reset, outage, cancellation, upgrade, disputed charge,
address change, shipping, autopay, payment arrangement, coverage, device), a
resolution dialogue, and a close. The non-auth parts deliberately contain
**hard negatives**: phone numbers, 6–8 digit order / ticket numbers,
alphanumeric confirmation codes, card last-four, dollar amounts, future dates,
customers introducing themselves by name, and a *new* address being given during
an address change. None of these are authentication and all of them live in
`NO_AUTH` chunks.

Transcripts are lowercase with no sentence punctuation and light fillers, in
the style of ASR output. Speaker turns are prefixed `agent:` / `customer:`.

## Chunking and labels

Conversations are cut into turn-aligned chunks that fit a `bert-base-uncased`
budget. Token counts in the files are **exact** WordPiece counts from the
`bert-base-uncased` vocabulary (special tokens excluded), so adding `[CLS]` and
`[SEP]` keeps every chunk under 200 tokens. Use `max_length=200` (or 256) when
tokenizing; nothing is truncated.

* The per-chunk budget is drawn uniformly from 48–200 tokens so that chunk
  length and chunk position do not leak the label (NO_AUTH chunks cover
  openings and resolution bodies, not just closings).
* An authentication exchange is treated as an atomic unit and is **never split**
  across chunks, so every chunk has one unambiguous label.

| Label | Meaning |
|-------|---------|
| `ID_AUTH` | Chunk contains the spoken 9–12 digit identifier exchange. |
| `OTHER_AUTH` | Chunk contains the "I don't have my ID" exchange and the name / address / DOB verification. |
| `NO_AUTH` | No authentication in the chunk (greeting, issue, resolution, close, distractor numbers). |

## Record schema

```json
{
  "id": "conv_00007_c01",
  "conversation_id": "conv_00007",
  "chunk_index": 1,
  "num_chunks": 4,
  "conversation_auth_type": "ID_AUTH",
  "label": "ID_AUTH",
  "text": "agent: sure thing let me pull up your account do you have your account id handy customer: yeah it's 789085186967 ...",
  "n_turns": 5,
  "bert_tokens": 61
}
```

`conversation_auth_type` is the conversation-level path and is provided for
analysis only. The training target is `label`. A full conversation can be
rebuilt by concatenating its chunks in `chunk_index` order.

## Regenerating

```bash
# exact token counts from the local bert-base-uncased model (what shipped here);
# uses the repo's uv environment, which has transformers installed
uv run python auth_data/generate_synthetic_auth_data.py \
    --tokenizer /Users/sam/models/bert-base-uncased

# same result with only the `tokenizers` package, pointing at the vocab file
uv run --with tokenizers python auth_data/generate_synthetic_auth_data.py \
    --tokenizer /Users/sam/models/bert-base-uncased/vocab.txt

# no dependencies at all: a built-in conservative token estimator is used instead
python3 auth_data/generate_synthetic_auth_data.py
```

Useful flags: `--n` (conversations), `--seed`, `--no-id-frac` (default 0.20),
`--min-tokens` / `--max-tokens` (chunk budget, default 48 / 200),
`--split` (default `0.8,0.1,0.1`), `--out`. Generation is deterministic for a
given seed and tokenizer.
