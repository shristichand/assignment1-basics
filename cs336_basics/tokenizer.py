import regex

from typing import Iterable, Iterator

# GPT-2 pre-tokenization pattern (same as tiktoken gpt2 encoding)
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.bytes_to_id = {
            value: key
            for key, value in vocab.items()
        }

    def encode(self, text: str) -> list[int]:

        # 1. Handle special tokens
        # Find special tokens from left to right.
        # If multiple special tokens start at the same position,
        # choose the longest one.

        if self.special_tokens:

            parts = []
            current_text = ""
            i = 0

            while i < len(text):

                matched_special = None

                for special_token in sorted(
                    self.special_tokens,
                    key=len,
                    reverse=True
                ):
                    if text.startswith(special_token, i):
                        matched_special = special_token
                        break

                if matched_special is not None:

                    # Save normal text before special token
                    if current_text:
                        parts.append(current_text)
                        current_text = ""

                    # Save special token
                    parts.append(matched_special)

                    # Move past the whole special token
                    i += len(matched_special)

                else:
                    current_text += text[i]
                    i += 1

            # Save remaining normal text
            if current_text:
                parts.append(current_text)

        else:
            parts = [text]

        token_ids = []

        # 2. Process each piece
        for part in parts:

            # Special token → directly convert to its ID
            if self.special_tokens and part in self.special_tokens:
                token_ids.append(
                    self.bytes_to_id[part.encode("utf-8")]
                )
                continue

            # 3. Pre-tokenize normal text
            pretokenized = regex.findall(GPT2_PATTERN, part)

            # 4. Process every pre-token separately
            for pretoken in pretokenized:

                # Convert to UTF-8 bytes
                tokens = [
                    bytes([byte])
                    for byte in pretoken.encode("utf-8")
                ]

                # 5. Apply BPE merges
                while len(tokens) > 1:

                    best_pair = None
                    best_rank = None

                    # Find the highest-priority merge that exists
                    for i in range(len(tokens) - 1):

                        pair = (tokens[i], tokens[i + 1])

                        if pair in self.merges:
                            rank = self.merges.index(pair)

                            if best_rank is None or rank < best_rank:
                                best_pair = pair
                                best_rank = rank

                    # No more applicable merges
                    if best_pair is None:
                        break

                    # Apply the best merge
                    new_tokens = []
                    i = 0

                    while i < len(tokens):

                        if (
                            i < len(tokens) - 1
                            and (tokens[i], tokens[i + 1]) == best_pair
                        ):
                            new_tokens.append(
                                tokens[i] + tokens[i + 1]
                            )
                            i += 2

                        else:
                            new_tokens.append(tokens[i])
                            i += 1

                    tokens = new_tokens

                # 6. Convert byte tokens to IDs
                for token in tokens:
                    token_ids.append(self.bytes_to_id[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:

        # Convert token IDs back to bytes
        token_bytes = b"".join(
            self.vocab[token_id]
            for token_id in token_ids
        )

        # Decode UTF-8 bytes into a string
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:

        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
):
    from collections import Counter

    # --------------------------------------------------
    # 1. Read corpus
    # --------------------------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------------------------------------------
    # 2. Initial vocabulary: 256 byte tokens
    # --------------------------------------------------
    vocab = {i: bytes([i]) for i in range(256)}

    # Add special tokens
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges = []

    # --------------------------------------------------
    # 3. Split around special tokens
    # --------------------------------------------------
    parts = [text]

    for special_token in special_tokens:
        new_parts = []

        for part in parts:
            pieces = part.split(special_token)

            for i, piece in enumerate(pieces):
                if piece:
                    new_parts.append(piece)

                if i < len(pieces) - 1:
                    new_parts.append(special_token)

        parts = new_parts

    # --------------------------------------------------
    # 4. Pre-tokenize and count identical sequences
    # --------------------------------------------------
    sequence_counts = Counter()

    for part in parts:

        # Special tokens are already in vocab.
        if part in special_tokens:
            continue

        pretokenized = regex.findall(GPT2_PATTERN, part)

        for pretoken in pretokenized:
            tokens = tuple(
                bytes([byte])
                for byte in pretoken.encode("utf-8")
            )

            if tokens:
                sequence_counts[tokens] += 1

    # --------------------------------------------------
    # 5. Count adjacent pairs
    # --------------------------------------------------
    pair_counts = Counter()

    for tokens, frequency in sequence_counts.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += frequency

    # --------------------------------------------------
    # 6. BPE training
    # --------------------------------------------------
    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        # Highest frequency wins.
        # If frequencies tie, lexicographically greater
        # pair wins.
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        left, right = best_pair
        merged_token = left + right

        # Add merge
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        # --------------------------------------------------
        # 7. Apply merge to UNIQUE sequences
        # --------------------------------------------------
        new_sequence_counts = Counter()

        for tokens, frequency in sequence_counts.items():

            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    new_tokens.append(merged_token)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_sequence_counts[tuple(new_tokens)] += frequency

        sequence_counts = new_sequence_counts

        # --------------------------------------------------
        # 8. Recalculate pair counts
        #    but only over UNIQUE sequences
        # --------------------------------------------------
        pair_counts = Counter()

        for tokens, frequency in sequence_counts.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += frequency

    return vocab, merges