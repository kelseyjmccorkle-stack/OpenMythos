from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> "MythosTokenizer":
    """Construct a MythosTokenizer (convenience wrapper).

    Args:
        model_id: HuggingFace tokenizer id or local path.

    Returns:
        A ready MythosTokenizer.
    """
    return MythosTokenizer(model_id)


def get_vocab_size(model_id: str = DEFAULT_MODEL_ID) -> int:
    """Return the vocabulary size of a tokenizer.

    Use this to size a model's embedding / output head so it matches the
    tokenizer that will actually feed it — the preset configs hardcode
    ``vocab_size=32000`` while the default tokenizer's vocabulary is much
    larger, so building a model without reconciling the two produces
    out-of-range token ids at runtime.

    Args:
        model_id: HuggingFace tokenizer id or local path.

    Returns:
        The tokenizer's vocabulary size.
    """
    return MythosTokenizer(model_id).vocab_size


def configure_vocab_size(cfg, model_id: str = DEFAULT_MODEL_ID):
    """Set ``cfg.vocab_size`` to match a tokenizer and return the config.

    Prevents the silent mismatch between a model's embedding/head size and
    the tokenizer's vocabulary. ``cfg`` is mutated in place and also
    returned for chaining, e.g.::

        cfg = configure_vocab_size(mythos_1b())
        model = OpenMythos(cfg)

    Args:
        cfg: any object with a writable ``vocab_size`` attribute (e.g.
            MythosConfig); typed loosely to avoid importing the model here.
        model_id: HuggingFace tokenizer id or local path.

    Returns:
        The same ``cfg`` with ``vocab_size`` updated.
    """
    cfg.vocab_size = get_vocab_size(model_id)
    return cfg
