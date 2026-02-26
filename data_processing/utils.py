"""Encoding and decoding utilities for base-4 and zero-width steganography."""


def _to_base4(n, width=0):
    """Convert an integer to base-4 string with optional zero-padding."""
    base4 = ""
    while n > 0:
        base4 = str(n % 4) + base4
        n //= 4
    base4 = base4.zfill(width)
    return base4


def encode_base4(text, separator="|"):
    """
    Encode text to base-4 digits (plain, human-readable).

    Example: "Hi" -> "0110|0210"
    """
    quant_chars = [_to_base4(ord(c), width=4) for c in text]
    return separator.join(quant_chars)


def encode_base4_steganography(text, separator="|"):
    """
    Encode text to base-4, then map each digit to zero-width Unicode chars.

    Mapping:
        0 -> U+200B (zero-width space)
        1 -> U+200C (zero-width non-joiner)
        2 -> U+200D (zero-width joiner)
        3 -> U+2060 (word joiner)
        | -> U+2062 (invisible times)
    """
    if separator != "|":
        raise NotImplementedError("Only separator='|' is supported.")
    zero_width_map = {
        "0": "\u200b",
        "1": "\u200c",
        "2": "\u200d",
        "3": "\u2060",
        "|": "\u2062",
    }
    quant_chars = [_to_base4(ord(c), width=4) for c in text]
    hidden_bits = separator.join(quant_chars)
    return "".join(zero_width_map[b] for b in hidden_bits)


def decode_base4_steganography(encoded_data, separator="|"):
    """
    Decode a zero-width steganography string back to the original text.

    Reverse mapping:
        U+200B -> 0, U+200C -> 1, U+200D -> 2, U+2060 -> 3, U+2062 -> |
    """
    reverse_map = {
        "\u200b": "0",
        "\u200c": "1",
        "\u200d": "2",
        "\u2060": "3",
        "\u2062": separator,
    }
    decoded_digits = "".join(reverse_map[c] for c in encoded_data if c in reverse_map)
    base4_values = decoded_digits.split(separator)

    hidden_text = ""
    for base4_str in base4_values:
        if not base4_str:
            continue
        try:
            hidden_text += chr(int(base4_str, 4))
        except (ValueError, OverflowError):
            continue
    return hidden_text
