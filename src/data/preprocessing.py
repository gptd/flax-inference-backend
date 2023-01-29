import re
from typing import List


def split_paragraph(paragraph: str, stride: int) -> List[str]:
    """
    Given a paragraph of text, split the paragraph
    by "." and group output into batches of `stride`.
    """
    delimiter_pattern = re.compile(r"\.\s")

    texts_raw = delimiter_pattern.split(paragraph)
    texts_buffer: List[str] = []
    texts_processed: List[str] = []

    for text in texts_raw:
        formatted_text = text.lstrip().rstrip()

        if len(formatted_text) >= 1:
            texts_buffer.append(formatted_text)

        if len(texts_buffer) >= stride:
            texts_processed.append(". ".join(texts_buffer) + ".")
            texts_buffer.clear()

    if len(texts_buffer) > 0:
        texts_processed.append(".".join(texts_buffer) + ".")

    return texts_processed
