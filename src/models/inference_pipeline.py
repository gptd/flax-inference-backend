from typing import List, Optional, Tuple

from math import ceil
import jax
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    FlaxAutoModelForSequenceClassification,
)

from ..data.preprocessing import split_paragraph

from tqdm.auto import tqdm


class FlaxSequenceClassificationPipeline:
    """
    Sequence classification pipeline for Flax HuggingFace
    models. Scales to more than one accelerators on a single host
    for faster inference.
    """

    def __init__(
        self,
        hf_model_name: str,
        hf_tokenizer_name: str,
        batch_size: int,
        max_length: int,
        devices=jax.devices(),
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

        self.batch_size = batch_size
        self.max_length = max_length

        try:
            self.model = FlaxAutoModelForSequenceClassification.from_pretrained(
                hf_model_name
            )
        except OSError:
            self.model = FlaxAutoModelForSequenceClassification.from_pretrained(
                hf_model_name, from_pt=True  # Load PyTorch model params and convert.
            )

        # Replicate params across all local accelerators to speed up inference.
        sharding_scheme = jax.sharding.PositionalSharding(devices).replicate()
        self.params = jax.device_put(self.model.params, sharding_scheme)

        # Warm up: Precompile
        print("Precompiling JIT pipeline.")
        _ = self.predict("Example", 1)

    def _tokenize(self, texts: List[str]) -> BatchEncoding:
        """
        Blank entries are added to the output
        to ensure uniform batch size (avoids JIT recompilation.)
        """
        if len(texts) % self.batch_size != 0:
            padding_entries_required = self.batch_size - (len(texts) % self.batch_size)
            texts += [""] * padding_entries_required

        output = self.tokenizer(
            texts,
            max_length=self.max_length,
            return_tensors="jax",
            padding="max_length",
            truncation=True,
        )

        return output

    def predict(
        self, texts: str, stride: int
    ) -> Tuple[List[Tuple[float, ...]], List[str]]:
        split_texts = split_paragraph(texts, stride)
        predictions: List[Tuple[float, ...]] = []
        num_entries = len(split_texts)
        tokenized_padded_texts = self._tokenize(split_texts)

        num_batches: int = ceil(num_entries / self.batch_size)

        padded_predictions: List[Tuple[float, ...]] = []
        for batch_index in tqdm(range(num_batches)):
            index_a = batch_index * self.batch_size
            index_b = index_a + self.batch_size

            batch = {
                "input_ids": tokenized_padded_texts.input_ids[index_a:index_b],
                "attention_mask": tokenized_padded_texts.attention_mask[
                    index_a:index_b
                ],
            }

            model_output = self.model(batch["input_ids"], batch["attention_mask"])
            model_predictions = jax.nn.softmax(model_output.logits, axis=-1)

            for entry_index in range(self.batch_size):
                prediction = model_predictions[entry_index, :].flatten().tolist()
                padded_predictions.append(prediction)

        assert len(padded_predictions) >= num_entries
        for entry_index in range(num_entries):
            prediction = padded_predictions[entry_index]
            predictions.append(prediction)

        return predictions, split_texts

    def __call__(
        self, texts: str, stride: int
    ) -> Tuple[List[Tuple[float, ...]], List[str]]:
        return self.predict(texts, stride)
