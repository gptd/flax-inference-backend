import unittest
import os

from ..models.inference_pipeline import FlaxSequenceClassificationPipeline
from ..data.preprocessing import split_paragraph

HF_MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME", "Hello-SimpleAI/chatgpt-detector-roberta"
)
HF_TOKENIZER_NAME = os.environ.get(
    "HF_TOKENIZER_NAME", "Hello-SimpleAI/chatgpt-detector-roberta"
)


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prediction_pipeline = FlaxSequenceClassificationPipeline(
            HF_MODEL_NAME, HF_TOKENIZER_NAME, 16, 128
        )

    def setUp(self):
        self.prediction_pipeline = InferenceTest.prediction_pipeline
        self.example_paragraph = (
            """
        We make AI more accessible by hosting bootcamps and workshops that give students the opportunity to learn about AI in a peer-based, supportive environment.

        We want to make AI fun and welcoming. Our initiatives focus on giving participants the opportunity to meet others interested in AI.

        Montreal is one of the world's leading AI hubs. Through our hackathon, learnathon, and industry events, we help connect McGill students with Montreal's AI ecosystem.
        """
            * 12
        )

    def test_inference_pipeline_output_shape(self):
        output = self.prediction_pipeline(self.example_paragraph, 6)
        print(output)
