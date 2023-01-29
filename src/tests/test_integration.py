import unittest
import requests


class IntegrationTest(unittest.TestCase):
    def test_predict(self):
        self.example_paragraph = """
        We make AI more accessible by hosting bootcamps and workshops that give students the opportunity to learn about AI in a peer-based, supportive environment.

        We want to make AI fun and welcoming. Our initiatives focus on giving participants the opportunity to meet others interested in AI.

        Montreal is one of the world's leading AI hubs. Through our hackathon, learnathon, and industry events, we help connect McGill students with Montreal's AI ecosystem.
        """
        response = requests.post(
            "https://api.gptd.me/predict", json={"query_text": self.example_paragraph}
        )

        print(response.json())
