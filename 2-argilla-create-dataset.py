# https://python.langchain.com/v0.1/docs/integrations/callbacks/argilla/

import argilla as rg
import os

ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
ARGILLA_WORKSPACE = os.getenv("ARGILLA_WORKSPACE")
ARGILLA_DATASET_NAME = os.getenv("ARGILLA_DATASET_NAME")

rg.init(
    api_url=ARGILLA_API_URL,
    api_key=ARGILLA_API_KEY,
)

dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="prompt"),
        rg.TextField(name="response"),
    ],
    questions=[
        rg.RatingQuestion(
            name="response-rating",
            description="How would you rate the quality of the response?",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.TextQuestion(
            name="response-feedback",
            description="What feedback do you have for the response?",
            required=False,
        ),
    ],
    guidelines="You're asked to rate the quality of the response and provide feedback.",
)

dataset.push_to_argilla(ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE)
