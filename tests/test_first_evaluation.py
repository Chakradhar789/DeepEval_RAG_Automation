from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel


def test_answer_relevancy():

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="paris is the capital of France."
    )

    ollama_model = OllamaModel(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
        model=ollama_model
    )

    assert_test(
        test_case=test_case,
        metrics=[metric]
    )