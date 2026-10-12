
# =========================================================
# ANSWER CORRECTNESS - DEEPEVAL + OLLAMA
# Answer Correctness checks whether the actual answer
# matches the expected answer in factual meaning.
# =========================================================

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import OllamaModel
from deepeval.metrics import GEval


def answer_correctness_metric(
    input_question,
    actual_answer,
    expected_answer
):

    # 1. Create the DeepEval test case
    test_case = LLMTestCase(
        input=input_question,
        actual_output=actual_answer,
        expected_output=expected_answer
    )

    # 2. Configure Ollama as the LLM judge
    ollama_model = OllamaModel(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0
    )

    # 3. Create the custom Answer Correctness metric
    metric = GEval(
        name="Answer Correctness",
        criteria=(
            "Evaluate the factual correctness of the actual answer "
            "by comparing it with the expected answer. Penalize "
            "contradictions and missing important facts. Accept "
            "paraphrases that preserve the same meaning."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.7,
        model=ollama_model
    )

    # 4. Run the evaluation
    metric.measure(test_case)

    # 5. Display the result
    print("\n" + "=" * 60)
    print("ANSWER CORRECTNESS RESULT")
    print("=" * 60)

    print("Question        :", input_question)
    print("Actual Answer   :", actual_answer)
    print("Expected Answer :", expected_answer)
    print("Score           :", metric.score)
    print("Threshold       :", metric.threshold)
    print("Reason          :", metric.reason)

    # 6. Assert the result against the threshold
    assert metric.score >= metric.threshold, (
        f"Answer Correctness FAILED | "
        f"Score={metric.score} | "
        f"Threshold={metric.threshold} | "
        f"Reason={metric.reason}"
    )


# =========================================================
# TEST 1 - CORRECT ANSWER
# Expected: Should pass if the judge scores it >= 0.7
# =========================================================

def test_correct_answer():

    answer_correctness_metric(
        input_question="How many days of paternity leave are provided?",
        actual_answer="Eligible employees receive 15 days of paternity leave.",
        expected_answer="Eligible employees are entitled to 15 days of paternity leave."
    )


# =========================================================
# TEST 2 - INCORRECT ANSWER
# Expected: Should fail if the judge scores it below 0.7
# =========================================================

def test_incorrect_answer():

    answer_correctness_metric(
        input_question="How many days of paternity leave are provided?",
        actual_answer="Eligible employees receive 30 days of paternity leave.",
        expected_answer="Eligible employees are entitled to 15 days of paternity leave."
    )
