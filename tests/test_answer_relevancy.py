from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric


def evaluate_answer(input_question, actual_answer):

    # Create test case
    test_case = LLMTestCase(
        input=input_question,
        actual_output=actual_answer
    )

    # Ollama LLM Judge
    ollama_model = OllamaModel(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0
    )

    # Answer Relevancy Metric
    metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
        model=ollama_model
    )

    # Run metric
    metric.measure(test_case)

    # Display result
    print("\n" + "=" * 60)
    print("ANSWER RELEVANCY RESULT")
    print("=" * 60)
    print("Question  :", input_question)
    print("Answer    :", actual_answer)
    print("Score     :", metric.score)
    print("Threshold :", metric.threshold)
    print("Reason    :", metric.reason)

    # =====================================================
    # ASSERT
    # =====================================================
    assert metric.score >= metric.threshold, (
        f"Answer Relevancy FAILED | "
        f"Score={metric.score} | "
        f"Threshold={metric.threshold} | "
        f"Reason={metric.reason}"
    )

    print("RESULT    : PASS")


# =========================================================
# TEST 1 - CORRECT ANSWER
# =========================================================

def test_correct_answer():

    evaluate_answer(
        input_question="How many days of paternity leave can an eligible employee take?",
        actual_answer="An eligible employee can take 15 days of paternity leave."
    )


# =========================================================
# TEST 2 - IRRELEVANT ANSWER
# =========================================================

def test_wrong_answer():

    evaluate_answer(
        input_question="How many days of paternity leave can an eligible employee take?",
        actual_answer="The weather is sunny today."
    )


# =========================================================
# TEST 3 - COMPLETELY UNRELATED ANSWER
# =========================================================

def test_unrelated_answer():

    evaluate_answer(
        input_question="How many days of paternity leave can an eligible employee take?",
        actual_answer="Hi everyone, I am Chakradhar from Hyderabad."
    )
