# 1. What is Contextual Recall?
# Contextual Recall measures whether the retrieved context contains the information needed to support the expected answer.
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import ContextualRecallMetric


def contextual_recall_metric(
    input_question,
    actual_answer,
    expected_answer,
    retrieval_context
):

    # 1. Create DeepEval test case
    test_case = LLMTestCase(
        input=input_question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context
    )

    # 2. Configure Ollama as LLM Judge
    ollama_model = OllamaModel(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0
    )

    # 3. Create Contextual Recall metric
    metric = ContextualRecallMetric(
        threshold=0.7,
        include_reason=True,
        model=ollama_model
    )

    # 4. Run evaluation
    metric.measure(test_case)

    # 5. Display result
    print("\n" + "=" * 60)
    print("CONTEXTUAL RECALL RESULT")
    print("=" * 60)

    print("Question          :", input_question)
    print("Actual Answer     :", actual_answer)
    print("Expected Answer   :", expected_answer)
    print("Retrieved Context :", retrieval_context)
    print("Score             :", metric.score)
    print("Threshold         :", metric.threshold)
    print("Reason            :", metric.reason)

    # 6. Assertion
    assert metric.score >= metric.threshold, (
        f"Contextual Recall FAILED | "
        f"Score={metric.score} | "
        f"Threshold={metric.threshold} | "
        f"Reason={metric.reason}"
    )


# =========================================================
# TEST 1 - COMPLETE RETRIEVED CONTEXT
# =========================================================

def test_complete_retrieved_context():

    contextual_recall_metric(
        input_question=(
            "How many days of paternity leave are provided, "
            "and when must the leave be taken?"
        ),

        actual_answer=(
            "Eligible employees receive 15 days of paternity leave, "
            "which must be taken within six months of the child's birth."
        ),

        expected_answer=(
            "Eligible employees receive 15 days of paternity leave. "
            "The leave must be taken within six months of the child's birth."
        ),

        retrieval_context=[
            "Eligible employees are entitled to 15 days of paternity leave.",
            "The leave must be taken within six months of the child's birth."
        ]
    )


# =========================================================
# TEST 2 - INCOMPLETE RETRIEVED CONTEXT
# =========================================================

def test_incomplete_retrieved_context():

    contextual_recall_metric(
        input_question=(
            "How many days of paternity leave are provided, "
            "and when must the leave be taken?"
        ),

        actual_answer=(
            "Eligible employees receive 15 days of paternity leave."
        ),

        expected_answer=(
            "Eligible employees receive 15 days of paternity leave. "
            "The leave must be taken within six months of the child's birth."
        ),

        retrieval_context=[
            "Eligible employees are entitled to 90 days of paternity leave."
        ]
    )
