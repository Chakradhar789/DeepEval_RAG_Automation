
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import ContextualPrecisionMetric


def contextual_precision_metric(
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

    # 3. Create Contextual Precision metric
    metric = ContextualPrecisionMetric(
        threshold=0.7,
        include_reason=True,
        model=ollama_model
    )

    # 4. Run evaluation
    metric.measure(test_case)

    # 5. Display result
    print("\n" + "=" * 60)
    print("CONTEXTUAL PRECISION RESULT")
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
        f"Contextual Precision FAILED | "
        f"Score={metric.score} | "
        f"Threshold={metric.threshold} | "
        f"Reason={metric.reason}"
    )


# =========================================================
# TEST 1 - RELEVANT CHUNK RANKED FIRST
# =========================================================

def test_relevant_chunk_ranked_first():

    contextual_precision_metric(
        input_question=(
            "What is the company's work-from-home policy?"
        ),

        actual_answer=(
            "Employees can work from home two days per week "
            "with manager approval."
        ),

        expected_answer=(
            "Employees can work from home two days per week "
            "with manager approval."
        ),

        retrieval_context=[
            "Employees can work from home two days per week "
            "with manager approval.",
            "The company cafeteria serves breakfast and lunch.",
            "The company provides medical insurance to employees."
        ]
    )


# =========================================================
# TEST 2 - RELEVANT CHUNK RANKED LOWER
# =========================================================

def test_relevant_chunk_ranked_lower():

    contextual_precision_metric(
        input_question=(
            "What is the company's work-from-home policy?"
        ),

        actual_answer=(
            "Employees can work from home two days per week "
            "with manager approval."
        ),

        expected_answer=(
            "Employees can work from home two days per week "
            "with manager approval."
        ),

        retrieval_context=[
            "The company cafeteria serves breakfast and lunch.",
            "The company provides medical insurance to employees.",
            "Employees can work from home two days per week "
            "with manager approval."
        ]
    )
