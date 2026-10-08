from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric


def evaluate_faithfulness(
    input_question,
    actual_answer,
    retrieval_context
):

    # 1. Create DeepEval test case
    test_case = LLMTestCase(
        input=input_question,
        actual_output=actual_answer,
        retrieval_context=retrieval_context
    )

    # 2. Configure Ollama as LLM Judge
    ollama_model = OllamaModel(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0
    )

    # 3. Create Faithfulness metric
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
        model=ollama_model
    )

    # 4. Run Faithfulness evaluation
    metric.measure(test_case)

    # 5. Display result
    print("\n" + "=" * 60)
    print("FAITHFULNESS RESULT")
    print("=" * 60)

    print("Question          :", input_question)
    print("Actual Answer     :", actual_answer)
    print("Retrieved Context :", retrieval_context)
    print("Score             :", metric.score)
    print("Threshold         :", metric.threshold)
    print("Reason            :", metric.reason)

    # 6. ASSERT
    assert metric.score >= metric.threshold, (
        f"Faithfulness FAILED | "
        f"Score={metric.score} | "
        f"Threshold={metric.threshold} | "
        f"Reason={metric.reason}"
    )


# =========================================================
# TEST 1 - FAITHFUL ANSWER
# =========================================================

def test_faithful_answer():

    evaluate_faithfulness(

        input_question=(
            "How many days of paternity leave can an eligible employee take?"
        ),

        actual_answer=(
            "An eligible employee can take 15 days of paternity leave."
        ),

        retrieval_context=[
            (
                "Eligible employees are entitled to 15 days of "
                "paternity leave."
            ),
            (
                "The leave must be taken within six months "
                "of the child's birth."
            )
        ]
    )


# =========================================================
# TEST 2 - UNFAITHFUL ANSWER
# =========================================================

def test_unfaithful_answer():

    evaluate_faithfulness(

        input_question=(
            "How many days of paternity leave can an eligible employee take?"
        ),

        actual_answer=(
            "An eligible employee can take 30 days of paternity leave."
        ),

        retrieval_context=[
            (
                "Eligible employees are entitled to 15 days of "
                "paternity leave."
            ),
            (
                "The leave must be taken within six months "
                "of the child's birth."
            )
        ]
    )
