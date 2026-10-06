from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


def test_all_rag_metrics():

    # ============================================================
    # TEST CASE
    # ============================================================

    test_case = LLMTestCase(

        input="How many days of paternity leave can an eligible employee take?",

        actual_output=(
            "Eligible employees can take 15 days of paternity leave. "
            "The leave must be taken within 6 months of the child's birth."
        ),

        expected_output=(
            "Eligible employees are entitled to 15 days of paternity leave, "
            "which must be taken within 6 months of the child's birth."
        ),

        retrieval_context=[
            (
                "Paternity Leave Policy: Eligible employees are entitled "
                "to 15 days of paternity leave."
            ),

            (
                "The paternity leave must be taken within 6 months "
                "from the date of the child's birth."
            ),

            (
                "Employees may apply for annual leave according "
                "to the organization's annual leave policy."
            )
        ]
    )

    # ============================================================
    # OLLAMA JUDGE MODEL
    # ============================================================

    ollama_model = OllamaModel(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0
    )

    # ============================================================
    # METRICS
    # ============================================================

    metrics = [

        AnswerRelevancyMetric(
            threshold=0.7,
            include_reason=True,
            model=ollama_model
        ),

        FaithfulnessMetric(
            threshold=0.7,
            include_reason=True,
            model=ollama_model
        ),

        ContextualRelevancyMetric(
            threshold=0.7,
            include_reason=True,
            model=ollama_model
        ),

        ContextualPrecisionMetric(
            threshold=0.7,
            include_reason=True,
            model=ollama_model
        ),

        ContextualRecallMetric(
            threshold=0.7,
            include_reason=True,
            model=ollama_model
        )
    ]

    # ============================================================
    # EXECUTE ALL METRICS
    # ============================================================

    print("\n")
    print("=" * 80)
    print("                 DEEPEVAL RAG EVALUATION")
    print("=" * 80)

    results = []

    for metric in metrics:

        metric_name = metric.__class__.__name__

        print("\n" + "-" * 80)
        print(f"METRIC: {metric_name}")
        print("-" * 80)

        try:

            # Run the metric
            metric.measure(test_case)

            score = metric.score
            threshold = metric.threshold
            reason = metric.reason

            # Determine status
            if score >= threshold:
                status = "PASS"
            else:
                status = "FAIL"

            # Display result
            print(f"Score     : {score}")
            print(f"Threshold : {threshold}")
            print(f"Status    : {status}")
            print(f"Reason    : {reason}")

            results.append({
                "metric": metric_name,
                "score": score,
                "threshold": threshold,
                "status": status
            })

        except Exception as error:

            print(f"ERROR     : {error}")

            results.append({
                "metric": metric_name,
                "score": None,
                "threshold": metric.threshold,
                "status": "ERROR"
            })

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n")
    print("=" * 80)
    print("                         SUMMARY")
    print("=" * 80)

    print(
        f"{'Metric':35} "
        f"{'Score':10} "
        f"{'Threshold':12} "
        f"{'Status'}"
    )

    print("-" * 80)

    for result in results:

        score = (
            f"{result['score']:.2f}"
            if result["score"] is not None
            else "ERROR"
        )

        print(
            f"{result['metric']:35} "
            f"{score:10} "
            f"{result['threshold']:<12} "
            f"{result['status']}"
        )

    print("=" * 80)
