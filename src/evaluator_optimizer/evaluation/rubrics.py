"""
Evaluation rubrics used by the Evaluator LLM.

The rubric converts an abstract evaluation metric into explicit
judging criteria. This reduces ambiguity in LLM-as-a-Judge scoring.
"""

from evaluator_optimizer.evaluation.metrics import EvaluationMetric


EVALUATION_RUBRICS: dict[EvaluationMetric, str] = {
    EvaluationMetric.RELEVANCE: """
Evaluate whether the answer directly addresses the user's question.

Consider:
- Does the response answer the requested question?
- Does it focus on information relevant to the request?
- Does it avoid unnecessary unrelated information?
- Does it address the requested company, metric, period, or comparison?

Score:
1.0 = Directly and completely relevant.
0.8 = Relevant with minor unnecessary content.
0.5 = Partially relevant or contains notable irrelevant content.
0.3 = Mostly unrelated or fails to address important parts.
0.0 = Does not answer the question.
""",

    EvaluationMetric.GROUNDEDNESS: """
Evaluate whether the answer is supported by the supplied context.

Consider:
- Are important claims supported by retrieved evidence?
- Does the answer avoid unsupported claims?
- Are numerical statements traceable to the supplied context?
- Does the answer avoid inventing facts not present in the evidence?

Score:
1.0 = Claims are strongly supported by the supplied evidence.
0.8 = Nearly all claims are supported with minor gaps.
0.5 = Several claims have incomplete support.
0.3 = Important claims are unsupported.
0.0 = The answer is substantially unsupported by the context.
""",

    EvaluationMetric.ACCURACY: """
Evaluate whether the answer is factually consistent with the supplied
evidence.

Consider:
- Are numerical values correct?
- Are calculations and comparisons correct?
- Are company facts represented correctly?
- Are conclusions consistent with the evidence?
- Are there contradictions between the answer and context?

Do not penalize the answer merely because the evidence itself is
incomplete; evaluate consistency with the supplied evidence.

Score:
1.0 = Factually consistent with the evidence.
0.8 = Minor inaccuracies that do not materially affect the answer.
0.5 = Meaningful inaccuracies are present.
0.3 = Significant factual or numerical errors exist.
0.0 = The answer substantially contradicts the evidence.
""",

    EvaluationMetric.TEMPORAL_VALIDITY: """
Evaluate whether the evidence and claims are temporally appropriate
for the user's question.

Consider:
- Requested reporting period.
- Evidence reporting period.
- Publication/evidence date.
- Whether the user asked for latest/current information.
- Whether historical information is appropriate for the question.
- Whether the answer mixes reporting periods without making the
  distinction clear.
- Whether stale evidence is being used for a current question.

Important:
Historical evidence is not automatically invalid. For example,
a question asking for FY2020 revenue should legitimately use FY2020
evidence.

However, using FY2020 evidence to answer a question asking for the
latest annual revenue would be temporally inappropriate when newer
evidence is required.

Score:
1.0 = Evidence and claims are fully appropriate for the requested
      time period.
0.8 = Minor temporal ambiguity exists.
0.5 = Meaningful temporal limitation exists.
0.3 = Significant stale or inappropriate evidence is used.
0.0 = Evidence is materially temporally invalid for the question.
""",

    EvaluationMetric.CLARITY: """
Evaluate whether the answer is clear and understandable.

Consider:
- Directness.
- Logical organization.
- Precision.
- Readability.
- Ambiguity.
- Appropriate terminology.
- Whether reporting periods and numbers are clearly identified.
- Whether the response avoids unnecessary repetition.

Score:
1.0 = Exceptionally clear, precise, and well organized.
0.8 = Clear with minor presentation issues.
0.5 = Understandable but contains ambiguity or weak organization.
0.3 = Difficult to follow or materially ambiguous.
0.0 = Substantially unclear or incomprehensible.
""",
}