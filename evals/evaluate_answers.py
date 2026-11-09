
import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")


# --------------------------------------------------
# 2. Initialize the evaluation model
# --------------------------------------------------

judge = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1200,
    reasoning_effort="low",
    reasoning_format="hidden",
    max_retries=2,
    timeout=60
)


# --------------------------------------------------
# 3. Load generation results
# --------------------------------------------------

INPUT_FILE = PROJECT_ROOT / "evals" / "generation_results.json"
OUTPUT_FILE = PROJECT_ROOT / "evals" / "answer_evaluation.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

# Load previous evaluation results if they exist
existing_results = []

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        previous_data = json.load(file)

    existing_results = previous_data.get("results", [])

# Store previous evaluations using question ID
existing_by_id = {
    item["id"]: item
    for item in existing_results
    if "id" in item
}

print("\nCheckpoint status:")

for item in results:
    question_id = item["id"]
    previous = existing_by_id.get(question_id)

    if previous and isinstance(previous.get("scores"), dict):
        print(f"{question_id}: Already evaluated - will skip")
    else:
        print(f"{question_id}: Not evaluated successfully - will retry")
        
# --------------------------------------------------
# 4. Evaluation prompt
# --------------------------------------------------



EVALUATION_PROMPT = """
You are a strict evaluator for a YouTube transcript-based RAG chatbot.

Use ONLY the retrieved transcript chunks as evidence.

The expected answer is provided ONLY to assess correctness.
It must NEVER be used as evidence for faithfulness.

Evaluate the ENTIRE generated answer, from its first sentence
to its final sentence.

IMPORTANT: Do not skip introductory sentences, summaries,
examples, instructions, or statements outside numbered lists.

SCORING:

1. Correctness:
5 = Fully correct.
4 = Correct with a minor omission.
3 = Partially correct.
2 = Mostly incorrect.
1 = Completely incorrect.

2. Faithfulness:
5 = Every factual claim is supported by retrieved evidence.
4 = Mostly supported, with minor unsupported details.
3 = Contains meaningful unsupported claims.
2 = Contains a significant contradiction or multiple unsupported claims.
1 = Mostly fabricated or substantially contradicts evidence.

3. Relevance:
5 = Directly answers the question.
4 = Relevant with minor unnecessary details.
3 = Partially relevant.
2 = Mostly irrelevant.
1 = Does not answer the question.

MANDATORY CLAIM EXTRACTION:

1. Read the complete generated answer.
2. Extract EVERY factual claim, including the opening sentence.
3. Do not combine contradictory statements into one claim.
4. Check each claim against the retrieved chunks.
5. Provide an exact supporting quotation where available.
6. Classify each claim as:
   - supported
   - unsupported
   - contradicted

A claim is contradicted when the retrieved transcript
states something incompatible with the generated answer.

IMPORTANT EXAMPLE:

Retrieved transcript:
"Set a bench to a slight incline. Lay your chest flat against it."

Generated answer:
"Lie face-up on a slightly inclined bench."

Verdict:
"contradicted"

Reason:
The transcript instructs the person to place their chest
against the bench. The generated answer says to lie face-up.
These are incompatible positions.

A significant contradiction MUST result in a faithfulness
score no higher than 2.

Do not overlook contradictions just because another sentence
in the answer correctly describes the technique.

Do not mark a claim supported merely because it sounds
plausible or because other parts of the answer are correct.

Evidence quotes must be copied from the retrieved chunks.
Never invent or paraphrase an evidence quote.

If a claim has no supporting evidence, use an empty
evidence_quote and classify it as unsupported, unless
the transcript directly contradicts it.

TIMESTAMP RULE:

Do not assume a timestamp is correct merely because it
looks plausible. Evaluate it only if the retrieved evidence
provides enough timestamp information to verify it.

ABSTENTION:

If the question is unanswerable:
- Score 5 if the chatbot clearly says the information
  is unavailable or not specified.
- Score 1 if it invents an answer.
- Use intermediate scores for partial abstention.

If answerable, abstention_score must be null.

Return ONLY valid JSON in this exact format:

{{
    "correctness": 1,
    "faithfulness": 1,
    "relevance": 1,
    "abstention_score": null,
    "claim_checks": [
        {{
            "claim": "Exact factual claim from the answer",
            "verdict": "supported",
            "evidence_quote": "Exact transcript quotation",
            "explanation": "Why the evidence supports or contradicts it"
        }}
    ],
    "reason": "Explain the scores and identify any contradictions."
}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

IS THE QUESTION ANSWERABLE?
{answerable}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CHUNKS:
{retrieved_chunks}
"""




# --------------------------------------------------
# 5. Helper to parse judge response
# --------------------------------------------------

def parse_json_response(response):
    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    content = str(content).strip()

    # Remove markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    return json.loads(content)

def evaluate_abstention(answer):
    answer = answer.lower()

    abstention_phrases = [
        "does not provide",
        "not provided",
        "not specified",
        "not mentioned",
        "no information",
        "cannot determine",
        "is unavailable",
        "doesn't mention"
    ]

    if any(phrase in answer for phrase in abstention_phrases):
        return 5

    return 1


def apply_regression_checks(item, scores, retrieved_chunks):
    """
    Deterministic checks for known contradiction patterns.
    These checks supplement the LLM judge.
    """

    question = item["question"].lower()
    answer = item["generated_answer"].lower()

    # Normalize different Unicode hyphens
    answer = (
        answer
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )

    transcript = " ".join(
        chunk.get("content", "")
        for chunk in retrieved_chunks
    ).lower()

    # Regression check:
    # Chest-supported lateral raises require the chest
    # to be placed against the bench.
    is_lateral_raise_question = (
        "chest-supported lateral raise" in question
        or "chest supported lateral raise" in question
    )

    says_face_up = any(
        phrase in answer
        for phrase in [
            "face-up",
            "face up",
            "lying on your back",
            "lie on your back",
            "supine"
        ]
    )

    transcript_requires_chest_support = (
        "chest flat against" in transcript
        or "chest against the bench" in transcript
        or "chest against it" in transcript
    )

    if (
        is_lateral_raise_question
        and says_face_up
        and transcript_requires_chest_support
    ):
        scores["faithfulness"] = min(
            scores["faithfulness"], 2
        )

        scores["claim_checks"].append({
            "claim": (
                "The generated answer instructs the user "
                "to lie face-up."
            ),
            "verdict": "contradicted",
            "evidence_quote": (
                "Lay your chest flat against it."
            ),
            "explanation": (
                "The generated answer says to lie face-up, "
                "but the retrieved transcript instructs the "
                "user to place their chest against the bench. "
                "This is a significant contradiction."
            )
        })

        scores["reason"] += (
            " Python regression check: detected a "
            "face-up instruction conflicting with the "
            "chest-supported position. Faithfulness capped at 2."
        )

    return scores


# --------------------------------------------------
# 6. Evaluate one answer
# --------------------------------------------------

def evaluate_answer(item):

    retrieved_chunks = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "timestamp": chunk.get("timestamp"),
            "content": chunk.get("content", "")
        }
        for chunk in item.get("retrieved_chunks", [])
    ]

    prompt = EVALUATION_PROMPT.format(
        question=item["question"],
        expected_answer=item["expected_answer"],
        answerable=item["answerable"],
        generated_answer=item["generated_answer"],
        retrieved_chunks=json.dumps(
            retrieved_chunks,
            ensure_ascii=False,
            indent=2
        )
    )

    response = judge.invoke(prompt)
    scores = parse_json_response(response)

    
    required_fields = [
        "correctness",
        "faithfulness",
        "relevance",
        "abstention_score",
        "claim_checks",
        "reason"
        ]


    for field in required_fields:
        if field not in scores:
            raise ValueError(f"Missing evaluation field: {field}")

    for field in ["correctness", "faithfulness", "relevance"]:
        if not isinstance(scores[field], (int, float)):
            raise ValueError(f"Invalid score for {field}")

        if not 1 <= scores[field] <= 5:
            raise ValueError(f"Score out of range for {field}")

    
    # --------------------------------------------------
    # Enforce claim-level faithfulness checks
    # --------------------------------------------------

    claim_checks = scores["claim_checks"]

    if not isinstance(claim_checks, list):
        raise ValueError("claim_checks must be a list")

    for claim in claim_checks:
        if not isinstance(claim, dict):
            raise ValueError("Invalid claim check")

        if claim.get("verdict") not in [
            "supported",
            "unsupported",
            "contradicted"
        ]:
            raise ValueError("Invalid claim verdict")

        if not claim.get("claim"):
            raise ValueError("Claim text is missing")

        if claim["verdict"] == "contradicted":
            scores["faithfulness"] = min(
                scores["faithfulness"], 2
            )

        elif claim["verdict"] == "unsupported":
            scores["faithfulness"] = min(
                scores["faithfulness"], 4
            )

    # Apply deterministic regression checks
    scores = apply_regression_checks(
        item,
        scores,
        retrieved_chunks
    )


    if item["answerable"]:
        scores["abstention_score"] = None
    else:
        scores["abstention_score"] = evaluate_abstention(
            item["generated_answer"]
        )

    return scores



# --------------------------------------------------
# 7. Run evaluation with checkpoint and resume
# --------------------------------------------------

if __name__ == "__main__":

    # Start with previously saved evaluation results
    evaluation_results = list(existing_results)

    def save_checkpoint():
        """Save progress after every evaluation attempt."""
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {"results": evaluation_results},
                file,
                indent=4,
                ensure_ascii=False
            )

    def update_result(new_result):
        """Replace an existing question result instead of duplicating it."""
        for index, old_result in enumerate(evaluation_results):
            if old_result.get("id") == new_result["id"]:
                evaluation_results[index] = new_result
                return

        evaluation_results.append(new_result)

    for index, item in enumerate(results, start=1):

        question_id = item["id"]

        # Skip questions that already have valid scores
        previous_result = existing_by_id.get(question_id)

        if (
            previous_result
            and isinstance(previous_result.get("scores"), dict)
        ):
            print(
                f"Skipping {question_id} ({index}/{len(results)}) "
                "- already evaluated."
            )
            continue

        print(
            f"Evaluating {question_id} ({index}/{len(results)})..."
        )

        try:
            scores = evaluate_answer(item)

            new_result = {
                "id": question_id,
                "question": item["question"],
                "answerable": item["answerable"],
                "generated_answer": item["generated_answer"],
                "scores": scores
            }

            update_result(new_result)

            # Update lookup so this question is recognized as completed
            existing_by_id[question_id] = new_result

            print(f"Successfully evaluated {question_id}.")

        except Exception as error:

            print(f"Evaluation failed for {question_id}: {error}")

            new_result = {
                "id": question_id,
                "question": item["question"],
                "answerable": item["answerable"],
                "error": str(error)
            }

            update_result(new_result)

            # If Groq's rate limit is reached, stop further API calls
            error_message = str(error).lower()

            if (
                "rate_limit_exceeded" in error_message
                or "429" in error_message
            ):
                save_checkpoint()
                print(
                    "\nGroq rate limit reached. "
                    "Progress saved. Stopping evaluation."
                )
                break

        # Save progress after every attempted question
        save_checkpoint()


    # --------------------------------------------------
    # 8. Calculate evaluation metrics
    # --------------------------------------------------

    successful = [
        item for item in evaluation_results
        if isinstance(item.get("scores"), dict)
    ]

    answerable_results = [
        item for item in successful
        if item["answerable"] is True
    ]

    unanswerable_results = [
        item for item in successful
        if item["answerable"] is False
    ]

    def average_score(items, field):
        valid_scores = [
            item["scores"][field]
            for item in items
            if isinstance(item.get("scores"), dict)
            and isinstance(item["scores"].get(field), (int, float))
        ]

        if not valid_scores:
            return None

        return round(sum(valid_scores) / len(valid_scores), 2)

    summary = {
        "total_questions": len(results),

        "successfully_evaluated": len(successful),

        "failed_evaluations": len(results) - len(successful),

        "answerable_questions": len(answerable_results),

        "average_correctness": average_score(
            answerable_results, "correctness"
        ),

        "average_faithfulness": average_score(
            answerable_results, "faithfulness"
        ),

        "average_relevance": average_score(
            answerable_results, "relevance"
        ),

        "unanswerable_questions": len(unanswerable_results),

        "average_abstention_score": average_score(
            unanswerable_results, "abstention_score"
        )
    }


    # --------------------------------------------------
    # 9. Save final evaluation report
    # --------------------------------------------------

    report = {
        "summary": summary,
        "results": evaluation_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------

    print("\n========== EVALUATION SUMMARY ==========")

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nReport saved to:", OUTPUT_FILE)
