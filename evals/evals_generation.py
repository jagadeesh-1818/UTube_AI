
import json
import sys
from pathlib import Path

# Add project root directory to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables before importing the RAG chain
load_dotenv(override=True)

from src.transcript import extract_video_id, extract_transcript
from src.vector_store import create_vector_store, create_retriever
from src.rag_chain import get_rag_response
from evals.dataset import evaluation_dataset


VIDEO_URL = "https://youtu.be/6E_5dW13R24?si=e4OmLvC5YFPYU_ni"

OUTPUT_FILE = Path(__file__).parent / "generation_results.json"


def run_generation_evaluation():

    print("\nLoading YouTube transcript...")

    video_id = extract_video_id(VIDEO_URL)

    documents, language = extract_transcript(video_id)

    print(f"Transcript language: {language}")

    print("\nCreating vector store...")

    vector_store = create_vector_store(documents)

    retriever = create_retriever(vector_store)

    # Load previously generated results so we can resume
    results_by_id = {}

    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
                previous_results = json.load(file)

            if isinstance(previous_results, list):
                results_by_id = {
                    item["id"]: item for item in previous_results
                    if isinstance(item, dict) and "id" in item
                }

            print(f"Loaded {len(results_by_id)} previous results.")

        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not load previous results: {e}")
            print("Starting with an empty results list.")

    total = len(evaluation_dataset)

    print(f"\nStarting generation evaluation for {total} questions...")

    
    for index, item in enumerate(evaluation_dataset, start=1):

        question_id = item["id"]
        question = item["question"]

        # Skip questions that already have successful answers
        if (
            question_id in results_by_id
            and results_by_id[question_id].get("generated_answer") is not None
        ):
            print(f"Skipping {question_id} - already generated.")
            continue

        print("\n" + "=" * 60)
        print(f"Question {index}/{total}: {question_id}")
        print(f"Question: {question}")

        try:
            # Empty history ensures independent evaluation
            result = get_rag_response(
                question=question,
                chat_history=[],
                retriever=retriever,
                return_details=True
            )

            generated_answer = result["answer"]

            evaluation_result = {
                "id": question_id,
                "question": question,
                "expected_answer": item["expected_answer"],
                "answerable": item["answerable"],
                "generated_answer": generated_answer,
                "retrieved_chunks": result["retrieved_chunks"],
                "manual_scores": {
                    "correctness": None,
                    "faithfulness": None,
                    "relevance": None,
                    "abstention": None
                }
            }

            print("\nExpected answer:")
            print(item["expected_answer"] or "Not provided in transcript")

            print("\nGenerated answer:")
            print(generated_answer)

            print("\nRetrieved chunk IDs:")
            print([
                chunk["chunk_id"]
                for chunk in result["retrieved_chunks"]
            ])

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")

            evaluation_result = {
                "id": question_id,
                "question": question,
                "expected_answer": item["expected_answer"],
                "answerable": item["answerable"],
                "generated_answer": None,
                "retrieved_chunks": [],
                "error": f"{type(e).__name__}: {e}",
                "manual_scores": {
                    "correctness": None,
                    "faithfulness": None,
                    "relevance": None,
                    "abstention": None
                }
            }

        # IMPORTANT: Save successful AND failed results
        results_by_id[question_id] = evaluation_result

        # Checkpoint after every question
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                [
                    results_by_id[item["id"]]
                    for item in evaluation_dataset
                    if item["id"] in results_by_id
                ],
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"\nCheckpoint saved: {len(results_by_id)}/{total}")


    # Final summary
    results = [
        results_by_id[item["id"]]
        for item in evaluation_dataset
        if item["id"] in results_by_id
    ]

    successful = sum(
        1 for result in results
        if result.get("generated_answer") is not None
    )

    failed = total - successful

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION COMPLETED")
    print(f"Total questions: {total}")
    print(f"Successful answers: {successful}")
    print(f"Failed answers: {failed}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_generation_evaluation()
