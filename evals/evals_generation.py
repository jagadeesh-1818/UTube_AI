
import json
from pathlib import Path

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

    results = []

    total = len(evaluation_dataset)

    print(f"\nStarting generation evaluation for {total} questions...")

    for index, item in enumerate(evaluation_dataset, start=1):

        question_id = item["id"]
        question = item["question"]

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

        results.append(evaluation_result)

    # Save all generated answers and retrieved evidence
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    successful = sum(
        1 for result in results
        if result.get("generated_answer") is not None
    )

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION COMPLETED")
    print(f"Total questions: {total}")
    print(f"Successful answers: {successful}")
    print(f"Failed answers: {total - successful}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":

    run_generation_evaluation()
