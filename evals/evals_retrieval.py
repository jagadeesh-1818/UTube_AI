
from src.transcript import extract_video_id, extract_transcript
from src.vector_store import create_vector_store
from evals.dataset import evaluation_dataset


VIDEO_URL = "https://youtu.be/6E_5dW13R24?si=e4OmLvC5YFPYU_ni"

K = 8
FETCH_K = 25
LAMBDA_MULT = 0.95

# Questions we particularly want to investigate
FOCUS_QUESTIONS = {"Q02", "Q04", "Q07"}


def calculate_metrics(results, dataset):

    answerable_questions = [
        item for item in dataset if item["answerable"]
    ]

    total_questions = len(answerable_questions)

    hits = 0
    reciprocal_ranks = []
    recall_scores = []

    for item in answerable_questions:

        question_id = item["id"]
        relevant_ids = set(item["relevant_chunk_ids"])

        retrieved_ids = [
            doc.metadata.get("chunk_id")
            for doc in results[question_id]
        ]

        relevant_found = [
            chunk_id for chunk_id in retrieved_ids
            if chunk_id in relevant_ids
        ]

        if relevant_found:
            hits += 1

        recall = (
            len(set(retrieved_ids) & relevant_ids)
            / len(relevant_ids)
        )
        recall_scores.append(recall)

        reciprocal_rank = 0

        for rank, chunk_id in enumerate(retrieved_ids, start=1):

            if chunk_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return {
        "Hit Rate": hits / total_questions,
        "Recall@K": sum(recall_scores) / total_questions,
        "MRR": sum(reciprocal_ranks) / total_questions
    }


def run_retrieval(vector_store, dataset, method):

    results = {}

    for item in dataset:

        question = item["question"]

        if method == "similarity":

            documents = vector_store.similarity_search(
                question,
                k=K
            )

        elif method == "mmr":

            documents = vector_store.max_marginal_relevance_search(
                question,
                k=K,
                fetch_k=FETCH_K,
                lambda_mult=LAMBDA_MULT
            )

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        results[item["id"]] = documents

    return results


def print_results(method, metrics):

    print(f"\n========== {method.upper()} RESULTS ==========")

    for metric, value in metrics.items():

        print(f"{metric}: {value:.4f} ({value * 100:.2f}%)")

    print("============================================")


def print_question_diagnostics(results, dataset, method):

    print(f"\n========== {method.upper()} QUESTION DIAGNOSTICS ==========")

    for item in dataset:

        if not item["answerable"]:
            continue

        question_id = item["id"]
        relevant_ids = set(item["relevant_chunk_ids"])

        documents = results[question_id]

        retrieved_ids = [
            doc.metadata.get("chunk_id")
            for doc in documents
        ]

        found = set(retrieved_ids) & relevant_ids
        missing = relevant_ids - set(retrieved_ids)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None
        )

        print(f"\n{question_id}: {item['question']}")
        print(f"Relevant chunks: {sorted(relevant_ids)}")
        print(f"Retrieved chunks: {retrieved_ids}")
        print(f"Found relevant chunks: {sorted(found)}")
        print(f"Missing relevant chunks: {sorted(missing)}")
        print(f"First relevant rank: {first_relevant_rank}")

        if question_id in FOCUS_QUESTIONS:

            print("\nRetrieved chunk content:")

            for rank, doc in enumerate(documents, start=1):

                chunk_id = doc.metadata.get("chunk_id")
                content = doc.page_content.replace("\n", " ")

                status = (
                    "RELEVANT"
                    if chunk_id in relevant_ids
                    else "NOT RELEVANT"
                )

                print(f"\n  Rank {rank}: {chunk_id} [{status}]")
                print(f"  {content[:450]}...")

        print("-" * 65)


def main():

    print("\nLoading YouTube transcript...")

    video_id = extract_video_id(VIDEO_URL)

    documents, language = extract_transcript(video_id)

    print(f"Transcript language: {language}")

    print("\nCreating vector store...")

    vector_store = create_vector_store(documents)

    print("\nRunning Similarity Search...")

    similarity_results = run_retrieval(
        vector_store,
        evaluation_dataset,
        "similarity"
    )

    similarity_metrics = calculate_metrics(
        similarity_results,
        evaluation_dataset
    )

    print("\nRunning MMR Search...")

    mmr_results = run_retrieval(
        vector_store,
        evaluation_dataset,
        "mmr"
    )

    mmr_metrics = calculate_metrics(
        mmr_results,
        evaluation_dataset
    )

    print_results("Similarity Search", similarity_metrics)
    print_results("MMR", mmr_metrics)

    print_question_diagnostics(
        similarity_results,
        evaluation_dataset,
        "Similarity Search"
    )

    print_question_diagnostics(
        mmr_results,
        evaluation_dataset,
        "MMR"
    )

    print("\n========== COMPARISON ==========")

    for metric in similarity_metrics:

        similarity_score = similarity_metrics[metric]
        mmr_score = mmr_metrics[metric]

        difference = mmr_score - similarity_score

        print(
            f"{metric}: "
            f"Similarity = {similarity_score:.4f} | "
            f"MMR = {mmr_score:.4f} | "
            f"Difference = {difference:+.4f}"
        )

    print("================================")


if __name__ == "__main__":
    main()
