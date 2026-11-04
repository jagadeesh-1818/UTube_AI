from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from langchain_core.documents import Document


PREFERRED_LANGUAGES = [
    "en",
    "hi",
    "kn",
    "ta",
    "te",
    "ml",
    "bn",
    "mr",
    "es",
    "fr",
    "de",
    "ja",
    "zh"
]


def extract_video_id(url):

    parsed_url = urlparse(url)

    hostname = (parsed_url.hostname or "").lower()

    if hostname in ["youtu.be", "www.youtu.be"]:

        return parsed_url.path.strip("/").split("/")[0]

    if hostname in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    ]:

        if parsed_url.path == "/watch":

            return parse_qs(
                parsed_url.query
            ).get("v", [None])[0]

        if parsed_url.path.startswith(
            ("/shorts/", "/embed/", "/live/")
        ):

            parts = parsed_url.path.strip("/").split("/")

            if len(parts) >= 2:
                return parts[1]

    return None


def extract_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()

    try:

        transcript_list = ytt_api.list(video_id)

        available_transcripts = list(transcript_list)

        if not available_transcripts:

            raise ValueError("No transcripts available.")

        print("\nAvailable transcripts:")

        for item in available_transcripts:

            print(
                f"{item.language} "
                f"({item.language_code}) "
                f"- Generated: {item.is_generated}"
            )

        selected_transcript = None

        # Select preferred language
        try:

            selected_transcript = transcript_list.find_transcript(
                PREFERRED_LANGUAGES
            )

        except Exception:

            # Fallback to first available transcript
            selected_transcript = available_transcripts[0]

        print(
            "\nSelected transcript:",
            selected_transcript.language
        )

        fetched_transcript = selected_transcript.fetch()

        # Group snippets into approximately 60-second documents.
        # This keeps timestamp metadata useful during retrieval.

        documents = []

        current_snippets = []
        block_start = None

        for snippet in fetched_transcript:

            if block_start is None:

                block_start = snippet.start

            if (
                snippet.start - block_start >= 60
                and current_snippets
            ):

                text = " ".join(current_snippets)

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "start": block_start,
                            "language": selected_transcript.language_code
                        }
                    )
                )

                current_snippets = []
                block_start = snippet.start

            current_snippets.append(snippet.text)

        # Save remaining snippets

        if current_snippets:

            documents.append(
                Document(
                    page_content=" ".join(current_snippets),
                    metadata={
                        "start": block_start,
                        "language": selected_transcript.language_code
                    }
                )
            )

        if not documents:

            raise ValueError("Transcript is empty.")

        return documents, selected_transcript.language_code

    except Exception as e:

        raise RuntimeError(
            f"Transcript extraction failed: {type(e).__name__}: {e}"
        ) from e