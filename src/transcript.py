from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_core.documents import Document
import re


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
            return parse_qs(parsed_url.query).get("v", [None])[0]

        if parsed_url.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed_url.path.strip("/").split("/")
            if len(parts) >= 2:
                return parts[1]

    return None


def _timestamp_to_seconds(value):
    """Convert MM:SS or HH:MM:SS timestamps to seconds."""
    value = value.replace(",", ".")
    parts = value.split(":")

    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    except ValueError:
        return None

    return None


def transcript_text_to_documents(transcript_text, language="unknown"):
    """
    Convert pasted plain text, YouTube transcript text, SRT, or VTT
    into LangChain Documents.

    Timestamped content is grouped into approximately 60-second
    documents. Plain text is returned as one Document and will be
    chunked by the existing vector-store splitter.
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Please paste a transcript before processing.")

    # Matches MM:SS, HH:MM:SS, optional milliseconds, and optional SRT/VTT end time.
    timestamp_pattern = re.compile(
        r"^\s*(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)"
        r"(?:\s*-->\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
        r"(?:\s+.*)?)?\s*(?P<text>.*)$"
    )

    entries = []
    current_start = None
    current_lines = []

    def save_entry():
        nonlocal current_start, current_lines

        content = " ".join(
            line.strip() for line in current_lines if line.strip()
        ).strip()

        if current_start is not None and content:
            entries.append((current_start, content))

        current_start = None
        current_lines = []

    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Ignore common subtitle-file headers and cue numbers.
        if line.upper().startswith("WEBVTT") or line.isdigit():
            continue

        match = timestamp_pattern.match(line)

        if match:
            timestamp = _timestamp_to_seconds(match.group("start"))

            if timestamp is not None:
                save_entry()
                current_start = timestamp

                inline_text = match.group("text").strip()
                if inline_text:
                    current_lines.append(inline_text)
                continue

        # Ignore SRT/VTT positioning/style metadata.
        if line.startswith(("NOTE", "Kind:", "Language:")):
            continue

        if current_start is not None:
            current_lines.append(line)

    save_entry()

    # If no timestamps were found, preserve the entire pasted transcript.
    if not entries:
        return [
            Document(
                page_content=transcript_text.strip(),
                metadata={"start": 0, "language": language}
            )
        ]

    # Group timestamped entries into approximately 60-second documents.
    documents = []
    block_start = None
    block_lines = []

    for timestamp, content in entries:

        if block_start is None:
            block_start = timestamp

        if timestamp - block_start >= 60 and block_lines:
            documents.append(
                Document(
                    page_content=" ".join(block_lines),
                    metadata={
                        "start": block_start,
                        "language": language
                    }
                )
            )
            block_start = timestamp
            block_lines = []

        block_lines.append(content)

    if block_lines:
        documents.append(
            Document(
                page_content=" ".join(block_lines),
                metadata={
                    "start": block_start,
                    "language": language
                }
            )
        )

    if not documents:
        raise ValueError("No readable transcript text was found.")

    return documents


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

        try:
            selected_transcript = transcript_list.find_transcript(
                PREFERRED_LANGUAGES
            )

        except Exception:
            selected_transcript = available_transcripts[0]

        print(
            "\nSelected transcript:",
            selected_transcript.language
        )

        fetched_transcript = selected_transcript.fetch()

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
                documents.append(
                    Document(
                        page_content=" ".join(current_snippets),
                        metadata={
                            "start": block_start,
                            "language": selected_transcript.language_code
                        }
                    )
                )

                current_snippets = []
                block_start = snippet.start

            current_snippets.append(snippet.text)

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
