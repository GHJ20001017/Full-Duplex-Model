from pydantic import BaseModel


class InputItemState(BaseModel):
    """Active state for one client-visible input transcription item."""

    transcript_prefix: str = ""
    latest_transcript: str = ""
    audio_duration_s: float = 0.0
    # Set when this item id already published an authoritative final for an
    # earlier revision of a reopened turn: the item id is reused so the client
    # keeps one entry, but re-streaming text would duplicate what that final
    # already replaced.
    finalized: bool = False
