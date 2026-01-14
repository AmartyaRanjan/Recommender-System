from pydantic import BaseModel, Field
from typing import List, Optional

class TelemetryEvent(BaseModel):
    event_type: str # e.g., "scroll", "video_pause", "tab_switch"
    context_id: int # The ID mapped to the concept (for SAINT Encoder)
    behavior_id: int # The ID mapped to the interaction (for SAINT Decoder)
    timestamp: float
    metadata: Optional[dict] = None

class TelemetryBatch(BaseModel):
    """The packet sent from Flutter every 30-60 seconds."""
    student_id: str
    events: List[TelemetryEvent]
    
    def to_saint_input(self):
        """Converts the batch into list format for the SAINT model."""
        context_seq = [e.context_id for e in self.events]
        behavior_seq = [e.behavior_id for e in self.events]
        return context_seq, behavior_seq