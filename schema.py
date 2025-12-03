from typing import Optional
from pydantic import BaseModel, Field


class ClaimAttributes(BaseModel):
    """
    Structured representation of normalized claim information.
    """
    loss_type: str = Field(default="Unknown", description="Cause or category of loss")
    severity: str = Field(default="Unknown", description="Low / Medium / High / Critical")
    asset: str = Field(default="Unknown", description="Affected asset or property")

    estimated_loss: Optional[str] = Field(
        default="Unknown", description="Approximate loss amount (if mentioned)"
    )
    incident_date: Optional[str] = Field(
        default="Unknown", description="Date or time information of the incident"
    )
    location: Optional[str] = Field(
        default="Unknown", description="Location of the incident, if provided"
    )
    confidence: Optional[str] = Field(
        default="Unknown", description="Model confidence: Low / Medium / High"
    )
    explanation: Optional[str] = Field(
        default="Not provided", description="Short natural language explanation"
    )

    def to_display_dict(self) -> dict:
        """
        Handy method to convert to a simple dict for UI rendering.
        """
        return self.model_dump()
