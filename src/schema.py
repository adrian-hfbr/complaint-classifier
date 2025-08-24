# src/schemas.py
from pydantic import BaseModel, Field
import uuid  # to give each prediction a unique ID


class ComplaintRequest(BaseModel):
    """
    This schema defines the structure for an incoming prediction request.
    JSON key must be named narrative, of type string & at least 10 characters long
    """

    narrative: str = Field(
        ..., min_length=10, description="The full text of the customer complaint."
    )


class ClassificationResponse(BaseModel):
    """
    This schema defines the structure for a prediction response.
    Response has key of type str and a key of type uuid
    """

    predicted_product: str
    request_id: uuid.UUID


# Schema for the correction request
class CorrectionRequest(BaseModel):
    """
    Represents a user-submitted correction for a prediction.
    """

    request_id: uuid.UUID
    narrative: str
    predicted_product: str
    correct_product: str
