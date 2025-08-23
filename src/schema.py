# src/schemas.py
from pydantic import BaseModel, Field
import uuid # to give each prediction a unique ID

# Basemodel has data validation and serialization code
class ComplaintRequest(BaseModel):
    # defines the required structure for the body of an incoming POST request
    # JSON key must be named narrative, of type string & at least 10 characters long
    narrative: str = Field(..., min_length=10, description="The full text of the customer complaint.")

# structure for the JSON response
# Response has key of type str and a key of type uuid
class ClassificationResponse(BaseModel):
    predicted_product: str
    request_id: uuid.UUID