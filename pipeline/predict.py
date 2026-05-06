# pipeline/predict.py
from pydantic import BaseModel, Field

class CrimePredictionInput(BaseModel):
    """
    Input schema for crime prediction.
    All fields are validated with min/max values and JSON schema examples.
    """
    annee: int = Field(
        ...,
        ge=2016,
        le=2030,
        json_schema_extra={"example": 2025}
    )
    dep_encoded: int = Field(
        ...,
        ge=0,
        le=100,
        json_schema_extra={"example": 5}
    )
    cat_encoded: int = Field(
        ...,
        ge=0,
        le=10,
        json_schema_extra={"example": 0}
    )
    annee_norm: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.8}
    )

# Example usage:
# input_data = CrimePredictionInput(
#     annee=2025,
#     dep_encoded=5,
#     cat_encoded=0,
#     annee_norm=0.8
# )