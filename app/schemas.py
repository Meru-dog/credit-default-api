"""Request and response schemas for the prediction API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Amount = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeAmount = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveAmount = Annotated[float, Field(gt=0, allow_inf_nan=False)]
RepaymentStatus = Annotated[int, Field(ge=-2, le=9)]


class PredictionRequest(BaseModel):
    """One record with the 19 features used by the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    LIMIT_BAL: PositiveAmount
    PAY_0: RepaymentStatus
    PAY_2: RepaymentStatus
    PAY_3: RepaymentStatus
    PAY_4: RepaymentStatus
    PAY_5: RepaymentStatus
    PAY_6: RepaymentStatus
    BILL_AMT1: Amount
    BILL_AMT2: Amount
    BILL_AMT3: Amount
    BILL_AMT4: Amount
    BILL_AMT5: Amount
    BILL_AMT6: Amount
    PAY_AMT1: NonNegativeAmount
    PAY_AMT2: NonNegativeAmount
    PAY_AMT3: NonNegativeAmount
    PAY_AMT4: NonNegativeAmount
    PAY_AMT5: NonNegativeAmount
    PAY_AMT6: NonNegativeAmount


class PredictionResponse(BaseModel):
    default_probability: float
    review: bool
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    features: list[str]
    limitations: list[str]
