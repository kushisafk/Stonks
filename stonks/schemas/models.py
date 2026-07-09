from pydantic import BaseModel, Field
from typing import Dict

class ModelInfo(BaseModel):
    weight: float = Field(..., description="The weight allocated to this model in the ensemble")
    is_active: bool = Field(..., description="Flag indicating if the model has active registration in the pipeline")
    status: str = Field(..., description="Model operational status (e.g. READY, STUB)")

class ModelsResponse(BaseModel):
    active_models: Dict[str, ModelInfo] = Field(..., description="Summary breakdown of active registered models")
    total_active_weight: float = Field(..., description="Cumulative sum of registered weights (normalizes to 1.0)")
