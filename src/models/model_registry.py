from typing import Dict, Type
from src.models.base_model import BaseModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.models.finbert import FinBERTModel

# Base registry dictionary containing Phase 1 stubs
_REGISTRY: Dict[str, Type[BaseModel]] = {
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "finbert": FinBERTModel
}

def register_model_class(name: str, cls: Type[BaseModel]) -> None:
    """
    Registers a new concrete model class with the global model registry.
    
    Args:
        name: Short key code name of the model (e.g. 'rf')
        cls: The class type inheriting from BaseModel
    """
    clean_name = name.strip().lower()
    _REGISTRY[clean_name] = cls

def get_model_class(name: str) -> Type[BaseModel]:
    """
    Retrieves a registered model class by its key name.
    Dynamically loads RandomForestModel to prevent import circularity during early setup.
    
    Args:
        name: Key name of the model ('rf', 'lstm', 'transformer', 'finbert')
        
    Returns:
        Type[BaseModel]: The class constructor matching the key
    """
    clean_name = name.strip().lower()
    
    # Handle lazy resolution of RandomForestModel
    if clean_name == "rf":
        try:
            from src.models.random_forest import RandomForestModel
            return RandomForestModel
        except ImportError as e:
            raise ImportError(
                "RandomForestModel class is not implemented or registered in the environment yet."
            ) from e
            
    if clean_name not in _REGISTRY:
        available = list(_REGISTRY.keys()) + ["rf"]
        raise KeyError(
            f"Model name '{name}' is not registered. Registered options: {available}"
        )
        
    return _REGISTRY[clean_name]

def list_registered_models() -> Dict[str, Type[BaseModel]]:
    """
    Returns a dictionary of all registered model classes.
    
    Returns:
        Dict[str, Type[BaseModel]]: Mappings of all active model classes
    """
    full_dict = _REGISTRY.copy()
    try:
        from src.models.random_forest import RandomForestModel
        full_dict["rf"] = RandomForestModel
    except ImportError:
        pass
    return full_dict
