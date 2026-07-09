import json
from typing import Dict, Type
from stonks.models.base_model import BaseModel
from stonks.models.lstm import LSTMModel
from stonks.models.transformer import TransformerModel
from stonks.models.finbert import FinBERTModel
from stonks.models.extra_trees import ExtraTreesModel
from stonks.models.xgboost import XGBoostModel
from stonks.models.lightgbm import LightGBMModel
from stonks.models.catboost import CatBoostModel
from stonks.models.logistic_regression import LogisticRegressionModel
from stonks.config.settings import settings
from stonks.logging.logger import logger

# Base registry dictionary mapping lowercase codes to model classes
_REGISTRY: Dict[str, Type[BaseModel]] = {
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "finbert": FinBERTModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "extra_trees": ExtraTreesModel,
    "logistic_regression": LogisticRegressionModel
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
        name: Key name of the model
        
    Returns:
        Type[BaseModel]: The class constructor matching the key
    """
    clean_name = name.strip().lower()
    
    # Handle lazy resolution of RandomForestModel to prevent circularity
    if clean_name in ("rf", "random_forest"):
        try:
            from stonks.models.random_forest import RandomForestModel
            return RandomForestModel
        except ImportError as e:
            raise ImportError(
                "RandomForestModel class is not implemented or registered in the environment yet."
            ) from e
            
    if clean_name not in _REGISTRY:
        available = list(_REGISTRY.keys()) + ["rf", "random_forest"]
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
        from stonks.models.random_forest import RandomForestModel
        full_dict["rf"] = RandomForestModel
        full_dict["random_forest"] = RandomForestModel
    except ImportError:
        pass
    return full_dict

def get_best_model() -> Type[BaseModel]:
    """
    Reads the leaderboard.json file, parses the model name of rank 1,
    and returns its registered model class wrapper from the registry.
    Falls back to the configured settings.MODEL if no leaderboard is found.
    
    Returns:
        Type[BaseModel]: The class constructor of the best-performing model.
    """
    leaderboard_path = settings.MODEL_DIR / "leaderboard.json"
    if leaderboard_path.exists():
        try:
            with open(leaderboard_path, "r") as f:
                data = json.load(f)
            
            best_name = None
            if isinstance(data, list) and len(data) > 0:
                best_name = data[0].get("model")
            elif isinstance(data, dict) and "leaderboard" in data:
                leaderboard = data["leaderboard"]
                if len(leaderboard) > 0:
                    best_name = leaderboard[0].get("model")
                    
            if best_name:
                logger.info(f"Model Registry: Resolved best model '{best_name}' from leaderboard.")
                return get_model_class(best_name)
        except Exception as e:
            logger.warning(f"Failed to read leaderboard.json, falling back to default: {e}")
            
    # Fallback to config MODEL parameter
    logger.info(f"Model Registry: Falling back to configured MODEL '{settings.MODEL}'.")
    return get_model_class(settings.MODEL)
