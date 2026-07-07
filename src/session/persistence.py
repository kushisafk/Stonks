import os
import shutil
import json
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.session.schemas import SessionState
from src.session.exceptions import SessionLoadError, SessionSaveError
from src.logging.logger import logger

class PersistenceManager:
    """Manages the serialization, atomic writing, schema validation, migration, and recovery of the SessionState."""
    
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.backup_path = self.filepath.with_suffix(".backup")
        self._lock = threading.RLock()
        
    def save(self, state: SessionState) -> None:
        """
        Saves the SessionState atomically to disk.
        Writes to a temporary file first, fsyncs it, then replaces the active file.
        Also creates a backup on successful write.
        """
        with self._lock:
            try:
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Serialize model to JSON string
                json_str = state.model_dump_json(indent=4)
                
                # Atomic Write using NamedTemporaryFile
                temp_dir = self.filepath.parent
                with NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as temp_file:
                    temp_file.write(json_str)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_filepath = Path(temp_file.name)
                    
                # Replace active file atomically
                os.replace(temp_filepath, self.filepath)
                
                # Copy active file to backup file
                shutil.copy2(self.filepath, self.backup_path)
                
            except Exception as e:
                logger.error(f"PersistenceManager: Failed to save session atomically: {e}")
                raise SessionSaveError(f"Failed to serialize and save session state: {e}") from e
            
    def load(self) -> SessionState:
        """
        Loads the SessionState from disk.
        If the active file is missing or corrupted, attempts to restore from the backup file.
        If both are missing or corrupted, returns a default clean SessionState.
        """
        with self._lock:
            # Try loading main file
            if self.filepath.exists():
                try:
                    state = self._read_file(self.filepath)
                    logger.info(f"PersistenceManager: Successfully loaded session from {self.filepath}")
                    return state
                except Exception as e:
                    logger.warning(f"PersistenceManager: Failed to load from main file '{self.filepath}': {e}. Attempting recovery from backup...")
                    
            # Try recovery from backup file
            if self.backup_path.exists():
                try:
                    state = self._read_file(self.backup_path)
                    logger.info(f"PersistenceManager: Recovered session state from backup file: {self.backup_path}")
                    # Restore backup to active file
                    shutil.copy2(self.backup_path, self.filepath)
                    return state
                except Exception as backup_err:
                    logger.error(f"PersistenceManager: Recovery from backup file failed: {backup_err}")
                    
            # Both failed/missing -> return clean default state
            logger.info("PersistenceManager: No valid session files found. Initializing a clean session state.")
            default_state = SessionState()
            # Save it immediately so file exists
            try:
                self.save(default_state)
            except Exception:
                pass
            return default_state

    def _read_file(self, path: Path) -> SessionState:
        """Helper to read, validate, and migrate JSON to SessionState."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Version Check & Migration hooks
        schema_version = data.get("schema_version", 1)
        if schema_version < 1:
            data = self._migrate(data, schema_version, 1)
            
        # Parse into Pydantic model
        return SessionState(**data)
        
    def _migrate(self, data: dict, from_version: int, to_version: int) -> dict:
        """Stubs migration hook pipeline for schema modifications."""
        logger.info(f"PersistenceManager: Running schema migrations from v{from_version} to v{to_version}")
        data["schema_version"] = to_version
        return data
