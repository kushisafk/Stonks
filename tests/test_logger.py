import os
import csv
from pathlib import Path
from src.logging.logger import DecisionLogger, logger

def test_decision_logger_initialization(tmp_path):
    """Verify that the CSV log file is created with correct headers on init."""
    csv_file = tmp_path / "decisions.csv"
    assert not csv_file.exists()
    
    # Init logger
    dec_logger = DecisionLogger(csv_path=csv_file)
    assert csv_file.exists()
    
    # Read headers
    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        
    assert headers == [
        "Timestamp", 
        "Ticker", 
        "Signal", 
        "Confidence", 
        "ClosePrice", 
        "ModelProbabilities"
    ]

def test_decision_logger_log_decision(tmp_path):
    """Verify that decisions are correctly formatted and appended to the CSV log."""
    csv_file = tmp_path / "decisions.csv"
    dec_logger = DecisionLogger(csv_path=csv_file)
    
    # Log a buy decision
    dec_logger.log_decision(
        ticker="AAPL",
        signal="BUY",
        confidence=0.7245,
        close_price=180.25,
        probabilities={"rf": 0.7245}
    )
    
    # Log a hold decision without optional parameters
    dec_logger.log_decision(
        ticker="TSLA",
        signal="HOLD",
        confidence=0.5100
    )
    
    # Read the data back
    rows = []
    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)
            
    assert len(rows) == 2
    
    # Verify AAPL entry
    aapl_row = rows[0]
    assert aapl_row[1] == "AAPL"
    assert aapl_row[2] == "BUY"
    assert aapl_row[3] == "0.7245"
    assert aapl_row[4] == "180.2500"
    assert aapl_row[5] == "{'rf': 0.7245}"
    
    # Verify TSLA entry (defaults check)
    tsla_row = rows[1]
    assert tsla_row[1] == "TSLA"
    assert tsla_row[2] == "HOLD"
    assert tsla_row[3] == "0.5100"
    assert tsla_row[4] == "N/A"
    assert tsla_row[5] == "{}"
