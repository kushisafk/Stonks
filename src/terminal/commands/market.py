from typing import List
from src.agent.pipeline import trading_agent
from src.terminal.errors import UsageError, CommandError
from src.terminal.formatter import TextFormatter

class MarketCommands:
    """Handles the 'market' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: market <subcommand> [args]\nSubcommands: analyze, compare, explain, news, inspect, research")
            
        subcmd = args[0].lower()
        
        if subcmd == "analyze":
            if len(args) < 2:
                raise UsageError("Usage: market analyze <ticker>")
            ticker = args[1].upper()
            print(f"Running predictive intelligence pipeline for {ticker}...")
            res = trading_agent.run_pipeline(ticker, force_train=False)
            print(res["intelligence"]["markdown_report"])
            
        elif subcmd == "compare":
            if len(args) < 3:
                raise UsageError("Usage: market compare <ticker1> <ticker2>")
            ticker1, ticker2 = args[1].upper(), args[2].upper()
            print(f"Analyzing and comparing {ticker1} vs {ticker2}...")
            
            res1 = trading_agent.run_pipeline(ticker1, force_train=False)
            res2 = trading_agent.run_pipeline(ticker2, force_train=False)
            
            intel1 = res1["intelligence"]["json_report"]
            intel2 = res2["intelligence"]["json_report"]
            
            headers = ["Metric", ticker1, ticker2]
            rows = [
                ["Prediction", intel1["prediction"], intel2["prediction"]],
                ["Probability", intel1["probability"], intel2["probability"]],
                ["Sentiment", intel1["news_sentiment"], intel2["news_sentiment"]],
                ["Risk Tier", intel1["risk_tier"], intel2["risk_tier"]],
                ["Risk Score", str(intel1["risk_score"]), str(intel2["risk_score"])],
                ["Recommendation", intel1["recommendation"], intel2["recommendation"]]
            ]
            print(f"\n{TextFormatter.bold('Comparison Matrix')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "explain":
            if len(args) < 2:
                raise UsageError("Usage: market explain <ticker>")
            ticker = args[1].upper()
            res = trading_agent.run_pipeline(ticker, force_train=False)
            print(f"\n{TextFormatter.bold('Explanation for ' + ticker)}")
            print(res["explanation"])
            
        elif subcmd == "news":
            if len(args) < 2:
                raise UsageError("Usage: market news <ticker>")
            ticker = args[1].upper()
            # Fetch raw data using market data service to retrieve news dict from yfinance
            from src.data.market_data import market_data_service
            print(f"Retrieving news articles for {ticker}...")
            ticker_obj = market_data_service._get_ticker_object(ticker)
            news = ticker_obj.news
            if not news:
                print(f"No news articles found for {ticker}.")
                return
                
            headers = ["Title", "Publisher", "Link"]
            rows = []
            for n in news[:5]:
                rows.append([n.get("title", "N/A"), n.get("publisher", "N/A"), n.get("link", "N/A")])
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "inspect":
            if len(args) < 2:
                raise UsageError("Usage: market inspect <ticker>")
            ticker = args[1].upper()
            res = trading_agent.run_pipeline(ticker, force_train=False)
            import json
            print(json.dumps(res["intelligence"]["json_report"], indent=4))
            
        elif subcmd == "research":
            if len(args) < 2:
                raise UsageError("Usage: market research <ticker>")
            ticker = args[1].upper()
            # Fetch recent price action
            from src.data.market_data import market_data_service
            df = market_data_service.fetch_data(ticker)
            recent = df.tail(5)
            headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
            rows = []
            for idx, row in recent.iterrows():
                rows.append([
                    idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    f"${row['Open']:.2f}",
                    f"${row['High']:.2f}",
                    f"${row['Low']:.2f}",
                    f"${row['Close']:.2f}",
                    f"{int(row['Volume']):,}"
                ])
            print(f"\n{TextFormatter.bold('Recent Price Action: ' + ticker)}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "chart":
            if len(args) < 2:
                raise UsageError("Usage: market chart <ticker>")
            ticker = args[1].upper()
            print(f"Chart feature is under active design (Placeholder for {ticker}).")
            
        else:
            raise CommandError(f"Unknown market subcommand '{subcmd}'. Type 'help market' for options.")
