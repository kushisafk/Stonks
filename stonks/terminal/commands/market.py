from typing import List
from stonks.agent.pipeline import trading_agent
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter

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
            intel = res["intelligence"]["json_report"]
            print(TextFormatter.to_intelligence_report(intel))
            
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
            import textwrap
            wrapped_explanation = textwrap.wrap(res["explanation"], width=74)
            print(TextFormatter.to_panel(f"Explanation: {ticker}", wrapped_explanation))
            
        elif subcmd == "news":
            if len(args) < 2:
                raise UsageError("Usage: market news <ticker>")
            ticker = args[1].upper()
            import yfinance as yf
            print(f"Retrieving news articles for {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            if not news:
                print(f"No news articles found for {ticker}.")
                return
                
            news_items = []
            for n in news[:5]:
                content = n.get("content", {})
                if isinstance(content, dict) and content:
                    title = content.get("title", "N/A")
                    
                    provider = content.get("provider")
                    publisher = provider.get("displayName", "N/A") if isinstance(provider, dict) else "N/A"
                    
                    click_url = content.get("clickThroughUrl")
                    link = click_url.get("url", "N/A") if isinstance(click_url, dict) else "N/A"
                else:
                    title = n.get("title", "N/A")
                    publisher = n.get("publisher", "N/A")
                    link = n.get("link", "N/A")
                news_items.append({"title": title, "publisher": publisher, "link": link})
            print(TextFormatter.to_news_panel(ticker, news_items))
            
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
            from stonks.data.market_data import market_data_service
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
