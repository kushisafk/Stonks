from typing import List, Any, Dict

class TextFormatter:
    """Helper formatting functions for rendering clean tabular listings and warning boxes."""
    
    # ANSI Color Codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    
    # Custom status prefixes
    SUCCESS = "[✓]"
    WARNING = "[⚠]"
    ALERT = "[🗲]"
    INFO = "[+]"
    
    @classmethod
    def colorize(cls, text: str, color_code: str) -> str:
        """Wraps text with the given ANSI escape sequence."""
        return f"{color_code}{text}{cls.RESET}"
        
    @classmethod
    def green(cls, text: str) -> str: return cls.colorize(text, cls.GREEN)
    @classmethod
    def red(cls, text: str) -> str: return cls.colorize(text, cls.RED)
    @classmethod
    def yellow(cls, text: str) -> str: return cls.colorize(text, cls.YELLOW)
    @classmethod
    def bold(cls, text: str) -> str: return cls.colorize(text, cls.BOLD)
    @classmethod
    def blue(cls, text: str) -> str: return cls.colorize(text, cls.BLUE)
    @classmethod
    def cyan(cls, text: str) -> str: return cls.colorize(text, cls.CYAN)
    @classmethod
    def underline(cls, text: str) -> str: return cls.colorize(text, cls.UNDERLINE)
    @classmethod
    def clean_ansi(cls, text: str) -> str:
        """Strips all ANSI color codes and OSC 8 terminal hyperlink sequences to get visible text length."""
        import re
        # Strip OSC 8 opening and closing hyperlink sequences
        text = re.sub(r'\x1b\]8;;.*?\x1b\\', '', text)
        text = re.sub(r'\x1b\]8;;\x1b\\', '', text)
        # Strip standard ANSI escape sequences
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        return text
        
    @classmethod
    def hyperlink(cls, url: str, text: str) -> str:
        """Returns an OSC 8 terminal hyperlink escape sequence."""
        return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"
        
    @classmethod
    def to_table(cls, headers: List[str], rows: List[List[Any]]) -> str:
        """
        Builds a robust, dynamic ASCII table with padded columns.
        
        Args:
            headers: Column header strings
            rows: List of data row lists
            
        Returns:
            str: Formatted ASCII table string
        """
        if not headers:
            return ""
            
        # Determine maximum column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                # Clean ANSI escape sequences for correct length measurement
                clean_str = cls.clean_ansi(str(cell))
                if idx < len(widths):
                    widths[idx] = max(widths[idx], len(clean_str))
                    
        # Build dividers
        div = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
        
        # Build headers row
        header_cells = [f" {headers[idx]:<{widths[idx]}} " for idx in range(len(headers))]
        header_row = "|" + "|".join(header_cells) + "|"
        
        lines = [div, header_row, div.replace("-", "=")]
        
        # Build data rows
        for row in rows:
            # We align right if data is numeric, left if text
            cells = []
            for idx, val in enumerate(row):
                val_str = str(val)
                # Calculate alignment padding considering ANSI codes length offsets
                clean_val = cls.clean_ansi(val_str)
                pad_len = widths[idx] - len(clean_val)
                
                # Check if field contains numeric values
                is_num = False
                try:
                    clean_str_check = clean_val.replace("$", "").replace("%", "").replace(",", "").strip()
                    float(clean_str_check)
                    is_num = True
                except ValueError:
                    pass
                    
                if is_num:
                    cells.append(f" {' ' * pad_len}{val_str} ")
                else:
                    cells.append(f" {val_str}{' ' * pad_len} ")
                    
            lines.append("|" + "|".join(cells) + "|")
            
        lines.append(div)
        return "\n".join(lines)
        
    @classmethod
    def to_panel(cls, title: str, lines: List[str], border_color: str = BOLD) -> str:
        """Wraps text lines inside a boxed outline panel with a header title."""
        max_len = max(len(cls.clean_ansi(title)) + 4, 40)
        for line in lines:
            max_len = max(max_len, len(cls.clean_ansi(line)))
            
        title_border = f"══ {title} " + "═" * (max_len - len(cls.clean_ansi(title)) - 2)
        out = [cls.colorize(f"╔{title_border}╗", border_color)]
        
        for line in lines:
            pad = " " * (max_len - len(cls.clean_ansi(line)))
            out.append(cls.colorize("║", border_color) + f" {line}{pad} " + cls.colorize("║", border_color))
            
        out.append(cls.colorize("╚" + "═" * (max_len + 2) + "╝", border_color))
        return "\n".join(out)
        
    @classmethod
    def to_intelligence_report(cls, intel: Dict[str, Any]) -> str:
        """Renders a beautifully colorized console panel containing key predictive metrics."""
        import textwrap
        
        ticker = intel.get("ticker", "UNKNOWN")
        pred = intel.get("prediction", "HOLD").upper()
        prob = intel.get("probability", "0.0%")
        conf = intel.get("confidence_tier", "Medium")
        conf_rat = intel.get("confidence_rationale", "")
        regime = intel.get("market_regime", "Sideways")
        sentiment = intel.get("news_sentiment", "0.00")
        rel_str = intel.get("relative_strength_20d", "+0.0%")
        risk_score = intel.get("risk_score", 0)
        risk_tier = intel.get("risk_tier", "Medium Risk")
        rec = intel.get("recommendation", "HOLD").upper()
        reasoning = intel.get("reasoning", "")
        confirmations = intel.get("confirmations", [])
        warnings = intel.get("warnings", [])
        risk_factors = intel.get("risk_factors", [])
        
        # Color Coding
        # 1. Recommendation Signal
        if "BUY" in rec:
            rec_color = cls.green(cls.bold(rec))
        elif "SELL" in rec:
            rec_color = cls.red(cls.bold(rec))
        else:
            rec_color = cls.yellow(cls.bold(rec))
            
        # 2. Confidence Tier
        if conf == "High":
            conf_color = cls.green(conf)
        elif conf == "Medium":
            conf_color = cls.yellow(conf)
        else:
            conf_color = cls.red(conf)
            
        # 3. Risk Profile
        if risk_score >= 70:
            risk_color = cls.red(f"{risk_tier} (Score: {risk_score}/100)")
        elif risk_score >= 40:
            risk_color = cls.yellow(f"{risk_tier} (Score: {risk_score}/100)")
        else:
            risk_color = cls.green(f"{risk_tier} (Score: {risk_score}/100)")
            
        # Sentiment label
        try:
            sent_val = float(sentiment)
            if sent_val > 0.05:
                sent_label = f"{sentiment} (Positive)"
            elif sent_val < -0.05:
                sent_label = f"{sentiment} (Negative)"
            else:
                sent_label = f"{sentiment} (Neutral)"
        except ValueError:
            sent_label = sentiment

        # Prepare details lines
        lines = []
        lines.append(f"  Recommended Action : {rec_color}")
        lines.append(f"  Confidence Level   : {conf_color}")
        lines.append(f"  Risk Profile       : {risk_color}")
        lines.append("")
        lines.append("  " + "─" * 74)
        lines.append("")
        lines.append("  " + cls.bold("Executive Setup Summary:"))
        lines.append(f"    • Base Prediction                 : {pred} ({prob})")
        lines.append(f"    • News Sentiment                  : {sent_label}")
        lines.append(f"    • Relative Strength (20d vs SPY)  : {rel_str}")
        lines.append(f"    • Market Regime                   : {regime}")
        lines.append("")
        lines.append("  " + "─" * 74)
        lines.append("")
        
        # Reasoning text wrapping (target width ~70 chars inside the panel)
        lines.append("  " + cls.bold("Reasoning & Rationale:"))
        wrap_width = 70
        for para in reasoning.split("\n"):
            if not para.strip():
                continue
            wrapped_para = textwrap.wrap(para, width=wrap_width)
            for wl in wrapped_para:
                lines.append(f"    {wl}")
        
        # Confidence rationale wrapping
        if conf_rat:
            wrapped_conf = textwrap.wrap(conf_rat, width=wrap_width)
            if wrapped_conf:
                lines.append(f"    • {wrapped_conf[0]}")
                for wl in wrapped_conf[1:]:
                    lines.append(f"      {wl}")
                    
        # Confirmations, Warnings, Risk Factors
        has_checks = bool(confirmations or warnings or risk_factors)
        if has_checks:
            lines.append("")
            lines.append("  " + "─" * 74)
            lines.append("")
            
            if confirmations:
                lines.append("  " + cls.green(cls.bold("Confirmations:")))
                for c in confirmations:
                    wrapped_c = textwrap.wrap(c, width=wrap_width - 4)
                    if wrapped_c:
                        lines.append(f"    {cls.green('✓')} {wrapped_c[0]}")
                        for wl in wrapped_c[1:]:
                            lines.append(f"      {wl}")
                lines.append("")
                            
            if warnings:
                lines.append("  " + cls.yellow(cls.bold("Warnings:")))
                for w in warnings:
                    wrapped_w = textwrap.wrap(w, width=wrap_width - 4)
                    if wrapped_w:
                        lines.append(f"    {cls.yellow('⚠')} {wrapped_w[0]}")
                        for wl in wrapped_w[1:]:
                            lines.append(f"      {wl}")
                lines.append("")
                            
            if risk_factors:
                lines.append("  " + cls.red(cls.bold("Active Risk Factors:")))
                for r in risk_factors:
                    wrapped_r = textwrap.wrap(r, width=wrap_width - 4)
                    if wrapped_r:
                        lines.append(f"    {cls.red('🗲')} {wrapped_r[0]}")
                        for wl in wrapped_r[1:]:
                            lines.append(f"      {wl}")
                lines.append("")
                
            # Pop last newline if empty
            if lines[-1] == "":
                lines.pop()
                
        # Build panel output
        box_width = 80
        out = []
        title_border = f"══ STONKS Intelligence Report: {ticker} "
        title_len = len(title_border)
        title_border += "═" * (box_width - title_len)
        out.append(cls.colorize(f"╔{title_border}╗", cls.BOLD))
        
        for line in lines:
            clean_len = len(cls.clean_ansi(line))
            pad_len = box_width - clean_len - 2
            if pad_len < 0:
                pad_len = 0
            
            padded_line = f" {line}{' ' * pad_len} "
            out.append(cls.colorize("║", cls.BOLD) + padded_line + cls.colorize("║", cls.BOLD))
            
        out.append(cls.colorize("╚" + "═" * box_width + "╝", cls.BOLD))
        return "\n".join(out)
        
    @classmethod
    def to_news_panel(cls, ticker: str, news_items: List[Dict[str, str]]) -> str:
        """Renders recent news articles in a beautiful, wrapped console list with working full hyperlinks."""
        import textwrap
        
        out = []
        out.append(cls.bold(f"\nRecent News: {ticker}"))
        out.append("")
        
        for idx, item in enumerate(news_items, 1):
            title = item.get("title", "N/A")
            publisher = item.get("publisher", "N/A")
            link = item.get("link", "N/A")
            
            # Wrap title to ~76 characters
            wrapped_title = textwrap.wrap(title, width=76)
            out.append(f"  {idx}. {cls.bold(wrapped_title[0])}")
            for wl in wrapped_title[1:]:
                out.append(f"     {cls.bold(wl)}")
                
            out.append(f"     Source: {cls.cyan(publisher)}")
            
            # Print full link, which wraps naturally and stays clickable in all terminals
            if link and link != "N/A":
                out.append(f"     Link  : {cls.underline(link)}")
            else:
                out.append("     Link  : N/A")
            out.append("")
            
        return "\n".join(out)
