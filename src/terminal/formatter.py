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
                cell_str = str(cell)
                clean_str = cell_str
                # Strip simple color codes for size estimation
                for code in [cls.GREEN, cls.RED, cls.YELLOW, cls.BLUE, cls.CYAN, cls.BOLD, cls.RESET, cls.UNDERLINE]:
                    clean_str = clean_str.replace(code, "")
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
                clean_val = val_str
                for code in [cls.GREEN, cls.RED, cls.YELLOW, cls.BLUE, cls.CYAN, cls.BOLD, cls.RESET, cls.UNDERLINE]:
                    clean_val = clean_val.replace(code, "")
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
        max_len = max(len(title) + 4, 40)
        for line in lines:
            max_len = max(max_len, len(line))
            
        title_border = f"== {title} " + "=" * (max_len - len(title) - 4)
        out = [cls.colorize(f"╔{title_border}╗", border_color)]
        
        for line in lines:
            pad = " " * (max_len - len(line))
            out.append(cls.colorize("║", border_color) + f" {line}{pad} " + cls.colorize("║", border_color))
            
        out.append(cls.colorize("╚" + "=" * (max_len + 2) + "╝", border_color))
        return "\n".join(out)
