import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on their severity levels."""
    # Define color codes
    COLORS = {
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'DEBUG': '\033[94m',  # Blue
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'   # Reset color
    }

    def format(self, record):
        # Apply color based on the level of the log
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        log_msg = super().format(record)
        return f"{log_color}{log_msg}{self.COLORS['RESET']}"

# Set up the logger
logger = logging.getLogger('colored_logger')
logger.setLevel(logging.INFO)

# Create a stream handler
handler = logging.StreamHandler()

# Apply the colored formatter
colored_formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(colored_formatter)

# Add the handler to the logger
logger.addHandler(handler)