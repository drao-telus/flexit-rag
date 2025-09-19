"""Enhanced logging configuration with traceback, colors, and detailed error context."""

import logging
import sys
import traceback
import inspect
from pathlib import Path
from typing import Optional, Dict, Any

# Color support
try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    class MockFore:
        RED = YELLOW = GREEN = CYAN = BLUE = MAGENTA = WHITE = RESET = ""

    class MockStyle:
        BRIGHT = DIM = RESET_ALL = ""

    Fore = MockFore()
    Style = MockStyle()
    COLORS_AVAILABLE = False


class EnhancedLocationFormatter(logging.Formatter):
    """Enhanced formatter with traceback support, colors, and detailed context."""

    # Color mapping for log levels
    LEVEL_COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def __init__(
        self,
        use_colors=True,
        include_traceback=True,
        traceback_limit=10,
        include_locals=False,
        source_context_lines=3,
    ):
        super().__init__()
        self.use_colors = use_colors and COLORS_AVAILABLE
        self.include_traceback = include_traceback
        self.traceback_limit = traceback_limit
        self.include_locals = include_locals
        self.source_context_lines = source_context_lines

    def format(self, record):
        # Get the relative path from the project root
        if hasattr(record, "pathname"):
            try:
                file_path = Path(record.pathname)
                try:
                    relative_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = file_path.name
                location = f"{relative_path}:{record.lineno}"
            except (AttributeError, ValueError):
                location = f"{record.filename}:{record.lineno}"
        else:
            location = f"{record.filename}:{record.lineno}"

        # Get function name from the call stack
        function_name = self._get_function_name(record)
        if function_name:
            location = f"{location} in {function_name}()"

        record.location = location

        # Apply colors if enabled
        level_color = ""
        reset_color = ""
        if self.use_colors:
            level_color = self.LEVEL_COLORS.get(record.levelname, "")
            reset_color = Style.RESET_ALL

        # Basic log format
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        basic_msg = f"{timestamp} - {record.name} - {level_color}{record.levelname}{reset_color} - [{record.location}] - {record.getMessage()}"

        # Add exception information if present
        if record.exc_info and self.include_traceback:
            exception_details = self._format_exception_details(record.exc_info)
            basic_msg += f"\n{exception_details}"

        return basic_msg

    def _get_function_name(self, record) -> Optional[str]:
        """Extract function name from the logging record or call stack."""
        # Try to get function name from record first
        if hasattr(record, "funcName") and record.funcName != "<module>":
            return record.funcName

        # Fallback: inspect the call stack
        try:
            frame = sys._getframe()
            while frame:
                if frame.f_code.co_filename == record.pathname:
                    return frame.f_code.co_name
                frame = frame.f_back
        except:
            pass

        return None

    def _format_exception_details(self, exc_info) -> str:
        """Format detailed exception information with traceback."""
        exc_type, exc_value, exc_traceback = exc_info

        # Get the traceback frames
        tb_frames = traceback.extract_tb(exc_traceback, limit=self.traceback_limit)

        # Color coding for exception details
        error_color = Fore.RED if self.use_colors else ""
        frame_color = Fore.CYAN if self.use_colors else ""
        source_color = Fore.YELLOW if self.use_colors else ""
        reset_color = Style.RESET_ALL if self.use_colors else ""

        lines = []
        lines.append(f"{error_color}┌─ {exc_type.__name__}: {exc_value}{reset_color}")

        # Add call stack information
        if len(tb_frames) > 1:
            lines.append(f"{frame_color}├─ Call Stack:{reset_color}")
            for i, frame in enumerate(tb_frames[:-1]):
                prefix = "│  ├─" if i < len(tb_frames) - 2 else "│  └─"
                lines.append(
                    f"{frame_color}{prefix} {frame.filename}:{frame.lineno} in {frame.name}(){reset_color}"
                )

        # Add the error location details
        if tb_frames:
            error_frame = tb_frames[-1]
            lines.append(
                f"{frame_color}├─ Error Location: {error_frame.filename}:{error_frame.lineno} in {error_frame.name}(){reset_color}"
            )

            if error_frame.line:
                lines.append(
                    f"{source_color}├─ Source: {error_frame.line.strip()}{reset_color}"
                )

        # Add local variables if requested and available
        if self.include_locals and exc_traceback:
            try:
                frame = exc_traceback.tb_frame
                local_vars = self._format_local_variables(frame.f_locals)
                if local_vars:
                    lines.append(
                        f"{frame_color}└─ Local Variables: {local_vars}{reset_color}"
                    )
                else:
                    lines.append(
                        f"{frame_color}└─ (No relevant local variables){reset_color}"
                    )
            except:
                lines.append(
                    f"{frame_color}└─ (Could not retrieve local variables){reset_color}"
                )
        else:
            # Close the box without local variables
            if lines[-1].startswith("├─"):
                lines[-1] = lines[-1].replace("├─", "└─")

        return "\n".join(lines)

    def _format_local_variables(self, local_vars: Dict[str, Any]) -> str:
        """Format local variables for display, filtering out irrelevant ones."""
        relevant_vars = {}

        # Filter out built-in and irrelevant variables
        skip_vars = {
            "__builtins__",
            "__file__",
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "__annotations__",
            "__cached__",
        }

        for name, value in local_vars.items():
            if (
                not name.startswith("_") or name in ["_", "__"]
            ) and name not in skip_vars:
                try:
                    # Limit the representation length
                    repr_value = repr(value)
                    if len(repr_value) > 100:
                        repr_value = repr_value[:97] + "..."
                    relevant_vars[name] = repr_value
                except:
                    relevant_vars[name] = "<repr failed>"

        if relevant_vars:
            var_strs = [
                f"{k}={v}" for k, v in list(relevant_vars.items())[:5]
            ]  # Limit to 5 vars
            return ", ".join(var_strs)

        return ""


def setup_enhanced_logging(
    level=logging.INFO,
    log_file=None,
    use_colors=True,
    include_traceback=True,
    traceback_limit=10,
    include_locals=False,
):
    """
    Set up enhanced logging with traceback support and colors.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to also log to a file
        use_colors: Enable colored output (default: True)
        include_traceback: Include detailed traceback for exceptions (default: True)
        traceback_limit: Maximum number of traceback frames to show (default: 10)
        include_locals: Include local variables in error reports (default: False)
    """
    # Create enhanced formatter
    console_formatter = EnhancedLocationFormatter(
        use_colors=use_colors,
        include_traceback=include_traceback,
        traceback_limit=traceback_limit,
        include_locals=include_locals,
    )

    # File formatter without colors
    file_formatter = EnhancedLocationFormatter(
        use_colors=False,
        include_traceback=include_traceback,
        traceback_limit=traceback_limit,
        include_locals=include_locals,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler without colors
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name=None):
    """
    Get a logger instance with enhanced formatting.

    Args:
        name: Logger name (defaults to calling module name)

    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module's name
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "unknown")

    return logging.getLogger(name)


def log_exception_with_context(logger, message="", exc_info=None, include_locals=False):
    """
    Log an exception with full context and traceback.

    Args:
        logger: Logger instance
        message: Additional message to include
        exc_info: Exception info tuple (default: current exception)
        include_locals: Include local variables in the output
    """
    if exc_info is None:
        exc_info = sys.exc_info()

    if exc_info[0] is None:
        logger.error(f"{message} (No exception context available)")
        return

    # Log with exception info
    full_message = message if message else "Exception occurred"
    logger.error(full_message, exc_info=exc_info)


def exception_handler(include_locals=False):
    """
    Decorator to automatically log exceptions with enhanced context.

    Args:
        include_locals: Whether to include local variables in error logs

    Usage:
        @exception_handler(include_locals=True)
        def my_function():
            # function code here
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(func.__module__)
                log_exception_with_context(
                    logger,
                    f"Exception in {func.__name__}()",
                    include_locals=include_locals,
                )
                raise

        return wrapper

    return decorator


# Convenience function for quick setup
def init_logging(
    level=logging.INFO,
    log_file=None,
    use_colors=True,
    include_traceback=True,
    include_locals=False,
):
    """Initialize enhanced logging for the entire application."""
    return setup_enhanced_logging(
        level=level,
        log_file=log_file,
        use_colors=use_colors,
        include_traceback=include_traceback,
        include_locals=include_locals,
    )


# Auto-configure logging when module is imported
if not logging.getLogger().handlers:
    init_logging()
