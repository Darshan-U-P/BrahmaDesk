"""BrahmaDesk integration adapters for the Universal LLM ToolKit."""

from .config import ULTKSettings
from .provider import BrahmaEngineProvider

__all__ = ["BrahmaEngineProvider", "ULTKSettings"]
