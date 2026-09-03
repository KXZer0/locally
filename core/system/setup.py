"""Public setup API assembled from discovery and mutation modules."""

from core.system.setup_actions import (setup_finish, setup_install,
                                       setup_install_status)
from core.system.setup_info import _read_setup_config, setup_state

__all__ = ["_read_setup_config", "setup_finish", "setup_install",
           "setup_install_status", "setup_state"]
