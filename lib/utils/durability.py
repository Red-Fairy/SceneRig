""

import os


def durable_fsync_enabled() -> bool:
    return os.environ.get("GRASE_DURABLE_FSYNC", "0") == "1"
