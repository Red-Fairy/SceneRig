"""One switch for fsync of recovery artifacts (2026-09-15, owner).

The transaction WAL, its snapshots and the authored-object capture files were written
with fsync on file AND directory after every publish. That defends against power loss;
the failure this pipeline actually meets is a process crash, which the page cache
survives, and ``os.replace`` alone keeps every published file whole. On the Lustre
mount one fsync costs 0.5-2.7 s (measured 09-15) and a transaction's prepare + commit
issued 20-40 of them: 9-122 s per transaction. Set ``GRASE_DURABLE_FSYNC=1`` to get the
old behaviour back.
"""

import os


def durable_fsync_enabled() -> bool:
    return os.environ.get("GRASE_DURABLE_FSYNC", "0") == "1"
