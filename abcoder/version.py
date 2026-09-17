"""Version information for ABC (Automated Behaviour Coder)."""

__version__ = "0.1.0"

#: Version of the job manifest / result JSON contract between client and server.
#: Bump whenever the on-disk schema changes incompatibly.
PROTOCOL_VERSION = 1

#: BORIS project format version that ABC reads and writes.
BORIS_FORMAT_VERSION = "7.0"
