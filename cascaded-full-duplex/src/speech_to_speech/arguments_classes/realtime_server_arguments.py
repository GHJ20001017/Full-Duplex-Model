from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RealtimeServerArguments:
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": "Host interface for the Realtime server. Default is 127.0.0.1. Pass 0.0.0.0 explicitly "
            "to expose the unauthenticated API on the network."
        },
    )
    port: int = field(
        default=7869,
        metadata={"help": "Port for the Realtime HTTP/WebSocket server. Default is 7869."},
    )
    interruption_route: Literal["keyword", "semantic"] = field(
        default="keyword",
        metadata={
            "help": "Interruption routing policy shared by every Realtime session. "
            "The client cannot override this server setting."
        },
    )


@dataclass
class LocalRealtimeServerArguments:
    port: int = field(
        default=7869,
        metadata={"help": "Loopback port for the local Realtime server and audio client. Default is 7869."},
    )
