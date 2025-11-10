import dataclasses
import enum
import logging
import socket
import tyro

import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.serving.websocket_policy_server as websocket_policy_server

import aera.autonomous.openpi.training_config as _training_config


class EnvMode(enum.Enum):
    """Supported environments."""
    AR4_MK3 = "ar4_mk3"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    # Training config name (e.g., "ar4_mk3").
    config: str
    # Checkpoint directory (e.g., "checkpoints/ar4_mk3/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    # Environment to serve the policy for.
    env: EnvMode = EnvMode.AR4_MK3
    
    # If provided, will be used in case the "prompt" key is not present in the data,
    # or if the model doesn't have a default prompt.
    default_prompt: str | None = None
    
    # Port to serve the policy on.
    port: int = 8000
    
    # Record the policy's behavior for debugging.
    record: bool = False
    
    # Checkpoint configuration to load the policy from.
    checkpoint: Checkpoint | None = None


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.checkpoint is None:
        raise ValueError("Checkpoint must be provided. No default checkpoints are configured.")
    
    config = _training_config.get_config(args.checkpoint.config)
    return _policy_config.create_trained_policy(
        config, args.checkpoint.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
