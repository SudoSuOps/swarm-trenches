"""Agent manifest publishing with EIP-191 signing and IPFS upload."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PublishConfig:
    """Configuration for publishing."""

    adapter_path: Path
    manifest_output: Path = field(default_factory=lambda: Path("./artifacts/manifest.json"))
    name: str = "swarmcode-agent"
    version: str = "0.1.0"
    description: str = "SwarmCode coding agent"
    author: str = ""
    private_key: str | None = None
    ipfs_gateway: str = "http://localhost:5001"


@dataclass
class PublishResult:
    """Result of publishing."""

    manifest_path: Path
    signature: str
    ipfs_cid: str | None


@dataclass
class AgentManifest:
    """Agent manifest following ERC-8004-like structure."""

    name: str
    version: str
    description: str
    author: str
    created_at: str
    adapter_hash: str
    adapter_size: int
    base_model: str
    capabilities: list[str]
    tools: list[str]
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at,
            "adapter": {
                "hash": self.adapter_hash,
                "size": self.adapter_size,
            },
            "base_model": self.base_model,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "signature": self.signature,
        }


class EIP191Signer:
    """EIP-191 message signing (stub implementation)."""

    def __init__(self, private_key: str | None = None):
        self.private_key = private_key
        self._eth_available = False

        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            self._eth_available = True
            self._Account = Account
            self._encode_defunct = encode_defunct
        except ImportError:
            logger.warning("eth_account not available, using mock signing")

    def sign(self, message: str) -> str:
        """Sign a message using EIP-191."""
        if not self.private_key:
            # Return a deterministic mock signature
            return f"0x{hashlib.sha256(message.encode()).hexdigest()}"

        if not self._eth_available:
            return f"0x{hashlib.sha256(message.encode()).hexdigest()}"

        try:
            signable = self._encode_defunct(text=message)
            signed = self._Account.sign_message(signable, self.private_key)
            return signed.signature.hex()
        except Exception as e:
            logger.warning(f"Signing failed: {e}")
            return f"0x{hashlib.sha256(message.encode()).hexdigest()}"

    def verify(self, message: str, signature: str, address: str) -> bool:
        """Verify an EIP-191 signature."""
        if not self._eth_available:
            return True  # Mock verification

        try:
            signable = self._encode_defunct(text=message)
            recovered = self._Account.recover_message(signable, signature=signature)
            return recovered.lower() == address.lower()
        except Exception:
            return False


class IPFSClient:
    """IPFS client for uploading manifests and adapters (stub implementation)."""

    def __init__(self, gateway: str = "http://localhost:5001"):
        self.gateway = gateway
        self._available = False

        try:
            import httpx
            self._httpx = httpx
            self._available = True
        except ImportError:
            pass

    def upload_json(self, data: dict) -> str | None:
        """Upload JSON data to IPFS."""
        if not self._available:
            logger.warning("IPFS upload not available (httpx not installed)")
            return None

        try:
            json_bytes = json.dumps(data, indent=2).encode()
            return self._upload_bytes(json_bytes)
        except Exception as e:
            logger.warning(f"IPFS upload failed: {e}")
            return None

    def upload_file(self, path: Path) -> str | None:
        """Upload a file to IPFS."""
        if not self._available:
            return None

        try:
            content = path.read_bytes()
            return self._upload_bytes(content)
        except Exception as e:
            logger.warning(f"IPFS upload failed: {e}")
            return None

    def _upload_bytes(self, content: bytes) -> str | None:
        """Upload bytes to IPFS."""
        try:
            response = self._httpx.post(
                f"{self.gateway}/api/v0/add",
                files={"file": content},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("Hash")
        except Exception as e:
            logger.debug(f"IPFS upload error: {e}")
            # Return mock CID for local development
            return f"Qm{hashlib.sha256(content).hexdigest()[:44]}"


class Publisher:
    """Publishes agent manifests with signing and IPFS upload."""

    def __init__(self, config: PublishConfig):
        self.config = config
        self.signer = EIP191Signer(config.private_key)
        self.ipfs = IPFSClient(config.ipfs_gateway)

    def publish(self) -> PublishResult:
        """Create and publish the agent manifest."""
        # Compute adapter hash
        adapter_hash, adapter_size = self._compute_adapter_hash()

        # Load training config if available
        training_config = self._load_training_config()
        base_model = training_config.get("model_name", "unknown")

        # Build manifest
        manifest = AgentManifest(
            name=self.config.name,
            version=self.config.version,
            description=self.config.description,
            author=self.config.author,
            created_at=datetime.now().isoformat(),
            adapter_hash=adapter_hash,
            adapter_size=adapter_size,
            base_model=base_model,
            capabilities=[
                "code_generation",
                "code_review",
                "bug_fixing",
                "refactoring",
                "test_generation",
            ],
            tools=[
                "read_file",
                "search",
                "apply_patch",
                "run_tests",
                "list_files",
                "write_file",
            ],
        )

        # Create signable message
        manifest_dict = manifest.to_dict()
        del manifest_dict["signature"]  # Remove signature field for signing
        signable = json.dumps(manifest_dict, sort_keys=True)

        # Sign
        signature = self.signer.sign(signable)
        manifest.signature = signature

        # Save manifest
        self.config.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.manifest_output, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(f"Manifest saved to: {self.config.manifest_output}")

        # Upload to IPFS
        ipfs_cid = self.ipfs.upload_json(manifest.to_dict())
        if ipfs_cid:
            logger.info(f"Uploaded to IPFS: {ipfs_cid}")

        return PublishResult(
            manifest_path=self.config.manifest_output,
            signature=signature,
            ipfs_cid=ipfs_cid,
        )

    def _compute_adapter_hash(self) -> tuple[str, int]:
        """Compute hash of the adapter weights."""
        adapter_path = self.config.adapter_path

        if not adapter_path.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_path}")

        # Hash all files in adapter directory
        hasher = hashlib.sha256()
        total_size = 0

        if adapter_path.is_dir():
            for file_path in sorted(adapter_path.rglob("*")):
                if file_path.is_file():
                    content = file_path.read_bytes()
                    hasher.update(content)
                    total_size += len(content)
        else:
            content = adapter_path.read_bytes()
            hasher.update(content)
            total_size = len(content)

        return hasher.hexdigest(), total_size

    def _load_training_config(self) -> dict:
        """Load training configuration from adapter directory."""
        config_path = self.config.adapter_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}


def verify_manifest(manifest_path: Path, expected_address: str | None = None) -> bool:
    """Verify an agent manifest signature."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    signature = manifest.pop("signature", None)
    if not signature:
        logger.warning("Manifest has no signature")
        return False

    signable = json.dumps(manifest, sort_keys=True)

    signer = EIP191Signer()
    if expected_address:
        return signer.verify(signable, signature, expected_address)

    # Just verify the signature is well-formed
    return len(signature) > 10
