"""Tests for publish module."""

import json
import tempfile
from pathlib import Path

import pytest

from swarmcode_registry.publish import (
    AgentManifest,
    EIP191Signer,
    IPFSClient,
    PublishConfig,
    Publisher,
    verify_manifest,
)


@pytest.fixture
def adapter_dir():
    """Create a temporary adapter directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = Path(tmpdir)

        # Create mock adapter files
        (adapter_path / "adapter_config.json").write_text('{"r": 64}')
        (adapter_path / "adapter_model.bin").write_bytes(b"mock weights" * 1000)
        (adapter_path / "training_config.json").write_text(json.dumps({
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "lora_r": 64,
        }))

        yield adapter_path


class TestAgentManifest:
    """Tests for AgentManifest."""

    def test_to_dict(self):
        manifest = AgentManifest(
            name="test-agent",
            version="1.0.0",
            description="Test agent",
            author="test",
            created_at="2024-01-01T00:00:00",
            adapter_hash="abc123",
            adapter_size=12345,
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            capabilities=["code_generation"],
            tools=["read_file"],
            signature="0xsig",
        )

        data = manifest.to_dict()

        assert data["name"] == "test-agent"
        assert data["version"] == "1.0.0"
        assert data["adapter"]["hash"] == "abc123"
        assert data["adapter"]["size"] == 12345
        assert "code_generation" in data["capabilities"]
        assert data["signature"] == "0xsig"


class TestEIP191Signer:
    """Tests for EIP-191 signing."""

    def test_sign_without_key(self):
        signer = EIP191Signer()
        signature = signer.sign("test message")

        # Should return a deterministic mock signature
        assert signature.startswith("0x")
        assert len(signature) > 10

    def test_sign_deterministic(self):
        signer = EIP191Signer()
        sig1 = signer.sign("test message")
        sig2 = signer.sign("test message")

        # Same message should produce same mock signature
        assert sig1 == sig2

    def test_sign_different_messages(self):
        signer = EIP191Signer()
        sig1 = signer.sign("message 1")
        sig2 = signer.sign("message 2")

        # Different messages should produce different signatures
        assert sig1 != sig2


class TestIPFSClient:
    """Tests for IPFS client."""

    def test_upload_json_mock(self):
        client = IPFSClient("http://localhost:5001")

        # Should return mock CID when IPFS not available
        cid = client.upload_json({"key": "value"})

        assert cid is None or cid.startswith("Qm")


class TestPublisher:
    """Tests for Publisher."""

    def test_publish_creates_manifest(self, adapter_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            manifest_path = Path(output_dir) / "manifest.json"

            config = PublishConfig(
                adapter_path=adapter_dir,
                manifest_output=manifest_path,
                name="test-agent",
                version="1.0.0",
                description="Test description",
                author="test-author",
            )

            publisher = Publisher(config)
            result = publisher.publish()

            assert result.manifest_path.exists()
            assert result.signature is not None
            assert len(result.signature) > 10

    def test_publish_manifest_content(self, adapter_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            manifest_path = Path(output_dir) / "manifest.json"

            config = PublishConfig(
                adapter_path=adapter_dir,
                manifest_output=manifest_path,
                name="my-agent",
                version="2.0.0",
            )

            publisher = Publisher(config)
            publisher.publish()

            # Load and verify manifest
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["name"] == "my-agent"
            assert manifest["version"] == "2.0.0"
            assert "adapter" in manifest
            assert "hash" in manifest["adapter"]
            assert "size" in manifest["adapter"]
            assert "signature" in manifest

    def test_publish_includes_training_config(self, adapter_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            manifest_path = Path(output_dir) / "manifest.json"

            config = PublishConfig(
                adapter_path=adapter_dir,
                manifest_output=manifest_path,
            )

            publisher = Publisher(config)
            publisher.publish()

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Should load base_model from training_config.json
            assert "Qwen" in manifest["base_model"]

    def test_publish_nonexistent_adapter(self):
        config = PublishConfig(
            adapter_path=Path("/nonexistent/adapter"),
            manifest_output=Path("./manifest.json"),
        )

        publisher = Publisher(config)

        with pytest.raises(ValueError, match="does not exist"):
            publisher.publish()


class TestVerifyManifest:
    """Tests for manifest verification."""

    def test_verify_manifest(self, adapter_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            manifest_path = Path(output_dir) / "manifest.json"

            config = PublishConfig(
                adapter_path=adapter_dir,
                manifest_output=manifest_path,
            )

            publisher = Publisher(config)
            publisher.publish()

            # Verify the manifest
            result = verify_manifest(manifest_path)
            assert result is True

    def test_verify_manifest_no_signature(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "test", "version": "1.0.0"}, f)
            f.flush()

            result = verify_manifest(Path(f.name))
            assert result is False
