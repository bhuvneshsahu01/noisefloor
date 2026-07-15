import base64
from typing import Dict, Any, Tuple, Optional
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

class AuditSigner:
    """
    Ed25519 cryptographic signer to verify that statistical verdicts 
    have not been tampered with. Crucial for compliance (SEBI/EU AI Act).
    """
    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]:
        """Generates a new private/public keypair in PEM/RAW bytes."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize keys
        private_bytes = private_key.private_bytes_raw()
        public_bytes = public_key.public_bytes_raw()
        return private_bytes, public_bytes

    def __init__(self, private_key_raw: Optional[bytes] = None):
        """
        Initialize the signer. If private_key_raw is omitted, the signer 
        will run in verification-only mode or sign with a transient key.
        """
        self.private_key = None
        if private_key_raw:
            self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_raw)

    def sign_verdict(self, verdict_data: Dict[str, Any]) -> str:
        """
        Signs the verdict payload and returns a base64 encoded signature.
        """
        if not self.private_key:
            raise ValueError("Signer initialized without a private key. Cannot sign.")
            
        # Serialize fields deterministically
        serialized = self._serialize_payload(verdict_data)
        signature = self.private_key.sign(serialized)
        return base64.b64encode(signature).decode('utf-8')

    def verify_verdict(self, verdict_data: Dict[str, Any], signature_b64: str, public_key_raw: bytes) -> bool:
        """
        Verifies the signature of the verdict data using the public key.
        """
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw)
            serialized = self._serialize_payload(verdict_data)
            signature = base64.b64decode(signature_b64)
            public_key.verify(signature, serialized)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

    def _serialize_payload(self, data: Dict[str, Any]) -> bytes:
        """Helper to serialize payload dict keys sorted to prevent hashing mismatches."""
        # Convert dictionary to a sorted JSON string
        import json
        serialized_str = json.dumps(data, sort_keys=True)
        return serialized_str.encode('utf-8')
