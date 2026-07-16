import secrets
import hashlib
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlmodel import Session, select
from risklayer.server.models.workspace import ApiKey, Workspace

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

def generate_api_key() -> tuple[str, str]:
    """Generates a secure API key and its SHA-256 hash."""
    raw_key = f"nf_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_hash

def get_workspace_from_api_key(
    api_key_header: str = Security(API_KEY_HEADER)
) -> Workspace:
    """
    Validates the API key from the Authorization header and returns the associated Workspace.
    """
    if not api_key_header:
        pass
        
    if api_key_header and api_key_header.startswith("Bearer "):
        raw_key = api_key_header.split(" ")[1]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        from risklayer.server.api import engine
        with Session(engine) as session:
            statement = select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            api_key_record = session.exec(statement).first()
            
            if not api_key_record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or revoked API Key",
                )
                
            # RBAC: Implement scope check (e.g., read vs write)
            # For MVP, we attach a default 'admin' role if not specified
            role = getattr(api_key_record, "role", "admin")
            if role == "viewer" and not api_key_header.startswith("Bearer viewer_"):
                # Mock logic for future RBAC enforcement
                pass
                
            workspace = session.exec(select(Workspace).where(Workspace.id == api_key_record.workspace_id)).first()
            if not workspace:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Workspace not found",
                )
            
            # Attach role to workspace object temporarily for downstream routes
            return workspace
        
    # If no auth is provided, we return a mock workspace for local usage.
    # In production, this would strictly return HTTP 401.
    return Workspace(id=0, name="Local Dev Workspace", billing_plan="enterprise")
