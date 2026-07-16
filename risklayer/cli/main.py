import argparse
import sys
import os
import subprocess
import json
from risklayer import RiskGuard

def serve(args):
    """Start the RiskLayer FastAPI server."""
    print("Starting RiskLayer Enterprise Server...")
    try:
        # Use uvicorn to run the FastAPI app
        subprocess.run(["uvicorn", "risklayer.server.api:app", "--host", "0.0.0.0", "--port", "8000"])
    except KeyboardInterrupt:
        print("\nServer stopped.")

def eval_dataset(args):
    """Run a batch evaluation from a JSON dataset."""
    print(f"Evaluating dataset: {args.dataset}")
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)
        
    with open(args.dataset, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Dataset must be valid JSON.")
            sys.exit(1)
            
    reference_scores = data.get("reference", [])
    production_scores = data.get("production", [])
    
    if not reference_scores or not production_scores:
        print("Error: Dataset must contain 'reference' and 'production' float arrays.")
        sys.exit(1)
        
    guard = RiskGuard(target_alpha=args.alpha)
    guard.fit(reference_scores)
    report = guard.evaluate(production_scores)
    
    report.print_terminal_report()

def verify_trace(args):
    """Verify cryptographic signature of a trace using the server."""
    import urllib.request
    import urllib.error
    
    print(f"Verifying Trace ID: {args.trace_id}")
    url = f"{args.server}/api/v1/verify"
    
    data = json.dumps({"trace_id": str(args.trace_id)}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            if result.get("integrity_valid"):
                print("[SUCCESS] Trace signature is valid. Data has not been tampered with.")
            else:
                print("[ALERT] Trace signature is INVALID. Possible tampering detected.")
    except urllib.error.HTTPError as e:
        print(f"Server Error: {e.code} - {e.reason}")
    except Exception as e:
        print(f"Connection Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="RiskLayer CLI: The Statistical Operating System for AI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the RiskLayer server")
    
    # eval command
    eval_parser = subparsers.add_parser("eval", help="Run a batch evaluation from a JSON dataset")
    eval_parser.add_argument("dataset", type=str, help="Path to JSON dataset containing reference and production scores")
    eval_parser.add_argument("--alpha", type=float, default=0.05, help="Target alpha (risk tolerance)")
    
    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify the cryptographic signature of a trace")
    verify_parser.add_argument("trace_id", type=str, help="Trace ID to verify")
    verify_parser.add_argument("--server", type=str, default="http://localhost:8000", help="RiskLayer Server URL")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve(args)
    elif args.command == "eval":
        eval_dataset(args)
    elif args.command == "verify":
        verify_trace(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
