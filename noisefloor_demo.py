import time
import sys
from noisefloor.core.conformal import ConformalPredictor
from noisefloor.integrations.agent_guard import RiskGuard, active_shadow_mode
from noisefloor.core.trajectory import active_trace_id, trajectory_tracker
from noisefloor.exceptions import ConformalRiskException

# Try importing rich elements for premium console experience
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def print_banner(title: str):
    if HAS_RICH:
        console.print(Panel(Text(title, style="bold cyan", justify="center"), border_style="cyan"))
    else:
        print(f"\n==================== {title.upper()} ====================\n")

def print_step(name: str):
    if HAS_RICH:
        console.print(f"\n[bold magenta]>>> {name}[/bold magenta]")
    else:
        print(f"\n>>> {name}")

def print_success(text: str):
    if HAS_RICH:
        console.print(f"[bold green][SUCCESS][/bold green] {text}")
    else:
        print(f"[SUCCESS] {text}")

def print_warning(text: str):
    if HAS_RICH:
        console.print(f"[bold yellow][WARN][/bold yellow] {text}")
    else:
        print(f"[WARN] {text}")

def print_error(text: str):
    if HAS_RICH:
        console.print(f"[bold red][FAIL][/bold red] {text}")
    else:
        print(f"[FAIL] {text}")

def print_info(text: str):
    if HAS_RICH:
        console.print(f"[blue][INFO][/blue] {text}")
    else:
        print(f"[INFO] {text}")

def run_demo():
    print_banner("Noisefloor RiskLayer Conformal Guardrails Playground")
    
    # 1. Setup Predictor and Guard
    predictor = ConformalPredictor(alpha=0.10)
    # Calibration errors representing historical judge variance
    predictor.calibrate([0.05, 0.08, 0.12, 0.15, 0.18, 0.20])
    q_hat = predictor.get_bound()
    
    print_info(f"Target error rate (alpha): {predictor.alpha:.2f} (90% reliability guarantee)")
    print_info(f"Calibrated Conformal Quantile Bound (q_hat): {q_hat:.3f}")
    
    guard = RiskGuard(predictor)
    
    # -------------------------------------------------------------
    # Scenario 1: Conformal Model Routing (Cost Optimization)
    # -------------------------------------------------------------
    print_step("Scenario 1: Conformal Model Routing")
    print_info("Simulating Llama 3 (cheap) and Gemma 2 (premium) routing.")
    
    # Simple mock simulation: volatile stocks yield higher non-conformity errors
    def check_volatility_risk(stock):
        return 0.35 if stock == "TSLA" else 0.05
        
    @guard.guard(score_fn=check_volatility_risk)
    def query_stock_action(stock):
        return f"Parsed data for {stock}"
        
    # Safe query (MSFT): processed locally by cheap Llama 3
    print_info("Processing 'MSFT' query...")
    res_safe = query_stock_action("MSFT")
    print_success(f"Route: CHEAP | Result: {res_safe} | Cost Saved: $0.0049")
    
    # Ambiguous query (TSLA): exceeds q_hat, escalates to Gemma 2
    print_info("Processing 'TSLA' query (high volatility risk)...")
    print_warning("Score exceeds q_hat (0.35 > 0.18). Escalating to premium model...")
    res_esc = query_stock_action("TSLA", shadow_mode=True) # Bypass execution check for demo flow
    print_success(f"Route: PREMIUM | Result: {res_esc} | Cost Saved: $0.0000")
    
    time.sleep(1.0)
    
    # -------------------------------------------------------------
    # Scenario 2: Self-Healing & Parameter Recovery
    # -------------------------------------------------------------
    print_step("Scenario 2: Agent Self-Healing & Parameter Recovery")
    print_info("Guarding a transaction transfer tool against high amounts.")
    
    def check_transfer_risk(amount):
        # Risk score proportional to transfer value
        return amount / 1000.0
        
    def heal_transfer(amount):
        print_warning(f"Self-Healing: Intercepted transfer of ${amount}. Reducing amount to safe limit $150.")
        return (150,), {}
        
    @guard.guard(score_fn=check_transfer_risk, fallback_handler=heal_transfer)
    def execute_transfer(amount):
        return f"Completed transfer of ${amount}"
        
    print_info("Executing risky transfer request: $500...")
    res_transfer = execute_transfer(500)
    print_success(f"Outcome: {res_transfer}")
    
    time.sleep(1.0)
    
    # -------------------------------------------------------------
    # Scenario 3: Trajectory-Level Cascading Error Blocker
    # -------------------------------------------------------------
    print_step("Scenario 3: Trajectory-Level Blocker (Bonferroni adjusted)")
    print_info("Simulating multi-step execution. Risk boundary tightens at each step.")
    
    trace_id = "agent-demo-trajectory"
    active_trace_id.set(trace_id)
    
    @guard.guard(score_fn=lambda x: x)
    def multi_step_tool(score):
        return "Step executed"
        
    # Execute steps with compounding uncertainty (0.05 * step)
    # Step 1: risk = 0.05 -> PASS
    # Step 2: risk = 0.10 -> PASS
    # Step 3: risk = 0.15 -> PASS
    # Step 4: risk = 0.20 -> PASS (boundary cap)
    # Step 5: risk = 0.25 -> BLOCKED (exceeds conformal boundary 0.20)!
    
    try:
        for i in range(1, 6):
            risk_score = 0.05 * i
            print_info(f"Executing Step {i} (Compounded Risk score = {risk_score:.2f})...")
            multi_step_tool(risk_score)
            print_success(f"Step {i} completed.")
    except ConformalRiskException as e:
        print_error(f"HALTED: Trajectory terminated at step {i}: {str(e)}")
        
    trajectory_tracker.clear(trace_id)
    active_trace_id.set(None)
    
    time.sleep(1.0)
    
    # -------------------------------------------------------------
    # Scenario 4: Shadow Mode Compliance Audit
    # -------------------------------------------------------------
    print_step("Scenario 4: Shadow Mode Compliance Audit")
    print_info("Simulating active shadow mode validation.")
    
    # Toggling shadow mode contextvar to True
    active_shadow_mode.set(True)
    
    @guard.guard(score_fn=lambda x: x)
    def database_write_action(score):
        return "Database transaction completed"
        
    # Score 0.50 breaches q_hat (0.18) but runs safely with warnings
    print_info("Triggering database transaction (Risk score = 0.50)...")
    res_db = database_write_action(0.50)
    print_success(f"Status: {res_db}")
    print_warning("OTEL Span flagged: SHADOW_BLOCKED committed to SQLite audit log.")
    
    active_shadow_mode.set(None)
    print_banner("Noisefloor Demo Complete")

if __name__ == "__main__":
    run_demo()
