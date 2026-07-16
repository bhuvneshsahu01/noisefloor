from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class EvaluationReport:
    """
    Generates a terminal-friendly or JSON statistical evaluation report.
    """
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def print_terminal_report(self):
        """Outputs the stunning Volume 15 RiskLayer terminal report."""
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        
        table.add_row("Calibration Error", f"{self.metrics.get('calibration_error', 0.0) * 100:.1f}%")
        table.add_row("Coverage", f"{self.metrics.get('coverage', 0.0) * 100:.1f}%")
        
        guaranteed = "YES" if self.metrics.get("guaranteed_coverage") else "NO"
        guaranteed_style = "[bold green]YES[/bold green]" if guaranteed == "YES" else "[bold red]NO[/bold red]"
        table.add_row("Guaranteed Coverage", guaranteed_style)
        
        drift = self.metrics.get("distribution_shift", "UNKNOWN")
        if drift == "LOW":
            drift_style = "[bold green]LOW[/bold green]"
        elif drift == "MEDIUM":
            drift_style = "[bold yellow]MEDIUM[/bold yellow]"
        else:
            drift_style = "[bold red]HIGH[/bold red]"
            
        table.add_row("Distribution Shift", drift_style)
        table.add_row("Conformal Radius", f"{self.metrics.get('conformal_radius', 0.0):.2f}")
        table.add_row("Expected Risk", f"{self.metrics.get('expected_risk', 0.0):.3f}")
        
        console.print("\n")
        console.print(table)
        
        console.print("\n[bold cyan]Recommendation[/bold cyan]")
        
        if guaranteed == "YES" and drift in ["LOW", "MEDIUM"]:
            console.print(Panel(Text("SAFE TO DEPLOY", justify="center", style="bold green"), border_style="green"))
        else:
            console.print(Panel(Text("DO NOT DEPLOY", justify="center", style="bold red"), border_style="red"))
        
    def to_dict(self):
        return self.metrics
