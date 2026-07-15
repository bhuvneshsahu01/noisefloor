"""
CLI Entrypoint for Noisefloor.
"""
import click
import json
from ..compare import compare_evals
from ..sprt import sprt_gate
from ..quant.audit import audit_backtest

@click.group()
def main():
    """Noisefloor: The convergent statistical verdict engine for AI decisions and quantitative backtesting."""
    pass

@main.command()
@click.argument('baseline_file', type=click.Path(exists=True))
@click.argument('candidate_file', type=click.Path(exists=True))
@click.option('--method', default='auto', help='Statistical method (auto, bootstrap, ttest)')
@click.option('--alpha', default=0.05, type=float, help='Significance level')
def compare(baseline_file, candidate_file, method, alpha):
    """Compare two evaluation result files."""
    def load_scores(fpath):
        scores = []
        with open(fpath, 'r') as f:
            for line in f:
                data = json.loads(line)
                scores.append(data.get("score", 0.0))
        return scores
        
    base = load_scores(baseline_file)
    cand = load_scores(candidate_file)
    
    res = compare_evals(base, cand, method=method, alpha=alpha)
    
    click.echo(f"[noisefloor] Comparing {len(base)} eval scores (paired)")
    click.echo(f"[noisefloor] Baseline: {res.baseline_mean:.3f}")
    click.echo(f"[noisefloor] Candidate: {res.candidate_mean:.3f}")
    click.echo(f"[noisefloor] Observed delta: {res.delta:+.3f}")
    click.echo(f"[noisefloor] 95% CI of delta: ({res.ci_lower:.3f}, {res.ci_upper:.3f})")
    click.echo(f"[noisefloor] p-value: {res.p_value:.3f} ({res.method_used})")
    click.echo(f"[noisefloor] ──────────────────────────────────────")
    click.echo(f"[noisefloor] VERDICT: {res.verdict}")
    click.echo(f"[noisefloor] Reason: {res.interpretation}")

if __name__ == '__main__':
    main()
