"""
CLI Entrypoint for risklayer.
"""
import click
import json
from ..compare import compare_evals
from ..sprt import sprt_gate
from ..quant.audit import audit_backtest

@click.group()
def main():
    """risklayer: The convergent statistical verdict engine for AI decisions and quantitative backtesting."""
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
    
    click.echo(f"[risklayer] Comparing {len(base)} eval scores (paired)")
    click.echo(f"[risklayer] Baseline: {res.baseline_mean:.3f}")
    click.echo(f"[risklayer] Candidate: {res.candidate_mean:.3f}")
    click.echo(f"[risklayer] Observed delta: {res.delta:+.3f}")
    click.echo(f"[risklayer] 95% CI of delta: ({res.ci_lower:.3f}, {res.ci_upper:.3f})")
    click.echo(f"[risklayer] p-value: {res.p_value:.3f} ({res.method_used})")
    click.echo(f"[risklayer] ──────────────────────────────────────")
    click.echo(f"[risklayer] VERDICT: {res.verdict}")
    click.echo(f"[risklayer] Reason: {res.interpretation}")

@main.command()
@click.argument('dataset_file', type=click.Path(exists=True))
@click.option('--output', default='conformal_calibration.json', help='Where to save the JSON profile')
@click.option('--provider', default=None, help='LLM provider (groq, openrouter)')
def autotune(dataset_file, output, provider):
    """Auto-tune conformal calibration profiles from benchmark datasets."""
    from ..judges.llm import LiveLLMJudge
    from ..core.autotune import AutoTuner
    
    click.echo(f"[risklayer] Loading LLM judge (provider={provider or 'default'})...")
    try:
        judge = LiveLLMJudge(provider=provider)
    except Exception as e:
        click.echo(f"[error] Failed to initialize live LLM: {str(e)}")
        click.echo("[info] Initializing mock judge for local validation...")
        class MockJudge:
            def evaluate_correctness(self, q, r): return 0.85
        judge = MockJudge()
        
    tuner = AutoTuner(judge)
    click.echo(f"[risklayer] Running evaluations on {dataset_file}...")
    scores = tuner.tune(dataset_file)
    
    # Save the calibration profile to JSON
    profile = {
        "provider": provider or "default",
        "calibration_scores": scores
    }
    with open(output, "w") as f:
        json.dump(profile, f, indent=2)
        
    click.echo(f"[risklayer] Auto-tuning completed successfully!")
    click.echo(f"[risklayer] Calibration profile saved to: {output}")
    click.echo(f"[risklayer] Dataset size: {len(scores)} samples.")
    if scores:
        click.echo(f"[risklayer] Average error score: {sum(scores)/len(scores):.4f}")

if __name__ == '__main__':
    main()

