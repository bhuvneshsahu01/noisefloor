"""
Implementation Risk Audit.
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class ImplementationRiskResult:
    iui: float
    daf: float
    csi: float
    engine_sharpes: Dict[str, float]
    risk_level: str
    recommendation: str

def implementation_risk_audit(
    returns_by_engine: Dict[str, np.ndarray],
    verdict_threshold: float = 0.5,
) -> ImplementationRiskResult:
    """
    Quantifies the Implementation Risk of a strategy based on 
    divergence across backtesting engines.
    """
    if len(returns_by_engine) < 2:
        raise ValueError("Must provide returns from at least 2 engines for risk audit.")
        
    engine_sharpes = {}
    verdicts = []
    
    for engine, returns in returns_by_engine.items():
        ret = np.asarray(returns)
        mean_ret = np.mean(ret)
        std_ret = np.std(ret, ddof=1)
        sr = mean_ret / std_ret if std_ret > 0 else 0
        
        # Annualized SR approx assuming daily
        ann_sr = sr * np.sqrt(252)
        engine_sharpes[engine] = ann_sr
        verdicts.append(ann_sr > verdict_threshold)
        
    sr_values = list(engine_sharpes.values())
    
    # Implementation Uncertainty Index
    iui = float(np.std(sr_values, ddof=1))
    
    # Decision Agreement Factor
    daf = max(np.mean(verdicts), 1 - np.mean(verdicts))
    
    # Conclusion Stability Index (heuristic)
    csi = 1.0 - (iui / max(1e-5, np.mean(sr_values)))
    
    if iui > 1.0 or daf < 0.7:
        risk = "HIGH"
        rec = "Significant divergence between engines. Check slippage and fill assumptions."
    elif iui > 0.5:
        risk = "MEDIUM"
        rec = "Moderate divergence. Implementation details matter."
    else:
        risk = "LOW"
        rec = "Strong agreement across engines. Implementation risk is low."
        
    return ImplementationRiskResult(
        iui=iui,
        daf=float(daf),
        csi=float(csi),
        engine_sharpes=engine_sharpes,
        risk_level=risk,
        recommendation=rec
    )
