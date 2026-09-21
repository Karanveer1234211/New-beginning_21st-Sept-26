"""Does the static checker CATCH leaks, or is it just quiet?"""
import sys, types, tempfile, textwrap
from pathlib import Path
for n in ("kiteconnect","kiteconnect.exceptions"): sys.modules[n]=types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect=object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"],n,type(n,(Exception,),{}))
import importlib.util
sp=importlib.util.spec_from_file_location("fd","features_daily.py"); fd=importlib.util.module_from_spec(sp)
sys.modules["fd"]=fd; sp.loader.exec_module(fd)

LEAKS = {
  "whole-series rank":      "df['x'] = s.rank(pct=True)",
  "whole-series mean":      "df['x'] = s.mean()",
  "whole-series std":       "df['x'] = s.std()",
  "whole-series quantile":  "df['x'] = s.quantile(0.9)",
  "whole-series max":       "df['x'] = s.max()",
  "whole-series corr":      "df['x'] = s.corr(t)",
  "negative shift":         "df['x'] = s.shift(-3)",
  "negative diff":          "df['x'] = s.diff(-1)",
  "backfill":               "df['x'] = s.bfill()",
  "fillna bfill":           "df['x'] = s.fillna(method='bfill')",
  "centred rolling":        "df['x'] = s.rolling(20, center=True).mean()",
  "interpolate both":       "df['x'] = s.interpolate(limit_direction='both')",
  "series reversal":        "df['x'] = s[::-1]",
}
CAUSAL = {
  "rolling mean":           "df['x'] = s.rolling(20).mean()",
  "expanding rank":         "df['x'] = s.expanding(min_periods=60).rank(pct=True)",
  "ewm mean":               "df['x'] = s.ewm(span=20, adjust=False).mean()",
  "window via variable":    "r = s.rolling(20)\n    df['x'] = r.mean()\n    df['y'] = r.std()",
  "row-wise max (axis=1)":  "df['x'] = pd.concat([s, t], axis=1).max(axis=1)",
  "row-wise min kwarg":     "df['x'] = pd.concat([s, t], axis=1).min(axis=1)",
  "positive shift":         "df['x'] = s.shift(1)",
  "forward fill":           "df['x'] = s.ffill()",
  "groupby rank":           "df['x'] = df.groupby('k')['v'].rank(pct=True)",
  "cumsum (OBV)":           "df['x'] = (s.diff() * t).cumsum()",
  "annotated exception":    "df['x'] = s.shift(-5)  # leak-free: the label",
}
def run(body):
    src = "import pandas as pd\ndef compute_daily_indicators(df):\n    " + \
          body.replace("\n", "\n") + "\n    return df\n"
    p = Path(tempfile.mkdtemp())/"probe.py"; p.write_text(src, encoding="utf-8")
    try:
        fd.static_leak_check(path=p); return None
    except RuntimeError as e:
        return str(e).splitlines()[1].strip() if len(str(e).splitlines())>1 else "flagged"

bad=[]
print("MUST BE CAUGHT:")
for name, code in LEAKS.items():
    r = run(code)
    print(f"  {'ok  ' if r else 'MISS'} {name:<24} {r or '<<< NOT DETECTED >>>'}")
    if not r: bad.append(f"missed: {name}")
print("\nMUST NOT BE FLAGGED:")
for name, code in CAUSAL.items():
    r = run(code)
    print(f"  {'ok  ' if not r else 'FALSE+'} {name:<24} {r or ''}")
    if r: bad.append(f"false positive: {name}")
print("\n" + ("FAILURES: " + "; ".join(bad) if bad else "LEAK CHECKER VERIFIED: 13 caught, 11 clean"))
