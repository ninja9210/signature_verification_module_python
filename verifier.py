import itertools
import numpy as np
from .preprocess import preprocess
from .scoring import centroid_align, all_scores, KEYS

THRESHOLD = -0.36            # evaluate2.py (unseen persons) se
SINGLE_REF_THRESHOLD = 0.45  # ANDAZA hai, abhi tune nahi hua

def _sigmoid_percent(x):
    return int(round(100 / (1 + np.exp(-x))))

def verify_signature(refs, new_path, threshold=THRESHOLD):
    try:
        if isinstance(refs, str):
            refs = [refs]
        ref_imgs = [centroid_align(preprocess(p)) for p in refs]
        img = centroid_align(preprocess(new_path))
        sc = [all_scores(r, img) for r in ref_imgs]
        m = {k: float(np.mean([s[k] for s in sc])) for k in KEYS}

        if len(ref_imgs) >= 3:
            pair = [all_scores(a, b) for a, b in itertools.combinations(ref_imgs, 2)]
            mu = {k: np.mean([s[k] for s in pair]) for k in KEYS}
            sd = {k: np.std([s[k] for s in pair]) + 1e-3 for k in KEYS}
            z = float(np.mean([(m[k] - mu[k]) / sd[k] for k in KEYS]))
            sim = _sigmoid_percent(1.5 * (z - threshold))   # threshold par 50%
            ok = z >= threshold
        else:   
            raw = float(np.mean([m[k] for k in KEYS]))
            sim = _sigmoid_percent(10 * (raw - SINGLE_REF_THRESHOLD))
            ok = raw >= SINGLE_REF_THRESHOLD

        return {"similarity": sim,
                "result": "GENUINE" if ok else "FORGED",
                "references_used": len(ref_imgs),
                "error": None}
    except Exception as e:
        return {"similarity": None, "result": None,
                "references_used": 0, "error": str(e)}

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) < 2:
        print("Use: python signature_verifier.py ref1.png [ref2.png ...] new.png")
        sys.exit()
    r = verify_signature(args[:-1], args[-1])
    if r["error"]:
        print("Error:", r["error"])
    else:
        print(f"Similarity: {r['similarity']}%")
        print(f"Result: {r['result']}")
        print(f"(References used: {r['references_used']})")