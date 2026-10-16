"""Linear-model fits in parallel on Modal CPUs (owner, 2026-10-16). Input: lin/bundle.age from branch lin-input — an
age-encrypted tar with the model code and the training texts — decrypted only inside the containers with REQ_AGE_KEY
(ephemeral secret for this run). Each container runs one fit (out-of-fold fold, cv split or final model); every result
is encrypted to recipient.txt before it leaves the container. usage (GitHub Actions): modal run modal_jobs/linear.py"""
import io, os, tarfile
import modal

app = modal.App("hebai-linear")
image = modal.Image.debian_slim(python_version="3.11").pip_install("scikit-learn==1.5.2", "scipy==1.14.1", "numpy==1.26.4",
                                                                    "pyrage==1.2.3")
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})


@app.function(image=image, cpu=2.0, memory=24576, timeout=45 * 60, secrets=[secret])
def fit(bundle: bytes, recipient: str, job: str) -> tuple:
    import pickle, sys, tempfile, time
    import numpy as np
    from pyrage import decrypt, encrypt, x25519
    t0 = time.time()
    d = tempfile.mkdtemp()
    tarfile.open(fileobj=io.BytesIO(decrypt(bundle, [x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())]))).extractall(d)
    sys.path.insert(0, d)
    from hebai.scorer import Scorer
    D = pickle.load(open(f"{d}/data.pkl", "rb"))
    tr, te = D["jobs"][job]; T, y = D["texts"], D["y"]
    np_ = D["no_punct"]
    m = Scorer(no_punct=np_, rare_heb=bool(np_)).fit([T[i] for i in tr], y[tr].tolist(), vocab=D["vocab"])
    if job == "final":
        out = pickle.dumps(m, protocol=4)
    else:
        b = io.BytesIO(); np.save(b, m.decision([T[i] for i in te]).astype(np.float64)); out = b.getvalue()
    rcp = x25519.Recipient.from_str(recipient)
    return job, encrypt(out, [rcp]), D["fp"], round((time.time() - t0) / 60, 1)


@app.local_entrypoint()
def main():
    bundle = open("lin/bundle.age", "rb").read()
    rcp = [w for w in open("recipient.txt").read().split() if w.startswith("age1")][0]
    names = ["oof0", "oof1", "oof2", "oof3", "oof4", "cvlin", "final"]
    os.makedirs("out", exist_ok=True)
    for job, blob, fp, mins in fit.starmap([(bundle, rcp, j) for j in names]):
        open(f"out/{job}.{'pkl' if job == 'final' else 'npy'}.age", "wb").write(blob)
        print(f"job {job} done in {mins} min", flush=True)
    open("out/fp.txt", "w").write(fp + "\n")
    print("saved", sorted(os.listdir("out")))
