"""Linear-model fits in parallel on Modal CPUs (owner, 2026-10-16). Input: lin/bundle.age from branch lin-input — an
age-encrypted tar with the model code and the training texts — decrypted only inside the containers with REQ_AGE_KEY
(ephemeral secret for this run). Each container runs one fit (out-of-fold fold, cv split or final model); every result
is encrypted to recipient.txt before it leaves the container. usage (GitHub Actions): modal run modal_jobs/linear.py"""
import io, os, tarfile
import modal

app = modal.App("hebai-linear")
image = modal.Image.debian_slim(python_version="3.13").pip_install("scikit-learn==1.9.1", "scipy==1.18.1", "numpy==2.5.3",   # = local versions (the final model is unpickled locally)
                                                                    "pyrage==1.4.0", "cloudpickle==3.1.2")
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", "")})
vol = modal.Volume.from_name("hebai-lin", create_if_missing=True)   # the encrypted bundle, uploaded ONCE per run (owner: ~10 min)


@app.function(image=image, cpu=2.0, memory=24576, timeout=45 * 60, secrets=[secret], volumes={"/vol": vol})
def fit(bundle: bytes, recipient: str, job: str, env: dict | None = None, bpath: str = "") -> tuple:
    import pickle, sys, tempfile, time
    import numpy as np
    from pyrage import decrypt, encrypt, x25519
    t0 = time.time()
    if bpath:   # grid mode (Fable #67 L2): one bundle in the volume, the config (HEBAI_CHAR / HEBAI_LIN_C) per call
        bundle = open(f"/vol/{bpath}", "rb").read()
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    d = tempfile.mkdtemp()
    tarfile.open(fileobj=io.BytesIO(decrypt(bundle, [x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip())]))).extractall(d)
    sys.path.insert(0, d)
    from hebai.scorer import Scorer
    D = pickle.load(open(f"{d}/data.pkl", "rb"))
    tr, te = D["jobs"][job]; T, y = D["texts"], D["y"]
    np_ = D["no_punct"]
    m = Scorer(no_punct=np_, rare_heb=bool(np_)).fit([T[i] for i in tr], y[tr].tolist(), vocab=D["vocab"])
    if job == "final":
        import cloudpickle; out = cloudpickle.dumps(m, protocol=4)   # the vectorizer holds a lambda
    else:
        b = io.BytesIO(); np.save(b, m.decision([T[i] for i in te]).astype(np.float64)); out = b.getvalue()
    rcp = x25519.Recipient.from_str(recipient)
    return job, encrypt(out, [rcp]), D["fp"], round((time.time() - t0) / 60, 1)


@app.local_entrypoint()
def main():
    import json, time, uuid
    bundle = open("lin/bundle.age", "rb").read()
    rcp = [w for w in open("recipient.txt").read().split() if w.startswith("age1")][0]
    names = [l.strip() for l in open("lin/jobs.txt") if l.strip()] if os.path.exists("lin/jobs.txt") else \
        ["oof0", "oof1", "oof2", "oof3", "oof4", "cvlin", "final"]
    os.makedirs("out", exist_ok=True); t0 = time.time()
    if os.path.exists("lin/configs.json"):   # grid: jobs "cfg|job"; one upload, all fits of all configs in parallel
        cfgs = json.load(open("lin/configs.json")); bpath = f"bundle_{uuid.uuid4().hex}.age"
        with vol.batch_upload() as b:
            b.put_file("lin/bundle.age", f"/{bpath}")
        print(f"bundle uploaded in {time.time() - t0:.0f}s, {len(names)} fits", flush=True)
        args = [(b"", rcp, n.split("|")[1], cfgs[n.split("|")[0]], bpath) for n in names]
        for (job, blob, fp, mins), n in zip(fit.starmap(args, order_outputs=True), names):
            c = n.split("|")[0]
            open(f"out/{c}__{job}.{'pkl' if job == 'final' else 'npy'}.age", "wb").write(blob)
            print(f"job {c}|{job} done in {mins} min (t+{(time.time() - t0) / 60:.1f})", flush=True)
        vol.remove_file(f"/{bpath}")
    else:
        for job, blob, fp, mins in fit.starmap([(bundle, rcp, j) for j in names]):
            open(f"out/{job}.{'pkl' if job == 'final' else 'npy'}.age", "wb").write(blob)
            print(f"job {job} done in {mins} min (t+{(time.time() - t0) / 60:.1f})", flush=True)
    open("out/fp.txt", "w").write(fp + "\n")
    print("saved", sorted(os.listdir("out")), f"total {(time.time() - t0) / 60:.1f} min")
