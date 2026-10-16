"""DictaBERT-tiny fine-tune (ft_job.py, unchanged logic) on a Modal GPU (owner, 2026-10-16: faster than the CPU runner).
The encrypted inputs (ft/*.age from branch ft-input) go in as bytes and are decrypted only inside the container with
REQ_AGE_KEY (ephemeral secret for this run). Everything that leaves the container is encrypted to recipient.txt.
usage (GitHub Actions): modal run modal_jobs/ft.py --variant cv --epochs 2"""
import io, os, tarfile
import modal

app = modal.App("hebai-ft")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.46.3", "pyrage==1.2.3", "numpy<2")
secret = modal.Secret.from_dict({"REQ_AGE_KEY": os.environ.get("REQ_AGE_KEY", ""), "SAVE_MODEL": "1"})
vol = modal.Volume.from_name("hebai-models", create_if_missing=True)   # models encrypted to the REQ key only (for ft_score.py)


@app.function(gpu="A10G", image=image, timeout=5 * 60 * 60, secrets=[secret], volumes={"/vol": vol})
def run(bundle: bytes, variant: str, epochs: str, model: str = "dicta-il/dictabert-tiny", max_len: str = "128",
        tag: str = "", max_minutes: str = "280") -> bytes:
    import subprocess, tempfile
    from pyrage import encrypt, x25519
    d = tempfile.mkdtemp(); tarfile.open(fileobj=io.BytesIO(bundle)).extractall(d)
    os.makedirs(f"{d}/out_model", exist_ok=True)
    extra = ["--fp16"] if model != "dicta-il/dictabert-tiny" else []
    subprocess.run(["python", "ft_job.py", "--variant", variant, "--epochs", epochs, "--model", model, "--max-len", max_len,
                    "--max-minutes", max_minutes] + extra, cwd=d, check=True)
    if tag and os.path.isdir(f"/tmp/model_{variant}"):   # a copy for remote scoring, encrypted to the REQ key
        mb = io.BytesIO()
        with tarfile.open(fileobj=mb, mode="w:gz") as tf:
            tf.add(f"/tmp/model_{variant}", arcname=f"model_{variant}")
        rcp = x25519.Identity.from_str(os.environ["REQ_AGE_KEY"].strip()).to_public()
        open(f"/vol/ft_model_{variant}_{tag}.tar.gz.age", "wb").write(encrypt(mb.getvalue(), [rcp])); vol.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(f"{d}/out", arcname="out"); tf.add(f"{d}/out_model", arcname="out_model")
    return buf.getvalue()


@app.local_entrypoint()
def main(variant: str = "cv", epochs: str = "2", model: str = "dicta-il/dictabert-tiny", max_len: str = "128", tag: str = "",
         max_minutes: str = "280"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in ("ft_job.py", "recipient.txt", "ft"):
            tf.add(p)
    out = run.remote(buf.getvalue(), variant, epochs, model, max_len, tag, max_minutes)
    tarfile.open(fileobj=io.BytesIO(out)).extractall(".")
    print("saved", sorted(os.listdir("out")), sorted(os.listdir("out_model")))
