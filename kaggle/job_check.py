# Kaggle check job: GPU present, secret REQ_AGE_KEY readable, encrypted inputs decryptable. Prints counts only; writes an
# encrypted status file (to recipient.txt) into /kaggle/working.
import glob, json, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyrage"], check=False)
st = {}
try:
    import torch; st["gpu"] = torch.cuda.is_available(); st["gpu_name"] = torch.cuda.get_device_name(0) if st["gpu"] else None
    st["gpu_count"] = torch.cuda.device_count()
except Exception as e:
    st["gpu"] = f"torch error {type(e).__name__}"
files = sorted(glob.glob("/kaggle/input/**/*.age", recursive=True)); st["input_files"] = len(files)
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("REQ_AGE_KEY").strip(); st["secret"] = "ok"
except Exception as e:
    key = None; st["secret"] = f"missing ({type(e).__name__})"
if key and files:
    from pyrage import decrypt, x25519
    try:
        b = decrypt(open(files[0], "rb").read(), [x25519.Identity.from_str(key)]); st["decrypt"] = f"ok ({len(b)} bytes)"
    except Exception as e:
        st["decrypt"] = f"failed ({type(e).__name__})"
print("STATUS", json.dumps(st))
rcp = [l for l in open(glob.glob("/kaggle/input/**/recipient.txt", recursive=True)[0]).read().split() if l.startswith("age1")][0] \
    if glob.glob("/kaggle/input/**/recipient.txt", recursive=True) else None
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(st, open("/kaggle/working/status.json", "w"))   # non-secret status only
