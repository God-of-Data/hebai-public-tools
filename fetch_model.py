"""Download a public Hugging Face model and export it to ONNX. Nothing else lives in this repository:
no data, no texts, no results. Output: dist/<name>/ (model.onnx + tokenizer files), published as a release.
usage: python fetch_model.py dicta-il/dictabert-tiny
"""
import os
import shutil
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = sys.argv[1]
out = os.path.join("dist", name.split("/")[-1])
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()
tok.save_pretrained(out)
ids = tok("שלום עולם", return_tensors="pt")
torch.onnx.export(model, (ids["input_ids"], ids["attention_mask"]), os.path.join(out, "model.onnx"),
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "b", 1: "n"}, "attention_mask": {0: "b", 1: "n"}, "logits": {0: "b", 1: "n"}},
                  opset_version=17)
with torch.no_grad():
    ref = model(**ids).logits[0, 1, :5].tolist()
open(os.path.join(out, "check.txt"), "w").write(f"{name}\nparams {sum(p.numel() for p in model.parameters())}\nlogits[0,1,:5] {ref}\n")
shutil.make_archive(out, "gztar", out)
print(open(os.path.join(out, "check.txt")).read())
