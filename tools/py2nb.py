"""Convert a percent-format .py file (# %% / # %% [markdown]) into a .ipynb notebook.

Usage: python3 tools/py2nb.py src.py out.ipynb [KEY=VALUE ...]
KEY=VALUE pairs replace lines of the form `KEY = ...` (first match) so one source
can produce a quick smoke-test build and a full build.
"""
import re
import sys

import nbformat


def convert(src: str, overrides: dict) -> nbformat.NotebookNode:
    for key, value in overrides.items():
        src, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", src, count=1, flags=re.M)
        if n == 0:
            raise SystemExit(f"override {key} not found in source")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for block in re.split(r"^# %%", src, flags=re.M):
        if not block.strip():
            continue
        header, _, body = block.partition("\n")
        if "[markdown]" in header:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.strip("\n").splitlines()]
            nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
    return nb


if __name__ == "__main__":
    src_path, out_path, *pairs = sys.argv[1:]
    overrides = dict(p.split("=", 1) for p in pairs)
    with open(src_path, encoding="utf-8") as f:
        nb = convert(f.read(), overrides)
    nbformat.write(nb, out_path)
    print(f"wrote {out_path} ({len(nb.cells)} cells)")
