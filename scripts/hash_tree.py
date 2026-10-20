#!/usr/bin/env python3
"""Deterministic SHA256 for a vendored source tree.

Sorts POSIX relative paths lexicographically; hashes path bytes, NUL, then each
file's SHA256 digest. Excludes generated __pycache__ directories and .pyc/.pyo.
"""
from pathlib import Path
import argparse, hashlib, json

def hash_tree(root):
    root=Path(root).resolve(); h=hashlib.sha256(); files=[]
    for p in sorted(root.rglob('*'), key=lambda x:x.relative_to(root).as_posix()):
        rel=p.relative_to(root)
        if not p.is_file() or '__pycache__' in rel.parts or p.suffix in ('.pyc','.pyo'): continue
        digest=hashlib.sha256(p.read_bytes()).digest(); h.update(rel.as_posix().encode()); h.update(b'\0'); h.update(digest); files.append(rel.as_posix())
    return {'root':str(root),'file_count':len(files),'sha256':h.hexdigest(),'excluded':['__pycache__/**','*.pyc','*.pyo'],'path_encoding':'UTF-8 POSIX relative path bytes'}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); print(json.dumps(hash_tree(p.parse_args().root),indent=2))
