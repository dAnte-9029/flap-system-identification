#!/usr/bin/env python3
"""Backport PyTorch PR165066 in the explicitly isolated v4 clone, never production."""
import difflib,sys
from pathlib import Path
import torch
expected=Path('/home/zn/anaconda3/envs/pahbd-compile-v4')
if Path(sys.prefix)!=expected:raise RuntimeError('requires the isolated pahbd-compile-v4 environment')
base=Path(torch.__file__).parent
p=base/'_inductor/codegen/triton.py';old=p.read_text();s=old
helper='''def is_sympy_integer_like(expr):
    # Backport https://github.com/pytorch/pytorch/pull/165066
    return isinstance(expr, sympy.Expr) and (
        isinstance(expr, sympy.Integer) or (expr.is_integer and not expr.free_symbols)
    )


'''
if 'def is_sympy_integer_like(' not in s:
    s=s.replace('class OpDtypeSupport:',helper+'class OpDtypeSupport:',1)
    assert 'if isinstance(index, sympy.Integer):' in s
    assert 'elif isinstance(original_index, sympy.Integer):' in s
    s=s.replace('if isinstance(index, sympy.Integer):','if is_sympy_integer_like(index):')
    s=s.replace('elif isinstance(original_index, sympy.Integer):','elif is_sympy_integer_like(original_index):')
    # Atomic replacement also breaks any hardlink to the original environment.
    tmp=p.with_suffix('.pahbd_tmp');tmp.write_text(s);tmp.replace(p)
    print(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(p))))
p=base/'utils/_sympy/functions.py';old=p.read_text();s=old
start=s.index('class Identity(');end=s.index('\nclass ',start+1) if '\nclass ' in s[start+1:] else len(s)
section=s[start:end]
if 'def _sympystr(' not in section:
    section=section.replace('    def _eval_is_real(self):','    def _sympystr(self, printer):\n        return f"({printer.doprint(self.args[0])})"\n\n    def _eval_is_real(self):',1)
    s=s[:start]+section+s[end:];tmp=p.with_suffix('.pahbd_tmp');tmp.write_text(s);tmp.replace(p)
    print(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(p))))
