#!/usr/bin/env python3
"""Vérifie le style « linéaire » des notebooks (CONVENTIONS.md § 3 bis).

Usage : python3 verif_style.py J01 [J02 ...]   (par défaut : tous les jours)
Analyse les cellules de code du notebook instructeur ET du notebook étudiant (après construction) et signale les
constructions interdites. Code de sortie 1 s'il y a au moins un signalement.
"""
import ast
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

APPELS_INTERDITS = {"display", "zip", "enumerate", "map", "filter", "twinx", "subplots"}
ATTR_INTERDITS = {"r_", "c_", "rcParams", "style", "iterrows", "itertuples", "apply", "applymap", "pipe"}


def analyser(src):
    """Retourne la liste des problèmes (ligne, message) d'une cellule de code."""
    pb = []
    lignes = src.splitlines()
    for i, l in enumerate(lignes, 1):
        code = l.split("#", 1)[0]
        if ";" in code and not re.search(r"['\"].*;.*['\"]", code):
            pb.append((i, "plusieurs instructions sur une ligne (;)"))
        if re.match(r"\s*#\s*-{4,}", l):
            pb.append((i, "ligne de séparation décorative"))
        if re.search(r"\S\s{2,}=\s", code) and not code.strip().startswith("#") and "==" not in code:
            pb.append((i, "alignement décoratif des ="))
    # rendre analysable le squelette étudiant : « x = # À COMPLÉTER ... » -> « x = None »
    src_lignes = []
    for l in src.splitlines():
        if l.lstrip().startswith("%"):
            continue
        if "À COMPLÉTER" in l:
            code = l.split("#", 1)[0].rstrip()
            if code.strip() == "":
                src_lignes.append(l)
            elif code.endswith("=") or code.endswith("return"):
                src_lignes.append(code + " None")
            elif code.endswith("("):
                src_lignes.append(code + "None)")
            else:
                src_lignes.append(code)
        else:
            src_lignes.append(l)
    src_py = "\n".join(src_lignes)
    try:
        tree = ast.parse(src_py)
    except SyntaxError as e:
        return pb + [(e.lineno or 0, f"syntaxe non analysable ({e.msg})")]
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 0)
        if isinstance(node, ast.Lambda):
            pb.append((ln, "lambda"))
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            pb.append((ln, "compréhension / générateur"))
        elif isinstance(node, ast.ClassDef):
            pb.append((ln, "classe"))
        elif isinstance(node, ast.FunctionDef):
            if node.args.vararg or node.args.kwarg:
                pb.append((ln, "*args / **kwargs"))
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub is not node:
                    pb.append((sub.lineno, f"fonction imbriquée {sub.name} dans {node.name}"))
            n_lignes = (node.end_lineno or ln) - ln + 1
            if n_lignes > 40:
                pb.append((ln, f"fonction {node.name} trop longue ({n_lignes} lignes)"))
        elif isinstance(node, ast.Call):
            f = node.func
            nom = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if nom in APPELS_INTERDITS:
                pb.append((ln, f"appel interdit : {nom}()"))
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Attribute) \
                    and isinstance(f.value.func.value, ast.Call):
                pb.append((ln, "chaînage de méthodes sur plus de deux appels"))
        elif isinstance(node, ast.Attribute) and node.attr in ATTR_INTERDITS:
            pb.append((ln, f"attribut interdit : .{node.attr}"))
        elif isinstance(node, ast.If) and node.body and node.body[0].lineno == node.lineno:
            pb.append((ln, "if ... : action sur une seule ligne"))
        elif isinstance(node, (ast.For, ast.While)) and node.body and node.body[0].lineno == node.lineno:
            pb.append((ln, "boucle avec corps sur la même ligne"))
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) > 3:
            pb.append((ln, "déballage de plus de trois variables"))
    return sorted(set(pb))


def verifier(jour):
    total = 0
    for suffix in ("instructeur", "etudiant"):
        path = ROOT / jour / f"{jour}_exercices_{suffix}.ipynb"
        if not path.exists():
            print(f"{jour} : {path.name} absent")
            continue
        nb = nbformat.read(path, as_version=4)
        k = 0
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            k += 1
            for ln, msg in analyser(c.source):
                print(f"{jour} {suffix} cellule {k} ligne {ln} : {msg}")
                total += 1
    print(f"{jour} : {total} signalement(s)")
    return total


if __name__ == "__main__":
    jours = sys.argv[1:] or [f"J{i:02d}" for i in range(1, 11)]
    n = sum(verifier(j) for j in jours)
    sys.exit(1 if n else 0)
