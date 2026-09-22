"""
nbbuild.py — construction des notebooks « étudiant » et « instructeur » à partir
d'une même source Python (une par jour : J01.py … J10.py).

Principe : chaque jour est décrit par une suite de cellules ; les exercices
comportent un énoncé (markdown), un squelette (code à compléter, version
étudiant) et une solution (code complet, version instructeur), plus un
commentaire pédagogique optionnel (instructeur seulement).

Usage :
    python nbbuild.py J01            # construit et exécute J01
    python nbbuild.py all            # tous les jours
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat import v4 as nbf

ROOT = Path(__file__).resolve().parents[1]      # dossier notebooks/


class Notebook:
    def __init__(self, jour: str, titre: str, sous_titre: str = ""):
        self.jour = jour            # ex. "J01"
        self.titre = titre
        self.sous_titre = sous_titre
        self.cells_e: list = []     # étudiant
        self.cells_i: list = []     # instructeur
        self._n_ex = 0
        entete = (f"# {jour} — {titre}\n\n"
                  f"**Physique et hydrodynamique des sols** — S. J. Gumiere\n\n"
                  + (f"*{sous_titre}*\n\n" if sous_titre else ""))
        self.cells_e.append(nbf.new_markdown_cell(entete + "> Version **étudiant** : complétez les cellules marquées `# À COMPLÉTER`."))
        self.cells_i.append(nbf.new_markdown_cell(entete + "> Version **instructeur** : code complet et résultats."))

    # -- cellules communes -------------------------------------------------
    def md(self, text: str, who: str = "both"):
        c = nbf.new_markdown_cell(text.strip("\n"))
        if who in ("both", "etudiant"):
            self.cells_e.append(c)
        if who in ("both", "instructeur"):
            self.cells_i.append(nbf.new_markdown_cell(text.strip("\n")))

    def code(self, src: str, who: str = "both"):
        src = src.strip("\n")
        if who in ("both", "etudiant"):
            self.cells_e.append(nbf.new_code_cell(src))
        if who in ("both", "instructeur"):
            self.cells_i.append(nbf.new_code_cell(src))

    # -- exercice ---------------------------------------------------------
    def exercice(self, titre: str, enonce: str, solution: str, squelette: str | None = None,
                 commentaire: str = "", duree: str = ""):
        self._n_ex += 1
        head = f"## Exercice {self._n_ex} — {titre}" + (f"  *(≈ {duree})*" if duree else "")
        self.md(head + "\n\n" + enonce.strip("\n"))
        if squelette is None:
            squelette = "# À COMPLÉTER\n"
        self.cells_e.append(nbf.new_code_cell(squelette.strip("\n")))
        self.cells_i.append(nbf.new_code_cell(solution.strip("\n")))
        if commentaire:
            self.cells_i.append(nbf.new_markdown_cell("**Commentaire (instructeur).** " + commentaire.strip("\n")))

    # -- écriture ---------------------------------------------------------
    def build(self, execute: bool = True, timeout: int = 900):
        outdir = ROOT / self.jour
        outdir.mkdir(parents=True, exist_ok=True)
        meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"}}
        for suffix, cells in (("etudiant", self.cells_e), ("instructeur", self.cells_i)):
            nb = nbf.new_notebook(cells=cells, metadata=meta)
            path = outdir / f"{self.jour}_exercices_{suffix}.ipynb"
            nbformat.write(nb, path)
            print("écrit :", path.relative_to(ROOT))
        if execute:
            path = outdir / f"{self.jour}_exercices_instructeur.ipynb"
            cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                   "--inplace", f"--ExecutePreprocessor.timeout={timeout}", str(path)]
            print("exécution :", path.name)
            r = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-3000:])
                raise SystemExit(f"Échec de l'exécution de {path.name}")
            # vérification : aucune cellule en erreur
            nb = nbformat.read(path, as_version=4)
            errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
            if errs:
                raise SystemExit(f"{len(errs)} erreur(s) dans {path.name}")
            print(f"OK : {path.name} exécuté sans erreur ({sum(c.cell_type=='code' for c in nb.cells)} cellules de code)")


def build_day(jour: str, execute: bool = True):
    src = Path(__file__).resolve().parent / f"{jour}.py"
    spec = importlib.util.spec_from_file_location(jour, src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb: Notebook = mod.build()
    nb.build(execute=execute)


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    execute = "--no-exec" not in args
    args = [a for a in args if a != "--no-exec"]
    jours = [f"J{i:02d}" for i in range(1, 11)] if args == ["all"] else args
    for j in jours:
        if (Path(__file__).resolve().parent / f"{j}.py").exists():
            build_day(j, execute)
