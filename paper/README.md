# Workshop paper — LaTeX source (TikZ-only figures)

Self-contained workshop paper. All figures are drawn natively with `pgfplots`/TikZ from the
reported numbers; there are **no external image files**.

Each `tikzpicture` is additionally externalized to its own PDF in `tikz-figures/`
(`overview`, `decode`, `tfbind`, `meltome`, `gfp`, `feasible`), so individual figures can be
inspected or reused without rebuilding the whole paper.

## Files
- `paper.tex` — the paper (article class; pgfplots, booktabs, natbib, hyperref)
- `refs.bib` — bibliography (BibTeX)
- `tikz-figures/` — one PDF per figure, regenerated on build

## Compile
Externalization runs the TeX engine on each figure, so `-shell-escape` is required:
```bash
latexmk -pdf -pdflatex="pdflatex -shell-escape -interaction=nonstopmode" paper.tex
```
or, by hand:
```bash
pdflatex -shell-escape paper
bibtex   paper
pdflatex -shell-escape paper
pdflatex -shell-escape paper
```

`tikz-figures/` must exist before the first build (`mkdir -p tikz-figures`). Figures are
cached by content hash; delete `tikz-figures/*` to force a full redraw.

Requires a standard TeX Live with `pgfplots` (included in `collection-latexrecommended`).
The document sets `\pgfplotsset{compat=1.17}`.
