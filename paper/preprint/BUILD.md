# Building the TokenReceipt preprint

Extract `tokenreceipt-arxiv-source.zip`. Choose pdfLaTeX and `main.tex` for arXiv. The source uses the standard LaTeX `article` class and TeX Live packages. The ZIP contains the prebuilt bibliography and both vector figures.

From the extracted directory, run:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This creates `main.pdf`. To update references, run `bibtex main` after the first pdfLaTeX pass, then run pdfLaTeX twice more. The repository PDF is named `tokenreceipt.pdf`.

Suggested submission fields are in [arxiv-metadata.txt](arxiv-metadata.txt). See [arXiv's TeX submission instructions](https://info.arxiv.org/help/submit_tex.html) and [metadata guide](https://info.arxiv.org/help/prep.html).
