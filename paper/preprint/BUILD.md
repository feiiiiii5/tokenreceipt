# Building the TokenReceipt preprint

The signed preprint was revised on November 19, 2026. Extract [`tokenreceipt-arxiv-source.zip`](tokenreceipt-arxiv-source.zip) or work in [`source/`](source/). The source uses the standard LaTeX `article` class and includes the compiled bibliography and both vector figures.

Build locally with Tectonic. The public PDF was built with Tectonic 0.17.0.

```sh
tectonic main.tex
```

For arXiv, select `main.tex` and pdfLaTeX. With TeX Live installed, use the bundled bibliography and run from the extracted directory.

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

To update the bibliography with TeX Live, run `bibtex main` after the first pdfLaTeX pass and then run pdfLaTeX twice more. Both build routes produce `main.pdf`. The repository copy is named [`tokenreceipt.pdf`](tokenreceipt.pdf).

Suggested submission fields are in [arxiv-metadata.txt](arxiv-metadata.txt). See [arXiv's TeX submission instructions](https://info.arxiv.org/help/submit_tex.html) and [metadata guide](https://info.arxiv.org/help/prep.html).
