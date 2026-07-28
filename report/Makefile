name=main

all:
	pdflatex $(name).tex
	bibtex   $(name)
	pdflatex $(name).tex
	pdflatex $(name).tex
	@rm -f *.log *.blg *.aux *.toc *.bbl *.out *.lof *.lot

clean:
	@rm -f *.log *.blg *.aux *.toc *.bbl *.out *.lof *.lot $(name).pdf

.PHONY: all clean
