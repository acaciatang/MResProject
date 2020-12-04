#!/bin/bash
# Author: Acacia Tang tst116@ic.ac.uk
# Script: CompileLaTeX.sh
# Desc: compiles a .tex file to a .pdf
# Arguments: 1 -> .tex filem 2 -> Rscript that makes images
# Date: 12 Oct 2019

Rscript $2

if [ -z $1 ] #if there is no input of argument
    then
        echo "Please try again, this script requires an input of a .tex file."
    elif ! [ -f $1 ] #if argument is a file that does not exist
        then
            echo "Please try again, $1 does not exist."
        elif [ "$(basename "$1")" != "$(basename -s .tex $1).tex" ] #if argument does not have the extension .tex
            then
                echo "Please try again, this script only runs on .tex files."
            else #argument is a .tex file that exists, proceed to compile to pdf and open resulting pdf
                pdflatex $1
                pdflatex $1
                bibtex "$(basename $1 .tex)" #adds citations
                pdflatex $1
                pdflatex $1
                evince "$(basename $1 .tex).pdf" & #opens pdf 

                ## Cleanup: removes all auxillary files generated, leaving only the pdf.
                rm *~
                rm *.aux
                rm *.bbl
                rm *.blg
                rm *.dvi
                rm *.fdb_latexmk
                rm *.fls
                rm *.log
                rm *.nav
                rm *.out
                rm *.snm
                rm *.synctex.gz
                rm *.toc
                rm *.blg
                rm *.bbl
                rm *.bcf

                
fi
exit