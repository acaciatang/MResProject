#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

## laod package
require(R.matlab)

## load data
rawdata <- readMat(args[1])
print(head(rawdata))