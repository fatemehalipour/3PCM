# 3PCM

This repository contains all the source files and datasets for 3-Pronged Classification Method for Taxonomic Classification of Astrovirus Sequences. All the datasets used in the paper (raw and processed) can be found  in <a href="https://github.com/fatemehalipour/3PCM/tree/main/data">data</a> folder.
The source code of Prong 1 (supervised method), Prong 2 (unsupervised method), Prong 3 (host identification), and well as a Jupyter Notebook for an example run of the code can be found in 
<a href="https://github.com/fatemehalipour/3PCM/tree/main/src">src</a> folder.

## Datasets
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/dataset1.p">Dataset 1</a>: Dataset consisting of all 992 astrovirus sequences used in this study.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/dataset2.p">Dataset 2</a>: Dataset consisting of astrovirus sequences with available taxonomic labels.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/dataset2_NR.p">Dataset 2 (non-recombinant)</a>: Subset of Dataset 2 in which all sequences are not recombinants.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/dataset3.p">Dataset 3</a>: Dataset consisting of unclassified astrovirus sequences with mammalian or avian hosts.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/dataset3_NR.p">Dataset (non-recombinant)</a>: Subset of Dataset 3 in which all sequences are not recombinants.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/avastrovirus.p">Avastrovirus</a>: All avastrovirus sequences (previously classified or classified by 3PCM), labelled as goose avastrovirus or non-goose avastrovirus. 
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/mamastrovirus.p">Mamastrovirus</a>: All mamastrovirus sequences (previously classified or classified by 3PCM), labelled as human mamastrovirus or non-human mamastrovirus.
* <a href="https://github.com/fatemehalipour/3PCM/blob/main/data/5_hosts.p">5 Hosts</a>: Dataset consisting of unclassified astrovirus sequences belonging to 5 main hosts (Reptilia, Mammalia, Aves, Amphibia, Actinopterygii).

The format of the dataset files is a list of 3-tuples (label, sequence, accession id). The information about the host of astrovirus sequences used in this study can be found in
<a href="https://github.com/fatemehalipour/3PCM/blob/main/data/metadata_host.xlsx">metadata_host.xlsx</a> file.

---

## Required Dependencies
* `python 3.x`
Please use `pip install (package name)` for the following packages:
* `Biopython`
* `Sklearn`
* `Statistics`
* `Pandas`
* `Numpy`
* `Pathlib`
* `Itertools`
