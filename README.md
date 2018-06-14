# Embedding (TCGA RNASeq)
Source code of applying embedding on TCGA RNASeqV2 RSEM normalized data.

## Link
____________
Web Interactive Embedding Projector (powered by [TensorFlow](https://www.tensorflow.org/programmers_guide/embedding>):

Gene Embedding Matrix from:
* [cancer n=9544](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/zeochoy/bcbd669bd78b24e16e7c11a038e6b15d/raw/62f2efdd871226e161bef7dde283a116f38f6d4a/tcga-embedding_cancer_projector_config.json)
* [normal n=701](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/zeochoy/d01656dac8bf70bf3460acd968f17b6c/raw/a1a7128cf6368d93e91924542b1c2c661cb4941e/tcga-embedding_normal_projector_config.json)


## Source Code
Handy python scripts to load data (load_data.py) and functions for handling embeddings (util.py) are included.

### Dependencies
* numpy
* pandas
* matplotlib
* seaborn
* networkx
* scipy
* sklearn

## Folder Structure
```
tcga-embedding
|   README.rst
|   load_data.py
|   util.py
└───emb
    |   gemb_bias_CN.csv
    |   gemb_bias_normal.csv
    |   gemb_CN.csv
    |   gemb_normal.csv
    |   semb_bias_CN.csv
    |   semb_bias_normal.csv
    |   semb_CN.csv
    |   semb_normal.csv
    └───geneSCF
        |   gemb_d17_top_GO_BP.tsv
        |   gemb_d22_top_GO_BP.tsv
        |   gemb_d25_top_GO_BP.tsv
        |   gemb_d35_bottom_GO_BP.tsv
        |   gemb_d43_bottom_GO_BP.tsv
        |   gemb_d46_bottom_GO_BP.tsv
           
└───ipynb
    |   tcga_emb_dist.ipynb
    |   tcga_emb_pca.ipynb
    |   tcga_emb_subtyping.ipynb
    |   tcga_ioresponse.ipynb
    |   tcga_plot_gsea_compare.ipynb
    |   tcga_training_CN.ipynb
    |   tcga_training_normal.ipynb

└───ref
    |   genes_gids.tsv
    |   sid_ca.csv

```
