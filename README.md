# DVCAE: Semi-Supervised Dual Variational Cascade Autoencoders for Information Popularity Prediction

This project contains the source code for the work: **DVCAE: Semi-Supervised Dual Variational Cascade Autoencoders for Information Popularity Prediction**

**Authors**: Jiaxing Shang, Xiaoquan Li, Xueqi Jia, Ruiyuan Li, Fei Hao, Geyong Min

Please kindly give us a star if you find this code helpful.

## Table of Contents

- gene_cas.py: preprocessing procedure to generate cascade graphs and global interaction graph
- gene_emb.py: Genetate node embeddings and perform training, validating, and test splits
- main.py: the main procedure for model training and testing
- VGAE.py: the code for VGAE (Variational Grpah Auto-Encoder) module
- VTAE.py: the code for VTAE (Variational Temporal Auto-Encoder) module
- model.py: the overall model architecture

## Execution

python main.py

## Requirements

- dgl-cu101==0.6.1

- torch==1.4.0

- scikit-learn==1.0.2

- numpy==1.19.2

- networkx==2.5

- absl-py==0.14.1

- scipy==1.5.2

## Contact

+ shangjx@cqu.edu.cn (Jiaxing Shang)

+ jiaxueqi99@126.com (Xueqi Jia)
