# ngcf-neural-graph-collaborative-filtering
This repository contains replication package for paper written by Kobiela, D., Groth, J., Sieczczyński, M., Wolniak, R., Pastuszak, K. [[1]](#1) which continues the work performed by Wang et al. [[2]](#2). 
The goal of the study was to verify the robustness of the NGCF (Neural Graph Collaborative Filtering) technique by assessing its ability to generalize across different datasets. 
To achieve this, we first replicated the experiments conducted by Wang et al.to ensure that their replication package is functional. 
We received sligthly better results for ndcg@20 and somewhat poorer results for recall@20, which may be due to the randomness. 
Afterward, we applied their framework to four additional datasets (NYC2014, TOKYO2014, Yelp2022, and MovieLens1M) and compared NGCF with HOP-Rec and MF-BPR as in the original study. 
Our results confirm that NGCF model outperforms other models in terms of ndcg@20. 
However, when considering recall@20, either HOP-Rec or MF-BPR performed better on the new datasets. 
This finding suggests that NGCF may have been optimized for the datasets used in the original paper. 
Furthermore, we analyzed the models’ performance using recall@K and ndcg@K, where K was set to 1, 5, 10, and 40. 
The obtained results support our previous findings.

## References
<a id="1">[1]</a> 
Kobiela Dariusz, Groth Jan, Sieczczyński Michał, Wolnial Rafał, Pastuszak Krzysztof (2023). 
Neural Graph Collaborative Filtering: Analysis of Possibilities on Diverse Datasets. 
In: Abelló, A., et al. New Trends in Database and Information Systems. 
ADBIS 2023. Communications in Computer and Information Science, vol 1850. Springer, Cham. 
https://doi.org/10.1007/978-3-031-42941-5_54

<a id="2">[2]</a> 
Wang, X., He, X., Wang, M., Feng, F., Chua, T.S.: Neural graph collaborative filtering, pp. 165–174. 
SIGIR’19, Association for Computing Machinery, New York, NY, USA (2019). 
https://doi.org/10.1145/3331184.3331267