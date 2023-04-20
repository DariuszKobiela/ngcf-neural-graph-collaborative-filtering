# ngcf-neural-graph-collaborative-filtering
This repository contains replication package for paper which continues the work by Wang et al. [14]. 
Its goal wass to verify the robustness of the NGCF (Neural Graph Collaborative Filtering) technique by 
assessing its ability to generalize across different datasets. To achieve this, we first replicated the experiments conducted by Wang et al. [14] 
to ensure that their replication package is functional. We received sligthly better results for ndcg@20 and somewhat poorer results for recall@20, 
which may be due to the randomness. Afterward, we applied their framework to four additional datasets (NYC2014, TOKYO2014, Yelp2022, and MovieLens1M) 
and compared NGCF with HOP-Rec [15] and MF-BPR [13] as in the original study.
Our results confirm that NGCF outperforms other models in terms of ndcg@20. However, when considering recall@20, 
either HOP-Rec or MF-BPR performed better on the new datasets. This finding suggests that NGCF may have been optimized for the datasets used 
in the original paper. Furthermore, we analyzed the models’ performance using recall@K and ndcg@K, where K was set to 1, 5, 10, and 40. 
The obtained results support our previous findings.