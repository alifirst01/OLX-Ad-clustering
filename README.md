# OLX-aD-clustering
Clustering algorithm to detect ad spams (Different versions of same ad)

### Description
Detecting different versions of similar ads posted on olx by clustering together ads from a collection of over 40000 ads. Different versions of same ad are clustered together on the basis of td-idf scores and are assigned a unique cluster id. The ads inside the cluster are then ranked according to their similarity with the centeral ad of cluster.

### Affinity Propagation Clustering
Each ad is converted into a term frequency-document inverse frequency(tf-idf) vector and then put into the affinity clustering algorithm.
After clustering, intra cluster ranking of ads is performed using 6 different metrics including counter consine similarity, tf-idf cosine similarity, counter euclidean distance, tf-idf euclidean distance, counter manhatten distance, tf-idf manhatten distance.
