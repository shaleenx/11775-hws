import numpy
import os
#from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle
import sys
import pandas as pd
import pickle
import yaml

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} surf_csv_file cluster_num output_file".format(sys.argv[0]))
        print("surf_csv_file -- path to the surf csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    surf_csv_file = sys.argv[1]

    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    # Get parameters from config file

    cluster_num = my_params.get('kmeans_clusters')
    output_file = "kmeans." + str(cluster_num) + ".model"

    print("Using", surf_csv_file, "and storing kmeans with", cluster_num, "cluster in", output_file)

    df = pd.read_csv(surf_csv_file, delimiter=",", header=None)
    
    estimator = MiniBatchKMeans(init='k-means++', verbose=True, n_clusters = cluster_num)

    estimator.fit(df)
    pickle.dump(estimator, open(output_file, "wb"))
    print("Model saved in " + output_file)

    print("K-means trained successfully!")
