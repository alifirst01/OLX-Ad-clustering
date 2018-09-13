import os
import cv2
import pickle
import random
import numpy as np
from shutil import copy
from utils import show_images
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth


with open('features\\images_features_3716_images.txt', 'rb') as f:
    data = pickle.load(f)
    images = [x + '.jpg' for x in data.keys()]
    print(list(data.values())[0])
    X = np.vstack(list(data.values()))

    pca = PCA(n_components=2, whiten=True).fit(X)
    X = pca.transform(X)
    print('PCA transform done')

    # #############################################################################
    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    nlist = [100]
    quantile = 0.025
    for n_samples in nlist:
        print(quantile, n_samples)
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print('Done Clustering')

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        if n_clusters_ > 3:
            print("number of estimated clusters : %d" % n_clusters_)

            DIR = os.path.join('Mean Shift Clusters', str(quantile) + '_' + str(n_samples))
            if not os.path.exists(os.path.join('Results', DIR)):
                os.mkdir(os.path.join('Results', DIR))
            else:
                [os.unlink(os.path.join('Results', DIR, cf)) for cf in os.listdir(os.path.join('Results', DIR))]

            print('Making images clusters...')
            images_clusters = defaultdict()
            for img in images:
                if labels[images.index(img)] not in images_clusters.keys():
                    images_clusters[labels[images.index(img)]] = []
                images_clusters[labels[images.index(img)]].append(img)
                path = os.path.join('Results', DIR, str(labels[images.index(img)]))
                if not os.path.exists(path):
                    os.mkdir(path)
                copy(os.path.join('data\\OCTimages', img), path)
            print()

    # #############################################################################
    #Clusters images
    if not os.path.exists(os.path.join('Results', 'Mean Shift Clusters Images')):
        os.mkdir(os.path.join('Results', 'Mean Shift Clusters Images'))
    DIR = 'Mean Shift Clusters Images'
    print('forming cluster images...')
    for clus in images_clusters.keys():
        print('Cluster', clus)
        print('No of images', len(images_clusters[clus]))
        c_images = [cv2.resize(cv2.imread(os.path.join('data', 'OCTimages', c)), (700, 700)) for c in
                    images_clusters[clus]]

        if len(c_images) > 30:
            ran = random.sample(range(0, len(c_images)), 30)
            c_images = [c_images[i] for i in ran]
        cols = len(c_images) // 2 if len(c_images) // 2 > 0 else 1
        show_images(c_images, DIR=DIR, cluster=clus, cols=cols)
        print('Saved', clus)


    # if not os.path.exists('meanShift_cluster_images'):
    #     os.mkdir('meanShift_cluster_images')
    # for img in images:
    #     path = os.path.join('meanShift_cluster_images', str(labels[images.index(img)]))
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #     file.write(os.path.join('images', img) + ' --> ' + path)
    # file.close()

    # #############################################################################
    # Plot result
    # import matplotlib.pyplot as plt
    # from itertools import cycle
    #
    # plt.figure(1)
    # plt.clf()
    #
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    # plt.title('Mean Shift Clustering - Estimated clusters: %d' % n_clusters_)
    # plt.savefig('mean_shift.png')
    # plt.show()