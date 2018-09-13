import os
import cv2
import pickle
from collections import defaultdict
import numpy as np
from utils import show_images
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA

DIR = 'Affinity Clusters'

def main():
    with open(os.path.join('features', 'images_features_3716_images.txt'), 'rb') as f:
        data = pickle.load(f)
        images = [x + '.jpg' for x in data.keys()]
        X = np.vstack(list(data.values()))

        pca = PCA(n_components=2, whiten=True).fit(X)
        X = pca.transform(X)
        print('PCA transform done')
        # #############################################################################
        # Generate sample data
        # centers = [[1, 1], [-1, -1], [1, -1]]
        # X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
        #                             random_state=0)
        #
        # print(X)

        # #############################################################################
        # Compute Affinity Propagation
        af = AffinityPropagation().fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        print('Done clustering')

        n_clusters_ = len(cluster_centers_indices)
        print('Total clusters', n_clusters_)
        input('Press Enter to Continue')
        #
        # print('Estimated number of clusters: %d' % n_clusters_)
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        # print("Adjusted Rand Index: %0.3f"
        #       % metrics.adjusted_rand_score(labels_true, labels))
        # print("Adjusted Mutual Information: %0.3f"
        #       % metrics.adjusted_mutual_info_score(labels_true, labels))
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

        # #############################################################################
        # Plot result
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
            # path = os.path.join('affinity_cluster_images', str(labels[images.index(img)]))
            # if not os.path.exists(path):
            #     os.mkdir(path)
            # copy(os.path.join('images', img), path)

        print('forming cluster images...')
        for clus in images_clusters.keys():
            print('Cluster', clus)
            print('No of images', len(images_clusters[clus]))
            c_images = [cv2.resize(cv2.imread(os.path.join('data', 'OCTimages' ,c)), (700, 700)) for c in images_clusters[clus]]
            cols = len(c_images) // 2 if len(c_images) // 2 > 0 else 1
            show_images(c_images[:15], DIR=DIR, cluster=clus, cols=cols)
            print('Saved', clus)

        # plt.close('all')
        # fig = plt.figure(1)
        # plt.clf()
        # ax = fig.add_subplot(111)
        #
        # if not os.path.exists('affinity_cluster_images'):
        #     os.mkdir('affinity_cluster_images')

        # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        # for k, col in zip(range(n_clusters_), colors):
        #     class_members = labels == k
        #     cluster_center = X[cluster_centers_indices[k]]
        #     ax.plot(X[class_members, 0], X[class_members, 1], col + '.')
        #     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #              markeredgecolor='k', markersize=14)
        #     for x in X[class_members]:
        #         ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
        #


        # xy = (0.5, 0.7)
        # ax.plot(xy[0], xy[1], ".r")
        #
        # fn = get_sample_data("grace_hopper.png", asfileobj=False)
        # arr_img = plt.imread(fn, format='png')
        #
        # imagebox = OffsetImage(arr_img, zoom=0.2)
        # imagebox.image.axes = ax
        #
        # ab = AnnotationBbox(imagebox, xy,
        #                     xybox=(120., -80.),
        #                     xycoords='data',
        #                     boxcoords="offset points",
        #                     pad=0.5,
        #                     arrowprops=dict(
        #                         arrowstyle="->",
        #                         connectionstyle="angle,angleA=0,angleB=90,rad=3")
        #                     )
        #
        # ax.add_artist(ab)

        # plt.title('Affinity Clustering - Estimated clusters: %d' % n_clusters_)
        # plt.savefig('aff.png')
        # plt.show()

if __name__ == '__main__':
    exit(main())
