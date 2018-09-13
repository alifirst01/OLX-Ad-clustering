import os
import pandas as pd
import warnings
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize_and_stem, tokenize_only, counter_similarity, manhattan_distance

olx_ads = pd.read_csv(open(os.path.join('data', 'Ads-without-Image-2nd-Oct.csv'), encoding='utf8'), engine='python')
ad_ids = list(olx_ads['Ad ID'].unique())[:10000]
warnings.filterwarnings("ignore")
MDS()

def cosine_sim(vectorizer, text1, text2):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        vectorizer = TfidfVectorizer(stop_words=None, use_idf=True, tokenizer=tokenize_and_stem)
        tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]

def euclidean_freq(vectorizer, text1, text2):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        vectorizer = TfidfVectorizer(stop_words=None, use_idf=True, tokenizer=tokenize_and_stem)
        tfidf = vectorizer.fit_transform([text1, text2])
    return float(euclidean_distances(list(tfidf[0].toarray()), list(tfidf[1].toarray()), squared=True))

def manhatten_freq(vectorizer, text1, text2):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except:
        vectorizer = TfidfVectorizer(stop_words=None, use_idf=True, tokenizer=tokenize_and_stem)
        tfidf = vectorizer.fit_transform([text1, text2])
    return float(manhattan_distance(list(tfidf[0].toarray()[0]), list(tfidf[1].toarray()[0])))


def normalize_values(frame):
    scores = frame['Score'].values.tolist()
    m1 = max(scores)
    m2 = min(scores)
    ids = frame['ID'].unique()
    for id in ids:
        data = frame[frame['ID'] == id]
        score = data['Score'].iloc[0]
        frame.loc[frame['ID'] == id, "Score"] = 1 - ((score - m2) / (m1 - m2))
    return frame

def getClusterScore(descs):
    if len(descs) < 2 or len(descs) > 1000:
        return 0.0
    pairwise_scores = []
    for i in range(len(descs)):
        for j in range(len(descs)):
            if j > i:
                pairwise_scores.append(counter_similarity(descs[i], descs[j]))
    return mean(pairwise_scores)

def assign_Scores(descriptions, clusters, cluster_centers_indices):
    ads_clusters = {'ID': ad_ids, 'Description': descriptions, 'Cluster': clusters}
    frame1 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    frame2 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    frame3 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    frame4 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    frame5 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    frame6 = pd.DataFrame(ads_clusters, columns=['ID', 'Description', 'Cluster', 'New Cluster', 'Score'])
    clusters_no = frame1['Cluster'].unique()
    clusters_no = sorted(clusters_no)
    cluster_new_scores = []

    for cluster in clusters_no:
        print('Scoring Cluster...', cluster)
        data = frame1[frame1['Cluster'] == cluster]
        descs = data['Description']
        ids = data['ID']
        center_desc = descriptions[cluster_centers_indices[cluster]]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                           use_idf=True, tokenizer=tokenize_and_stem)
        for id, desc in zip(ids, descs):
            frame1.loc[frame1['ID'] == id, "Score"] = cosine_sim(tfidf_vectorizer, desc, center_desc)
            frame2.loc[frame2['ID'] == id, "Score"] = counter_similarity(desc, center_desc, 0)
            frame3.loc[frame3['ID'] == id, "Score"] = euclidean_freq(tfidf_vectorizer, desc, center_desc)
            frame4.loc[frame4['ID'] == id, "Score"] = counter_similarity(desc, center_desc, 1)
            frame5.loc[frame5['ID'] == id, "Score"] = manhatten_freq(tfidf_vectorizer, desc, center_desc)
            frame6.loc[frame6['ID'] == id, "Score"] = counter_similarity(desc, center_desc, 2)

        cluster_new_scores.append([cluster, getClusterScore(list(descs))])

    cluster_new_scores = sorted(cluster_new_scores,key=lambda l:l[1], reverse=True)
    frame3 = normalize_values(frame3)
    frame4 = normalize_values(frame4)
    frame5 = normalize_values(frame5)
    frame6 = normalize_values(frame6)

    for pair in cluster_new_scores:
        frame1.loc[frame1['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)
        frame2.loc[frame2['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)
        frame3.loc[frame3['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)
        frame4.loc[frame4['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)
        frame5.loc[frame5['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)
        frame6.loc[frame6['Cluster'] == pair[0], 'New Cluster'] = cluster_new_scores.index(pair)

    frame1.drop('Cluster', 1, inplace=True)
    frame1.rename(columns={'New Cluster' : 'Cluster'}, inplace=True)
    frame2.drop('Cluster', 1, inplace=True)
    frame2.rename(columns={'New Cluster': 'Cluster'}, inplace=True)
    frame3.drop('Cluster', 1, inplace=True)
    frame3.rename(columns={'New Cluster': 'Cluster'}, inplace=True)
    frame4.drop('Cluster', 1, inplace=True)
    frame4.rename(columns={'New Cluster': 'Cluster'}, inplace=True)
    frame5.drop('Cluster', 1, inplace=True)
    frame5.rename(columns={'New Cluster': 'Cluster'}, inplace=True)
    frame6.drop('Cluster', 1, inplace=True)
    frame6.rename(columns={'New Cluster': 'Cluster'}, inplace=True)

    return frame1.sort_values(['Cluster', 'Score'], ascending=[1, 0]), \
           frame2.sort_values(['Cluster', 'Score'], ascending=[1, 0]), \
           frame3.sort_values(['Cluster', 'Score'], ascending=[1, 0]), \
           frame4.sort_values(['Cluster', 'Score'], ascending=[1, 0]), \
           frame5.sort_values(['Cluster', 'Score'], ascending=[1, 0]), \
           frame6.sort_values(['Cluster', 'Score'], ascending=[1, 0])


def plot_Clusters(dist, clusters, titles):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: ''}
    # set up cluster names using a dict
    cluster_names = {0: 'Cluster 1',
                     1: 'Cluster 2',
                     2: 'Cluster 3'}

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
    groups = df.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                mec='none'), #color=cluster_colors[name], label=cluster_names[name])
        ax.set_aspect('auto')
        ax.tick_params(axis='x',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       labelbottom='off')
        ax.tick_params(axis='y',  # changes apply to the y-axis
                       which='both',  # both major and minor ticks are affected
                       left='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=5)

    plt.show()

def main():
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    descriptions = []
    titles = []
    wids = []
    print('Total Ads:', len(ad_ids))
    for id in ad_ids:
        ad_data = olx_ads[olx_ads['Ad ID'] == id]
        desc = ad_data['Description'].iloc[0]
        desc = ''.join([i if ord(i) < 128 else '' for i in desc])
        if len(tokenize_and_stem(desc)) < 2:
            wids.append(id)
            continue
        descriptions.append(desc)
        titles.append(ad_data['Title'].iloc[0])

        allwords_stemmed = tokenize_and_stem(desc)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = tokenize_only(desc)
        totalvocab_tokenized.extend(allwords_tokenized)

        if ad_ids.index(id) % 1000 == 0:
            print('Stemmed Descriptions', ad_ids.index(id))

    [ad_ids.remove(i) for i in wids]
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    print(vocab_frame.head())

    print('\ntfidf Vertorizing...')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem)
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    print('Shape', tfidf_matrix.shape)

    #terms = tfidf_vectorizer.get_feature_names()
    #dist = 1 - cosine_similarity(tfidf_matrix)

    print('\nApplying affinity propagation...')
    af = AffinityPropagation().fit(tfidf_matrix)
    print('Done clustering')
    cluster_centers_indices = af.cluster_centers_indices_
    clusters = af.labels_.tolist()

    print('Assigning Scores...')
    frame1, frame2, frame3, frame4, frame5, frame6 = assign_Scores(descriptions, clusters, cluster_centers_indices)

    if not os.path.exists('Results'):
        os.mkdir('Results')

    file_name1 = os.path.join('Results', 'Text_Clusters_tfidf-cosine.csv')
    file_name2 = os.path.join('Results', 'Text_Clusters_counter-cosine.csv')
    file_name3 = os.path.join('Results', 'Text_Clusters_tfidf-eucledean.csv')
    file_name4 = os.path.join('Results', 'Text_Clusters_counter-euclidean.csv')
    file_name5 = os.path.join('Results', 'Text_Clusters_tfidf-manhattan.csv')
    file_name6 = os.path.join('Results', 'Text_Clusters_counter-manhattan.csv')

    frame1.to_csv(file_name1 , index=False, sep=',', encoding='utf-8')
    print(file_name1, 'saved')
    frame2.to_csv(file_name2, index=False, sep=',', encoding='utf-8')
    print(file_name2, 'saved')
    frame3.to_csv(file_name3, index=False, sep=',', encoding='utf-8')
    print(file_name3, 'saved')
    frame4.to_csv(file_name4, index=False, sep=',', encoding='utf-8')
    print(file_name4, 'saved')
    frame5.to_csv(file_name5, index=False, sep=',', encoding='utf-8')
    print(file_name5, 'saved')
    frame6.to_csv(file_name6, index=False, sep=',', encoding='utf-8')
    print(file_name6, 'saved')
    # plot_Clusters(dist, clusters, titles)

if __name__ == '__main__':
    exit(main())
