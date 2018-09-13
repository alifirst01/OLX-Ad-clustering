import multiprocessing
from collections import defaultdict
import pandas as pd
import _pickle as pickle
from utils import counter_similarity
import os

max_sim = 10
olx_ads = pd.read_csv(open(os.path.join('data', 'Ads-without-Image-2nd-Oct.csv'), encoding='utf8'), engine='python')
ad_ids = olx_ads['Ad ID'].unique()[:20]
ads_similarity = defaultdict()
lock = multiprocessing.Lock()

#5308175520076694 03 18

def worker(worker_ids, id):
    print('Starting Thread', id, 'Total files', len(worker_ids))
    #x = chardet.detect(open('Ads-without-Image-2nd-Oct.csv', 'rb').read())
    for id1 in worker_ids:
        ad1_data = olx_ads[olx_ads['Ad ID'] == id1]
        desc1 = ad1_data['Description'].iloc[0].replace('\n', ' ')
        list = []
        for id2 in ad_ids:
            if id1 != id2:
                ad2_data = olx_ads[olx_ads['Ad ID'] == id2]
                desc2 = ad2_data['Description'].iloc[0].replace('\n', ' ')
                sim = counter_similarity(desc1, desc2)   # Closer to 1 means more similar
                #sim = edit_distance_similarity(desc1, desc2)
                list.append([desc2, sim])

        list = sorted(list, key=lambda l: l[1], reverse=True)
        lock.acquire()
        ads_similarity[desc1] = [li[0] for li in list[:max_sim]]
        print('Thread %s Processed Ad %d' % (id, worker_ids.index(id1)))
        lock.release()

    with open('thread' + str(id) + '.pickle', 'wb') as f:
        pickle.dump(ads_similarity, f)
    print('---------------------------------------------')
    print('Thread', id, 'done')
    print('---------------------------------------------')

def start_threads(No_of_Threads):
    threads = []
    size = len(ad_ids)
    shift = int(size / No_of_Threads)
    start = 0
    i = 0
    while 1:
        end = start + shift
        if end > size:
            end = size
        try:
            t = multiprocessing.Process(target=worker, args=(ad_ids[start:end], i))
            t.daemon = True
            t.start()
        except Exception as e:
            print("Error Initializing Thread")
            print(e)
            end = start - shift
            continue
        if end == size:
            break
        threads.append(t)
        start = end
        i += 1
    return threads


def main():
    DIR = 'pickle files'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    No_of_Threads = 20
    worker(ad_ids, 1)
    threads = start_threads(No_of_Threads)
    for t in threads:
        t.join()

def merge_pickles():
    DIR = 'pickles'
    files = os.listdir(DIR)
    big_dic = {}
    for f in files:
        with open(os.path.join(DIR, f), 'rb') as pf:
            diction = pickle.load(pf)
            big_dic.update(diction)

    with open('ads.pickle', 'wb') as f:
        pickle.dump(big_dic, f)

    for key in big_dic.keys():
        print('Key:', key)
        for desc in big_dic[key]:
            print('--->', desc)
        input('Press Enter to Continue')


if __name__ == '__main__':
    exit(main())



