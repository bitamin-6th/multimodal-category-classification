from sklearn.cluster import KMeans, DBSCAN
import pandas as pd


def cluster_cluster(data, N, algo, n_clusters, eps, min_samples):
    ndata = pd.read_csv(data)
    ndata = pd.DataFrame(ndata)

    sample_data = ndata.iloc[:, 1:].sample(n=N).reset_index(drop=True)
    nsample_data = sample_data.iloc[:, 1:].to_numpy()

    if algo == 'kmeans':
        eps = False
        min_samples = False
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=1).fit(nsample_data)
        labels = pd.DataFrame(kmeans.labels_)

    elif algo == 'dbscan':
        n_clusters = False
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(nsample_data)
        labels = pd.DataFrame(dbscan.labels_)

    else:
        print('Not yet...')

    result = pd.concat([sample_data.iloc[:, 0], labels],
                       axis=1, ignore_index=True)
    result.rename(columns={0: 'ItemName', 1: 'Labels'}, inplace=True)

    try:
        for i in range(len(result['Labels'].value_counts().keys())):
            print(result.groupby('Labels').get_group(i))

    except:
        print("DBSCAN still doesn't work")


cluster_cluster('image_feature.csv', 36000, 'dbscan', 7, 3, 2)
