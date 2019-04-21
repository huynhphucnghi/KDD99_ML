

# CONSTANT
categories = ['duration',
              'protocol_type',
              'service',
              'flag',
              'src_bytes',
              'dst_bytes',
              'land',
              'wrong_fragment',
              'urgent',
              'hot',
              'num_failed_logins',
              'logged_in',
              'num_compromised',
              'root_shell',
              'su_attempted',
              'num_root',
              'num_file_creations',
              'num_shells',
              'num_access_files',
              'num_outbound_cmds',
              'is_host_login',
              'is_guest_login',
              'count',
              'srv_count',
              'serror_rate',
              'srv_serror_rate',
              'rerror_rate',
              'srv_rerror_rate',
              'same_srv_rate',
              'diff_srv_rate',
              'srv_diff_host_rate',
              'dst_host_count',
              'dst_host_srv_count',
              'dst_host_same_srv_rate',
              'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate',
              'dst_host_serror_rate',
              'dst_host_srv_serror_rate',
              'dst_host_rerror_rate',
              'dst_host_srv_rerror_rate',
              'label']


def KNNmodel():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    return knn


def NBmodel():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    return gnb


def crossvalidation(model, dataset, target):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, dataset, target, cv=5)
    print(scores)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
