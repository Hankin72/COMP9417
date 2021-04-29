from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA, IncrementalPCA


def Voting_RF_ETC_KNN(X, y, X_val, y_val, X_test):
    pca = PCA(n_components=35, random_state=42)
    X_train = pca.fit_transform(X)
    X_val = pca.transform(X_val)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_val = min_max_scaler.transform(X_val)

    y_train = np.array(y).ravel()
    y_val = np.array(y_val).ravel()

    clf_rf = RandomForestClassifier(n_estimators=160, criterion='gini', min_samples_leaf=1, max_features='auto',
                                    min_samples_split=2, random_state=0)
    clf_etc = ExtraTreesClassifier(criterion='gini', max_features='auto', random_state=0)

    clf_knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', leaf_size=5, algorithm='auto', p=1)

    clf_rf.fit(X_train, y_train)
    clf_etc.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)

    eclf1 = VotingClassifier(estimators=[('rf', clf_rf), ('etc', clf_etc), ('knn', clf_knn)], voting='hard')
    eclf1.fit(X_train, y_train)

    y_pre_voting = eclf1.predict(X_val)

    f1_hard_voting = f1_score(y_val, y_pre_voting, average='weighted')

    print("Hard Voting for f1 (weighted) socre:  ", f1_hard_voting)

    X_test = pca.transform(X_test)
    X_test = min_max_scaler.transform(X_test)

    y_test_voting = eclf1.predict(X_test)

    return y_pre_voting, _, y_test_voting, _

