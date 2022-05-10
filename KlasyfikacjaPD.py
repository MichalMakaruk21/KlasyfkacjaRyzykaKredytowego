import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import os
import statistics as s

#Pobieram dane z csv i ładuje je do dataframe
data = pd.read_csv('bank-full.csv', sep=";")
#Zmieniam wyrażenia "yes" i "no" w parametrze "y" na warotści binarne
data['y'] = (data['y'] == "yes").astype(int)#zmienna do predykcji


#Konwersja df na dummmies, przygowowanie do wczytania do modelu.
#Usuwam zbędne kolumny.
dfDummies = pd.get_dummies(data, drop_first=True)#drop_first = False
#dfDummies.to_csv('dummies.csv', sep=';')

def SignificanceSelector():
    X = np.array((dfDummies.drop(['y'], axis=1)))  # usunięcię z dfDummies kolumny z parametrem y, i przyisananie do zmniennej "X"
    y = np.array((dfDummies['y']))

    # używam mutual_info_classif aby znaleźć istotność poszczególnych cech
    signifance = mutual_info_classif(X, y)

    # wypisuje ważność cech (tylko informacyjnie)
    waznosc_cech = pd.Series(signifance, dfDummies.columns.drop(['y'])).sort_values(ascending=False)
    print('Istotność cech:')
    print(waznosc_cech)

    #Filtruje i wybieram z modelu tylko cechy które przekraczają wybrany poziom istiności

    """
    Próbowałem przeprawdać kalsyfikację z pominięciem cech o instotności poniżej [0.001, 0.005, 0.01, 0.015, 0.2]
    Accuracy modelu zmieniało się nieznacznie(koło 2 - 3 procent), przez co nie wiem czy cięcie cech modelu miało duży sens. 
    """

    filteredIndexes = np.where(signifance > 0.005)[0]#0.001

    X_Filtered = X[:, filteredIndexes]

    #Podział danych na model testowy i treningowy
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_Filtered, y, test_size=0.25)

    return x_train, x_test, y_train, y_test

def SaveBestLogisticRegressionModel():
    # pętla wybierająca i zapsijąca z z 10 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):
        #Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        #Wywołuje obiekt klasyfikatora regresjii logistycznej i trenuje dane wg tej metody
        log_reg = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=10000)
        log_reg.fit(x_train, y_train)
        acc = log_reg.score(x_test, y_test)

        print(acc)
        accList.append(acc)

        #Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = log_reg.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        # zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:

            best_acc_score = acc
            """
            pickleFileName = f"LogisticRegression_Model_{round(best_acc_score, 2)}p.pickle"

            #round nie działa
            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(log_reg, f)
"""
    print(f'Najwyższe acc modelu: {best_acc_score}')

def SaveBestGaussianNB_Model():
    # pętla wybierająca i zapsijąca z z 10 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):

        # Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        # Wywołuje obiekt klasyfikatora naiwnego Bayesa i trenuje dane wg tej metody
        nbayes_model = GaussianNB()
        nbayes_model.fit(x_train, y_train)
        acc = nbayes_model.score(x_test, y_test)

        print(acc)
        accList.append(acc)

        # Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = nbayes_model.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        # zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:

            best_acc_score = acc
            pickleFileName = f"NaiveBayes_Model_{round(best_acc_score, 2)}p.pickle"

            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(nbayes_model, f)
        
        
    print(f'Najwyższe acc modelu: {best_acc_score}')

def SaveBestLDA_Model():
    # pętla wybierająca i zapsijąca z z 10 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):

        # Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        # Wywołuje obiekt klasyfikatora liniowej analizy dyskyminacyjnej i trenuje dane wg tej metody
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        acc = lda.score(x_test, y_test)

        print(acc)
        accList.append(acc)

        # Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = lda.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        # zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:
            best_acc_score = acc
            pickleFileName = f"LDA_Model_{round(best_acc_score, 2)}p.pickle"

            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(lda, f)

    print(f'Najwyższe acc modelu: {best_acc_score}')

def SaveBestKNN_Model():
    # pętla wybierająca i zapsijąca z z 10 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):

        # Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        # Wywołuje obiekt klasyfikatora najbliższych sąsiadów i trenuje dane wg tej metody
        Knn = KNeighborsClassifier(n_neighbors=5)
        Knn.fit(x_train, y_train)
        acc = Knn.score(x_test, y_test)

        print(acc)
        accList.append(acc)

        # Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = Knn.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        # zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:
            best_acc_score = acc
            pickleFileName = f"KNN_Model_{round(best_acc_score, 2)}p.pickle"

            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(Knn, f)

    print(f'Najwyższe acc modelu: {best_acc_score}')

def SaveBestDecTree_Model():
    # pętla wybierająca i zapsijąca z z 50 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):

        # Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        # Wywołuje obiekt klasyfikatora drzewa decyzyjego i trenuje dane wg tej metody
        DecTree = DecisionTreeClassifier()
        DecTree.fit(x_train, y_train)
        acc = DecTree.score(x_test, y_test)

        print(acc)
        accList.append(acc)

        # Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = DecTree.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        #zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:
            best_acc_score = acc
            pickleFileName = f"DecTree_Model_{round(best_acc_score, 2)}p.pickle"

            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(DecTree, f)

    print(f'Najwyższe acc modelu: {best_acc_score}')

def SaveBestRanForest_Model():
    # pętla wybierająca i zapsijąca z z 50 modeli, jednen z największą skutecznością
    best_acc_score = 0
    accList = []
    for acc_p in range(10):

        # Przypisuje do zmiennych treningowych i testowych, zestaw danch z przeprowadzoną selekcją cech.
        x_train, x_test, y_train, y_test = SignificanceSelector()

        # Wywołuje obiekt klasyfikatora Random Forrest i trenuje dane wg tej metody
        RanFroest = RandomForestClassifier(n_estimators=100)

        RanFroest.fit(x_train, y_train)

        acc = RanFroest.score(x_test, y_test)

        accList.append(acc)

        # Wykonuje predykcje cechy "y" przy użyciu tego modelu i wypisuje informacyjnie metryki modelu
        X_pred = RanFroest.predict(x_test)
        print(metrics.classification_report(y_test, X_pred))

        # zapisuje model z najlepszm wynikiem z 10
        if acc > best_acc_score:

            best_acc_score = acc
            pickleFileName = f"RanForest_Model_{round(best_acc_score, 2)}p.pickle"

            with open(f'Modele_Z_SelekcjaCech/{pickleFileName}', 'wb') as f:
                pickle.dump(RanFroest, f)

    print(f'Najwyższe acc modelu: {best_acc_score}')

"""
Wyniki najlepszej skuteczności z 10 prób bez selekcji cech: 
    -Regresja logistyczna: 85%
    -Naiwny Bayes: 86%
    -Liniowa analiza dyskryminacyjna: 90%
    -Najbliższi sąsiedzi: 89%
    -Drzewo dyecyzyjne: 88%
    -Random forest: 92%
"""

#SignificanceSelector()

#SaveBestLogisticRegressionModel()
#SaveBestGaussianNB_Model()
#SaveBestLDA_Model()
#SaveBestKNN_Model()
#SaveBestDecTree_Model()
#SaveBestRanForest_Model()
"""
Konkluzja:

Największą skutecznością do operacji na tak zebranych danych, okazała się klasyfikacja RandomForrest.
Jest to spowodowane specyficznym modelem który łączy w sobie cechy klasyfikatora lokalnego i globlanego. 

"""
