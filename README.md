# KlasyfkacjaRyzykaKredytowego
Skrypt klasyfikuje ryzyko kredytowe przy użyciu modeli uczenia maszynowego. 
Z próbi danych skrypt usówa cech o najmienjszym wpływie na model.
Następnie, każda funkcja odpowiada innemu modelowi klasyfikacjii. 
Klasyfikatory wykożystane:
  -Regresja liniowa
  -Naiwny bayes
  -LDA
  -KNN
  -Drzewo decyzyjne
  -Random Forest 
  
 Skrypt trenuje 10 przykładowych modeli, a następnie zapisuje ten z największą dokładnością(acc). 
 Następnie zapisuje gotowy model do pliku formatu "pickle", pozwala to na późniejsze wykożystanie modelu w klasyfikacjii. 

Konkluzja:
  Największą skutecznością do operacji na tak zebranych danych, okazała się klasyfikacja RandomForrest.
  Jest to spowodowane specyficznym modelem który łączy w sobie cechy klasyfikatora lokalnego i globlanego. 

