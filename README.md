# us-accidents-severity-ml
Projekt inżynierski: Analiza czynników wpływających na skutki zdarzeń drogowych z wykorzystaniem modeli uczenia maszynowego.

## Pobieranie Danych (Dataset)
Z uwagi na ogromny rozmiar bazy danych (~3gb), plik ze źródłowymi danymi nie jest trzymany bezpośrednio w tym repozytorium.
Aby uruchomić notatniki i potoki danych, należy samodzielnie pobrać najnowszą wersję zbioru `US Accidents (2016 - 2023)` z platformy Kaggle:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Pobrany plik US_Accidents_March23.csv należy umieścić w tym samym folderze, w którym uruchamiasz skrypty lub notatniki.

## Struktura Repozytorium

- data_pipeline.py – Skrypt przetwarzający dane dla wersji binarnej.

- data_pipeline_multiclass.py – Skrypt przetwarzający dane dla pełnej klasyfikacji wieloklasowej (Severity 1-4).

- praca inż_01.ipynb – Wstępny notatnik badawczy (analiza eksploracyjna, testy na próbie 10% danych oraz podejście binarne).

- praca inż_02.ipynb – Docelowy notatnik badawczy (pełny zbiór 100% danych, przejście na Multiclass...).

- catboost_info/ oraz __pycache__/ – Katalogi robocze generowane automatycznie przez środowisko wykonawcze.
