# SGF^2 - AI - Progetto di Intelligenza Artificiale

## Descrizione

Questo progetto universitario ha l'obiettivo di applicare tecniche di Machine Learning supervisato al dataset Adult (UCI) per predire se il reddito annuo di un individuo supera i 50.000 dollari. Il progetto include fasi di preprocessing, addestramento e valutazione di modelli, analisi della fairness e studio dell'importanza delle feature tramite ablation study.

## Dataset

Il dataset utilizzato e' l'**Adult Census Income Dataset** (UCI Machine Learning Repository), composto da circa 32.561 record con le seguenti caratteristiche principali:

- **Variabili**: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
- **Target**: income (<=50K / >50K)
- **Valori mancanti**: presenti nelle colonne `workclass`, `occupation`, `native.country` (indicati con `?`), rimossi in fase di preprocessing
- **Record dopo pulizia**: circa 30.162

## Struttura del Repository

```
SGF-AI---AI-PROJECT/
|-- notebooks/
|   |-- adult.csv                    # Dataset originale
|   |-- project_conablation.ipynb    # Notebook principale del progetto
|-- docs/
|   |-- fased1.jpg                   # Schema fase A (preprocessing)
|   |-- fased2.png                   # Schema fase B (modelli)
|   |-- fased3.png                   # Schema fasi C-D (fairness e ablation)
|   |-- LGP.png                      # Risultati Logistic Regression
|   |-- RFP.png                      # Risultati Random Forest
|   |-- SVM.png                      # Risultati SVM
|   |-- feature_importance.png       # Grafico importanza delle feature
|   |-- logounibo.png                # Logo Universita' di Bologna
|   |-- index.md                     # Documentazione aggiuntiva
|-- README.md
```

## Fasi del Progetto

### Fase A - Caricamento e Preprocessing del Dataset

- Caricamento del dataset `adult.csv`
- Rimozione dei record con valori mancanti (2.399 record rimossi)
- Raggruppamento delle categorie di `workclass` (es. self-employed, government, private)
- Codifica delle variabili categoriche tramite Label Encoding
- Suddivisione in Training set e Test set (80/20)

### Fase B - Addestramento e Valutazione dei Modelli

Sono stati addestrati e confrontati tre modelli di classificazione:

| Modello               | Accuratezza approssimativa |
|-----------------------|----------------------------|
| Logistic Regression   | ~80%                       |
| Random Forest         | ~82%                       |
| Support Vector Machine| ~81%                       |

Per ciascun modello sono state calcolate: accuracy, precision, recall, F1-score e matrice di confusione.

### Fase C - Analisi della Fairness

L'analisi della fairness e' stata condotta applicando la metrica **Disparate Impact (DI)** rispetto a due attributi sensibili:

- **Sesso (sex)**: DI ~0.30 - indica una significativa disparita' tra uomini e donne nella predizione del reddito >50K
- **Razza (race)**: DI ~0.60 - indica una disparita' moderata tra le diverse categorie razziali

Un valore di DI inferiore a 0.80 e' generalmente considerato indice di discriminazione.

### Fase D - Ablation Study e Importanza delle Feature

E' stato condotto uno studio di ablation sul modello Random Forest per valutare il contributo di ciascuna feature:

- La feature piu' importante risulta essere **marital-status**
- Seguono **capital-gain**, **age** ed **education-num**
- La rimozione progressiva delle feature meno rilevanti causa una diminuzione controllata delle prestazioni

## Requisiti

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Esecuzione

1. Clonare il repository:
   ```bash
   git clone https://github.com/FrancescoCastaldi/SGF-AI---AI-PROJECT.git
   ```
2. Installare le dipendenze:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Aprire il notebook:
   ```bash
   jupyter notebook notebooks/project_conablation.ipynb
   ```

## Risultati Principali

- Il modello Random Forest ottiene le prestazioni migliori in termini di accuratezza (~82%)
- L'analisi della fairness evidenzia disparita' significative legate al sesso e alla razza, con valori di Disparate Impact ben al di sotto della soglia di 0.80
- Lo studio sull'importanza delle feature mostra che lo stato civile e il guadagno di capitale sono i predittori piu' rilevanti per la stima del reddito .

## Licenza

MIT


