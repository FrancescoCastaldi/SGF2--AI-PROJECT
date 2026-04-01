# Analisi del Bias nel Dataset *Adult Census Income*

**Rilevazione e Quantificazione dei Bias Algoritmici in Contesti Socio-Economici**

**Autori:** Francesco Castaldi, Stefano Mercurio, Giovanni Previtera, Francesca Santoferrara  
*Università di Bologna — Corso di Intelligenza Artificiale*  
**Febbraio 2026**

---

## Indice

1. [Introduction](#1-introduction)
   - [Inquadramento del Problema](#11-inquadramento-del-problema)
   - [Soluzione Metodologica Proposta](#12-soluzione-metodologica-proposta)
   - [Suddivisione dei Task e Contributi](#13-suddivisione-dei-task-e-contributi)
2. [Proposed Method](#2-proposed-method)
   - [Preprocessing](#21-preprocessing)
   - [Training](#22-training)
   - [Methodology for Performance Measurement](#23-methodology-for-performance-measurement)
   - [Feature Attribution](#24-feature-attribution)
3. [Experimental Results](#3-experimental-results)
   - [Demonstration and Technologies](#31-demonstration-and-technologies)
   - [Results](#32-results)
   - [Feature Importance ed Interpretabilità](#33-feature-importance-ed-interpretabilità)
   - [Analisi della Proxy Discrimination](#34-analisi-della-proxy-discrimination)
4. [Discussion and Conclusions](#4-discussion-and-conclusions)
   - [Results Discussion](#41-results-discussion)
   - [Method Validity and Ablation Study](#42-method-validity-and-ablation-study)
   - [Limitations and Maturity](#43-limitations-and-maturity)
   - [Future Works](#44-future-works)
5. [Conclusioni Finali](#conclusioni-finali)
6. [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

## 1. Introduction

### 1.1 Inquadramento del Problema

Il presente studio si focalizza sullo sviluppo e sulla validazione critica di un sistema di classificazione binaria applicato al dataset *Adult Census Income* (UCI Machine Learning Repository). L'obiettivo primario consiste nel predire se il reddito annuo di un individuo superi la soglia dei 50.000 dollari, interpretando tale variabile non solo come un target predittivo, ma come un indicatore di potenziali disparità sistemiche riflesse nei dati.

Il dataset, basato sui dati del censimento statunitense del 1994, comprende 32.562 istanze e 14 variabili eterogenee. Oltre a descrittori demografici come `età`, `istruzione` e `stato civile`, il dataset include attributi sensibili quali `razza` e `sesso`. La distribuzione delle classi presenta un forte sbilanciamento (76% ≤50K vs 24% >50K), una caratteristica che riflette le asimmetrie economiche reali dell'epoca e che pone sfide significative in termini di generalizzazione del modello e di equità delle decisioni.

L'urgenza di tale analisi trascende l'esercizio accademico per investire la sfera etica e giuridica dell'Intelligenza Artificiale. In un contesto in cui gli algoritmi influenzano decisioni allocative, la presenza di bias non rilevati può cristallizzare discriminazioni storiche:

- **Settore Finanziario:** Un modello di *credit scoring* distorto potrebbe penalizzare sistematicamente le coorti femminili o le minoranze etniche, non per insolvibilità reale, ma per il riverbero di divari retributivi pregressi.
- **Human Resources:** Sistemi automatizzati di screening dei curricula rischiano di identificare erroneamente variabili proxy (come la zona di residenza o il percorso di studi) per discriminare indirettamente sulla base della razza.
- **Compliance Normativa:** L'adozione di modelli "scatola nera" in ambiti assicurativi o previdenziali solleva dubbi sulla legittimità costituzionale delle decisioni, violando i principi di uguaglianza sanciti a livello internazionale.

Per mitigare questi rischi, il framework proposto integra alle tradizionali metriche di performance (Accuracy, F1-score, AUC-ROC) una batteria di test per la *Algorithmic Fairness*. Nello specifico, vengono analizzati il *Disparate Impact* (DI) e l'*Equal Opportunity Difference* (EOD), focalizzando l'audit sugli attributi protetti di `genere` e `razza`. L'obiettivo finale è fornire uno strumento analitico utile sia per i **Data Scientist** orientati alla *Responsible AI*, sia per i **Regolatori** chiamati a far rispettare normative come l'EU AI Act.

### 1.2 Soluzione Metodologica Proposta

L'approccio adottato non si limita alla massimizzazione dell'accuratezza statistica, ma persegue un equilibrio tra performance e giustizia procedurale attraverso un protocollo strutturato in quattro fasi:

1. **Data Engineering e Preprocessing deterministico:** La pipeline include una normalizzazione robusta e un'estensione dello spazio delle feature tramite *One-Hot Encoding* per gestire l'eterogeneità delle variabili categoriali.
2. **Modellazione Multi-architettura:** Implementazione di tre paradigmi di apprendimento con diverse capacità di astrazione per verificare la stabilità del bias attraverso diversi confini decisionali.
3. **Valutazione Multidimensionale dell'Equità:** Quantificazione della disparità di trattamento secondo i parametri della EEOC, valutando il superamento della soglia critica dell'80%.
4. **Interpretabilità Post-hoc (XAI):** Impiego di tecniche di *Feature Importance* e *SHAP values* per isolare i fattori determinanti ed escludere l'influenza indebita di variabili discriminanti nascoste.

I modelli sono stati selezionati per coprire l'intero spettro tra trasparenza e potenza predittiva:

| Classificatore | Ruolo nell'Audit | Configurazione Strategica |
|---|---|---|
| Logistic Regression | Baseline per interpretabilità e analisi dei pesi lineari | `class_weight='balanced'` |
| Random Forest | Valutazione delle interazioni non lineari tra variabili socio-economiche | `n_estimators=100`, `max_depth=20` |
| SVM (RBF Kernel) | Analisi della separabilità dei gruppi in spazi ad alta dimensionalità | `C=1.0`, `gamma='scale'` |

Il framework affronta criticamente le **sfide computazionali** e metodologiche, come l'esplosione della dimensionalità (oltre 200 feature post-encoding) e la necessità di uno *Stratified Split* che preservi la distribuzione non solo del target, ma anche dei gruppi protetti, garantendo così la validità statistica dell'audit di fairness.

**Riferimenti Teorici e Stato dell'Arte:**

- **Hardt et al. (2016):** Il concetto di *Equality of Opportunity* viene qui utilizzato per garantire che i soggetti qualificati (reddito >50K) abbiano la stessa probabilità di essere identificati correttamente, indipendentemente dal gruppo di appartenenza.
- **Trade-off Accuracy-Fairness:** Si riconosce, come formalizzato da Barocas (2019), che l'imposizione di vincoli di equità può comportare una lieve riduzione dell'accuratezza globale, un costo eticamente necessario per evitare la discriminazione algoritmica.
- **Standard di Trasparenza:** L'integrazione di SHAP (Lundberg & Lee, 2017) risponde alla necessità di spiegabilità richiesta dai nuovi orientamenti normativi (AI Act), permettendo di decodificare il comportamento del modello in scenari ad alto rischio.

### 1.3 Suddivisione dei Task e Contributi

La realizzazione del progetto è stata ripartita tra i membri del gruppo di ricerca secondo una struttura funzionale volta all'ottimizzazione della pipeline analitica e della documentazione tecnica:

- **Francesco Castaldi:** Si è occupato delle fasi iniziali di *Data Ingestion* e *Preprocessing*. Ha curato il processo di codifica delle variabili (Feature Engineering) e la successiva implementazione del modello di addestramento, focalizzandosi sul *Fine-tuning* dei parametri algoritmici per l'ottimizzazione delle performance e "Ablation Study".
- **Stefano Mercurio:** Ha gestito le operazioni di *Data Cleaning* e la progettazione dei protocolli di addestramento. Ha condotto lo studio comparativo tra i diversi modelli di classificazione (Model Selection) e ha coordinato l'analisi sistematica del *Bias* algoritmico e delle metriche di equità.
- **Francesca Santoferrara:** Ha diretto la stesura della documentazione tecnica e la revisione del report scientifico. Ha contribuito attivamente alla fase di preparazione dei dati, occupandosi specificamente della pulizia e del campionamento stratificato per la definizione dei set di *Training* e *Testing*.
- **Giovanni Previtera:** Ha curato la sezione relativa alla interpretabilità implementando tecniche tramite framework **SHAP** e **LIME**. Ha inoltre collaborato alla redazione della documentazione tecnica e all'analisi dell'impatto delle feature sulle predizioni del modello.

---

## 2. Proposed Method

La scelta metodologica è stata guidata da un'attenta analisi comparativa delle alternative disponibili, bilanciando **efficacia predittiva**, **interpretabilità** e **fairness** sul dataset UCI Adult Income. Il metodo proposto può essere riassunto in 4 fasi che andremo a descrivere nel dettaglio.

### 2.1 Preprocessing

Trattamento sistematico del dataset:

| Operazione | Target | Razionale |
|---|---|---|
| `replace('?', pd.NA)` + `dropna` | workclass, occupation, native | Rimozione del 7.4% del dataset → null values |
| `drop('education')` | Ridondanza | education.num equivalente |
| `Grouping` | workclass | Raggruppamento in macro-categorie |
| `Encoding` | Target | ≤50K → 0, >50K → 1 |
| `Data Splitting` | All | 80% train, 20% test |
| `StandardScaler()` | 6 numeriche | μ=0, σ=1 |
| `OneHotEncoder()` | 7 categoriche | → 108 dummies |

> **Nota:** La tabella sopra riporta il preprocessing del Best case. Alcune modifiche hanno riscontrato dei casi peggiorativi che verranno discussi più avanti. Variabili come media e deviazioni standard, essenziali per eseguire lo scaling, vengono calcolati solo sul dataset di train. Il dataset di test non deve avere alcuna influenza, nemmeno durante la fase di preprocessamento.

### 2.2 Training

**Scelta dei modelli:** Logistic Regression (LR), Random Forest (RF), Support Vector Machine-Radial Basis Function (SVM-RBF), per copertura completa, ciascuno con vantaggi e svantaggi:

- **LR:** Fondamentale per avere una baseline di partenza. Usa il metodo della discesa del gradiente per imparare i pesi delle varie feature. Tale processo è iterativo e abbiamo settato l'iperparametro `max_iter` a 1000 in modo da dare al modello più tempo per raggiungere la convergenza. Grazie ai pesi è in grado di calcolare la probabilità del risultato. Graficamente è rappresentata come un iperpiano che separa le due classi.
- **RF:** Usa un certo numero di alberi di decisione. Grazie ad un algoritmo chiamato bagging lavorano su features e record del dataset differenti. Ognuno di essi fa una previsione e il risultato è dato da un meccanismo di voto. Gestisce al meglio gli outliers e non necessita dello scaling (ma non gli nuoce). Grazie agli iperparametri settati abbiamo controllato il numero di alberi, la loro profondità, il numero minimo di campioni presenti nelle foglie e nei nodi.
- **SVM-RBF:** L'obiettivo è massimizzare il margine (distanza) tra i vettori di supporto e il confine (iperpiano che separa le due classi). In caso di necessità proietta i dati in una dimensione più alta (Kernel Trick). La funzione kernel scelta è RBF che permette di creare confini curvi e complessi. L'alternativa Linear è troppo simile a LR.

Il processo di training e test segue la procedura del **Supervised Learning**:

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced',
    random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced',
    random_state=42, min_samples_split=5, min_samples_leaf=2),
    "SVM": SVC(kernel='rbf', class_weight='balanced', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_preprocessed, y_train)         # Allenamento
    y_pred = model.predict(X_test_preprocessed)       # Predizione
    print(f"{name} Performance:")
    print(classification_report(y_test, y_pred))      # Valutazione
    cm = confusion_matrix(y_test, y_pred)             # Matrice di confusione
    ConfusionMatrixDisplay(cm).plot()                 # Visualizzazione matrice di confusione
    plt.title(f"{name} Confusion Matrix")
    plt.show()
```

Nell'analisi del BIAS la valutazione è stata fatta applicando maschere al dataset di test. Questo è un procedimento cruciale per identificare il gruppo protetto e non protetto e calcolare le metriche di fairness.

### 2.3 Methodology for Performance Measurement

**Metriche ibride standard + fairness:**

| Categoria | Metriche Implementate |
|---|---|
| **Standard** | Accuracy, Precision/Recall/F1-score (>50K), matrici confusione, macro_avg, weighted_avg |
| **Fairness** | Disparate Impact (DI), Equal Opportunity Difference (EOD) |
| **Interpretability** | LR coefficients, RF feature importances |

**Disparate Impact (EEOC):**

$$\text{DI} = \frac{P(\hat{y}=1 \mid \text{protetto})}{P(\hat{y}=1 \mid \text{non-protetto})}$$

> **Soglia:** DI < 0.8 = ⚠️ discriminazione

**Equal Opportunity Difference:**

$$\text{EOD} = \text{TPR}_{\text{protetto}} - \text{TPR}_{\text{non-protetto}}$$

> **Soglia:** |EOD| > 0.1 = ⚠️ bias

### 2.4 Feature Attribution

- **LR:** Assume relazioni lineari. Attraverso i pesi assegnati a ciascuna feature determina non solo le feature più importanti ma anche la direzione.
- **RF:** Permette di catturare relazioni non lineari. Tuttavia fornisce solo informazioni di feature importance, non di direzione.
- **SVM:** A causa della scelta del kernel si perdono informazioni di interpretabilità.

---

## 3. Experimental Results

### 3.1 Demonstration and Technologies

Il progetto è implementato come notebook Jupyter strutturato in quattro celle sequenziali, ciascuna corrispondente a una fase della pipeline: preprocessing, addestramento, fairness audit e analisi delle feature. Il design deterministico (`random_state=42`, applicato a `train_test_split`, `LogisticRegression`, `RandomForestClassifier` e `SVC`) garantisce un elevato grado di riproducibilità dei risultati.

La riproduzione completa richiede due passi: il download del dataset UCI Adult Income dal repository ufficiale e l'installazione delle librerie Python necessarie, dopodiché è sufficiente eseguire le celle del notebook in sequenza.

| Libreria | Componente | Funzione Critica | Validazione |
|---|---|---|---|
| scikit-learn | ColumnTransformer | Preprocessing eterogeneo (StandardScaler + OneHotEncoder) | ✓ Gold standard ML |
| pandas | DataFrame | Caricamento, pulizia e rimozione valori mancanti | ✓ 7.4% missing rimossi |
| numpy | argsort / array ops | Ordinamento feature importance e operazioni vettoriali | ✓ Libreria numerica standard |
| matplotlib | ConfusionMatrixDisplay | Visualizzazione matrici di confusione e grafici fairness | ✓ Pubblicazioni peer-reviewed |

I valori mancanti presenti nelle colonne `workclass`, `occupation` e `native.country` sono gestiti tramite eliminazione diretta dei record incompleti (`dropna()`), che riduce il dataset da 32.561 a 30.162 istanze. La suddivisione train/test è effettuata in rapporto 80/20, con il test set composto da **6.033 istanze**.

### 3.2 Results

**Best configuration:** **Logistic Regression** (`C=1.0`, `class_weight='balanced'`) raggiunge **accuracy 84%** su test set bilanciato.

**Performance Completa — Test Set (6.033 istanze, 23.6% >50K):**

| Modello | Accuracy | Prec (>50K) | Rec (>50K) | F1 (>50K) | Support Cl.1 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.84** | 0.71 | 0.57 | 0.63 | 1537 |
| Random Forest | 0.83 | 0.61 | 0.80 | **0.69** | 1537 |
| SVM-RBF | 0.80 | 0.55 | **0.87** | 0.67 | 1537 |
| Majority Baseline | 0.76 | — | — | — | — |

I tre modelli evidenziano un chiaro trade-off nelle prestazioni. La Regressione Logistica massimizza l'accuratezza complessiva (0.84), risultando il modello più efficace in termini globali. La Random Forest ottimizza invece l'F1-score sulla classe minoritaria (0.69), grazie alla sua capacità di catturare relazioni non lineari e interazioni tra le feature. Infine, la SVM privilegia la recall (0.87), riducendo significativamente il numero di falsi negativi.

**Confusion Matrices** (n=1537 istanze classe >50K reali):

| Modello | Falsi Negativi | % FN |
|---|---|---|
| Logistic Regression | 660 FN | 43% falsi negativi |
| Random Forest | 309 FN | 20% falsi negativi |
| SVM-RBF | 207 FN | 13% — **best recall** |

Trade-off FN classe positiva: SVM > RF > LR

**Ablation Study — Sensitività e Ridondanza:**

| Ablazione / Modifica | ΔAcc. | ΔF1 | Significato Tecnico |
|---|---|---|---|
| **Baseline Completa** | 0% | 0% | Configurazione Ottimale |
| No `education` (solo `edu.num`) | **0.0%** | **0.0%** | **Eliminata Ridondanza** |
| No `class_weight='balanced'` | +2.4% | -2.9% | Perdita gestione sbilanciamento |

La rimozione della feature categorica `education` in favore della controparte numerica `education.num` non ha prodotto alcuna degradazione delle performance, confermando la ridondanza informativa della variabile. Al contrario, la rimozione di `class_weight='balanced'` causa una riduzione dell'F1-score del 2.9%, accompagnata però da un aumento dell'accuracy del 2.4% — un pattern tipico dello sbilanciamento del dataset, in cui il modello sacrifica la capacità di riconoscere la classe minoritaria a favore dell'accuratezza complessiva.

### 3.3 Feature Importance ed Interpretabilità

L'analisi dell'importanza delle feature tramite Random Forest consente di individuare i fattori più rilevanti per la predizione del reddito, fornendo una misura del contributo di ciascuna variabile. Questo approccio permette di identificare le feature con scarso potere predittivo, evidenziando quali risultano marginali nel processo decisionale del modello.

**Discussione dei fattori:**

- **Fattori Sociali:** `marital.status_Married-civ-spouse` è il predittore più forte (≈ 0.13), seguito da `education.num` (≈ 0.12). Il primo riflette la forte correlazione tra stato civile e reddito nei dati storici del 1994: la categoria è composta per l'89.5% da uomini, con reddito >50K nel 45.5% dei casi.
- **Fattori Economici e Demografici:** `age` e `capital.gain` si attestano entrambi a ≈ 0.095, riflettendo la naturale accumulazione di ricchezza nel tempo.
- **Giustificazione del Bias:** La presenza di `sex_Female`, `relationship_Husband` e `relationship_Wife` tra le top-15 feature spiega matematicamente le violazioni EEOC riscontrate: il modello *apprende attivamente* le disparità di genere presenti nei dati storici.

**Fairness Audit — Conformità normativa EEOC:**

| Modello | DI Gender | EOD Gender | DI Race | EOD Race |
|---|---|---|---|---|
| Logistic Regression | **0.298** | -0.142 | **0.580** | -0.064 |
| Random Forest | 0.277 | -0.155 | 0.522 | -0.114 |
| SVM-RBF | 0.291 | -0.152 | 0.601 | -0.066 |
| **EEOC: VIOLATO** | < 0.8 | \|>\|0.1 | < 0.8 | \|>\|0.1 |
| **Impatto pratico** | Donne: **-70%** | | Non-White: **-42%** | |

**DI=0.298** implica che le donne hanno il **70% in meno** di probabilità di ricevere una predizione positiva rispetto agli uomini. La convergenza di questi risultati su tre architetture differenti conferma che il **bias è ereditario del dataset** (historical bias) e non introdotto dall'implementazione algoritmica. **Violazione legale EEOC** su **entrambi gli attributi protetti**.

**Comparative Study — Benchmark UCI Adult:**

| Modello/Riferimento | Accuracy | DI Gender | Contesto |
|---|---|---|---|
| **LR (questo lavoro)** | **0.84** | 0.298 | **2026 — Fairness audit completo** |
| Random Forest | 0.83 | **0.277** | Questo lavoro |
| Zafar et al. — LR unconstrained [5] | 0.846 | ≈ 0.31 | JMLR 2019, Adult dataset |
| Ding et al. — LR standard [6] | 0.85 | — | NeurIPS 2021, UCI Adult |
| Majority Classifier | 0.76 | — | Baseline triviale |

I risultati di questo lavoro sono coerenti con i benchmark della letteratura: Zafar et al. (JMLR 2019) riportano accuracy 0.846 per una LR standard su UCI Adult con DI ≈ 0.31, mentre Ding et al. (NeurIPS 2021) confermano accuracy 0.85 per lo stesso modello. Il contributo distintivo di questo lavoro non è la performance predittiva in sé, ma l'integrazione sistematica del **fairness audit** — quantificazione di DI ed EOD su entrambi gli attributi protetti — all'interno di una pipeline riproducibile, assente nei lavori di riferimento citati.

### 3.4 Analisi della Proxy Discrimination

Durante il processo di sviluppo della pipeline, prima di giungere alla configurazione attuale, sono state esplorate due direzioni alternative che si sono rivelate controproducenti. Forniscono, però, evidenza di un principio fondamentale della fairness algoritmica: **il bias non è eliminabile per semplice omissione di dati sensibili**.

#### Rimozione dei Proxy Diretti del Genere

Per ridurre il bias di genere, è stata ipotizzata la rimozione di `marital.status` e `relationship`, le due colonne che fungono da proxy diretti del sesso. La motivazione era fondata su evidenze empiriche solide: la categoria `Married-civ-spouse` è composta per l'89.5% da uomini (12.585 vs 1.480 donne) e predice reddito elevato nel 45.5% dei casi; `Husband` è maschile al 100%, `Wife` femminile al 100%. Rimuovere queste colonne avrebbe dovuto rendere il modello *indipendente* al genere.

Il risultato è stato l'opposto: il bias di genere è peggiorato drasticamente su tutti i modelli.

**Confronto Fairness Genere: Soluzione Ottimale vs Rimozione Proxy:**

| Modello | DI | DI Test 1 | EOD | EOD Test 1 |
|---|---|---|---|---|
| Logistic Regression | 0.298 | 0.267 | -0.142 | **-0.333** |
| Random Forest | 0.277 | 0.185 | -0.155 | **-0.387** |
| SVM-RBF | 0.291 | 0.209 | -0.152 | **-0.350** |

**Confronto Fairness Razza: Soluzione Ottimale vs Rimozione Proxy:**

| Modello | DI | DI Test 1 | EOD | EOD Test 1 |
|---|---|---|---|---|
| Logistic Regression | 0.580 | 0.547 | -0.064 | -0.143 |
| Random Forest | 0.522 | 0.513 | -0.114 | -0.152 |
| SVM-RBF | 0.601 | 0.581 | -0.066 | -0.101 |

Il bias di genere è peggiorato drasticamente: l'EOD di SVM passa da -0.152 a -0.350, più che raddoppiato, mentre il bias razziale è rimasto quasi invariato. Questo dimostra che `marital.status` e `relationship` sono proxy quasi esclusivamente del genere. Privato di queste colonne, il modello ha spostato il peso predittivo su variabili correlate come `age` e `occupation`, amplificando il bias in modo più opaco: questo fenomeno è noto come **proxy discrimination**.

Come confermato dall'analisi della Fase D, `age` raggiunge un'importanza relativa di circa 0.22, quasi il doppio rispetto alla configurazione originale dove si attestava a circa 0.10. Il bias non è stato eliminato, ma *nascosto* in variabili demografiche che il modello usa come sostituti indiretti del sesso.

**Tentativo parallelo sulla razza.** È stato replicato lo stesso esperimento per la razza, rimuovendo le colonne più correlate ad essa: `native.country` (la più correlata, con varianza della percentuale di bianchi pari a 0.1737) e `occupation`. Il peggioramento del bias razziale è risultato tuttavia trascurabile.

**Confronto: Soluzione Ottimale vs Rimozione `native.country` e `occupation`:**

| Modello | DI | DI Tentativo | EOD | EOD Tentativo |
|---|---|---|---|---|
| Logistic Regression | 0.580 | 0.557 | -0.064 | -0.093 |
| Random Forest | 0.522 | 0.504 | -0.114 | -0.118 |
| SVM-RBF | 0.601 | 0.593 | -0.066 | -0.075 |

Il motivo è strutturale: il 93.3% dei neri e il 94.8% degli Amer-Indian-Eskimo è nato negli Stati Uniti, rendendo `native.country` un proxy debole della razza per la maggioranza delle minoranze. A differenza del genere, il bias razziale è distribuito su molteplici variabili in modo diffuso e non isolabile tramite la rimozione di poche colonne. Questa asimmetria strutturale tra le due forme di bias è rilevante: il bias di genere è **concentrato e isolabile**, quello razziale è **diffuso e sistemico**.

#### Rimozione del Bilanciamento delle Classi

Il secondo test preliminare interviene sulla gestione dello sbilanciamento del target. Il dataset presenta una distribuzione naturalmente asimmetrica — 76% ≤50K vs 24% >50K — che la configurazione ottimale compensa utilizzando `class_weight='balanced'`. In questa fase esplorativa tale parametro è stato rimosso da tutti e tre i modelli, simulando l'errore comune di ignorare lo sbilanciamento delle classi.

Senza questa compensazione, il modello minimizza la loss ignorando la classe minoritaria (>50K): l'accuracy apparente rimane elevata o addirittura aumenta, ma la recall sulla classe positiva crolla drasticamente, rendendo il sistema inutile per il task reale di identificare i redditi elevati.

**Confronto Performance: Soluzione Ottimale vs Senza Bilanciamento:**

| Modello | Acc. | Rec. cl.1 | F1 cl.1 | Acc. (no weight) | Rec. cl.1 (no weight) | F1 cl.1 (no weight) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.57 | 0.63 | 0.84 | 0.60 | 0.65 |
| Random Forest | 0.83 | **0.80** | **0.69** | 0.85 | ⚠️ 0.59 | 0.67 |
| SVM-RBF | 0.80 | **0.87** | 0.67 | 0.85 | ⚠️ 0.59 | 0.65 |

Il risultato è controintuitivo: l'accuracy apparente sale (RF: 0.83 → 0.85, SVM: 0.80 → 0.85), ma la recall sulla classe positiva crolla drasticamente — RF passa da 0.80 a 0.59 (-26%), SVM da 0.87 a 0.59 (-32%). Il modello smette di identificare i redditi elevati perché predice quasi sempre ≤50K: **l'accuracy alta maschera un collasso della capacità predittiva reale**.

Sul fronte fairness, le metriche di genere migliorano apparentemente (EOD LR: -0.142 → -0.074) perché uomini e donne ricevono lo stesso trattamento sbagliato — non equità, ma uniformità nel fallimento. Il bias razziale invece peggiora realmente, con RF che passa da EOD -0.114 a -0.135.

Questi test preliminari hanno orientato la configurazione finale verso la soluzione ottimale descritta nelle sezioni precedenti. La prima direzione esplorata ha dimostrato che rimuovere le variabili proxy sposta il peso predittivo su variabili correlate, amplificando la discriminazione in modo più opaco. La seconda ha evidenziato come ignorare lo sbilanciamento del target produca un'equità superficiale che nasconde un collasso delle performance. Entrambi i fallimenti confermano la necessità di un approccio *fairness-by-design* sistematico.

---

## 4. Discussion and Conclusions

### 4.1 Results Discussion

L'analisi comparativa evidenzia prestazioni che superano le aspettative teoriche per il dataset UCI Adult: il raggiungimento di un **accuracy dell'84%** con Logistic Regression segna un incremento del **+7.6%** rispetto al majority baseline (0.76), confermando che le scelte progettuali adottate — in particolare l'uso di `class_weight='balanced'` e il preprocessing eterogeneo tramite `ColumnTransformer` — hanno un impatto determinante sulle performance reali.

Il panorama dei modelli riflette un chiaro **trade-off prestazionale**, visibile nelle matrici di confusione:

- **Logistic Regression** (`C=1.0, class_weight='balanced'`): modello più equilibrato per l'accuracy globale (0.84), con precision 0.71 e recall 0.57 sulla classe positiva. Presenta però 660 falsi negativi (43%), il numero più alto tra i tre modelli.
- **Random Forest:** ottimizza l'F1-score sulla classe minoritaria (**0.69**) grazie alla capacità di catturare relazioni non lineari e interazioni tra le feature. La recall sulla classe positiva raggiunge 0.80, con soli 309 FN (20%).
- **SVM-RBF:** eccelle nella **recall (0.87)**, riducendo i falsi negativi al minimo (207 FN, 13%), parametro critico in contesti di screening socio-economico dove classificare erroneamente un individuo ad alto reddito come a basso reddito ha conseguenze concrete. Il costo è una precision più bassa (0.55).

La scelta del modello ottimale dipende quindi dal contesto applicativo: LR per massimizzare l'accuracy complessiva, RF per bilanciare precision e recall, SVM per minimizzare i falsi negativi.

**Fairness — convergenza e impossibilità di correzione semplice:**

I valori di *Disparate Impact* (DI) misurati mostrano una violazione sistematica della soglia normativa **EEOC di 0.8** su tutti e tre i modelli:

**Analisi della Disparità nei Sottogruppi (Logistic Regression):**

| Gruppo Protetto | False Negative Rate | Disparate Impact | Impatto Sociale |
|---|---|---|---|
| Uomini (Reference) | 20.1% | 1.00 | Baseline |
| **Donne** | **42.3% (+110%)** | **0.278** | -72% Opportunità |
| Bianchi (Reference) | 21.5% | 1.00 | Baseline |
| **Minoranze Etniche** | **37.2% (+73%)** | **0.514** | -49% Opportunità |

La convergenza di questi risultati su tre architetture differenti (DI Gender: 0.277–0.298; DI Race: 0.522–0.601) conferma che il bias è **ereditario del dataset** (historical bias) e non introdotto dall'implementazione algoritmica. I dati del censimento del 1994 riflettono strutture socio-economiche storicamente discriminatorie che il modello apprende e riproduce fedelmente.

### 4.2 Method Validity and Ablation Study

La validità metodologica è supportata da un protocollo sperimentale rigoroso e da tre evidenze chiave emerse dallo **Studio di Ablation (Fase D)**:

1. **Ridondanza feature (`education` vs `education.num`):** La rimozione della variabile categorica `education` in favore della controparte numerica `education.num` non ha prodotto alcuna degradazione (ΔAcc = 0%, ΔF1 = 0%), confermando la ridondanza informativa e giustificando la semplificazione della pipeline.

2. **Criticità del bilanciamento (`class_weight='balanced'`):** La rimozione di questo parametro causa un crollo della recall (RF: da 0.80 a 0.59; SVM: da 0.87 a 0.59). Il risultato paradossale è che l'accuracy *apparente* sale (RF: 0.83 → 0.85), mascherando il collasso reale. Sul fronte fairness, le metriche di genere migliorano apparentemente (EOD LR: -0.142 → -0.074) non per equità, ma perché uomini e donne ricevono lo stesso trattamento errato.

3. **Interpretabilità — Gini Importance:** I driver principali della predizione sono `marital.status_Married-civ-spouse` (≈0.13), `education.num` (≈0.12), `capital.gain` (≈0.095), `age` (≈0.095), `hours.per.week` (≈0.05). La presenza di `sex_Female`, `relationship_Husband` e `relationship_Wife` nelle top-15 spiega matematicamente le violazioni EEOC riscontrate: il modello apprende e codifica attivamente le disparità di genere presenti nei dati storici del 1994.

### 4.3 Limitations and Maturity

Il framework si colloca a **TRL 5** (Technology Readiness Level): la pipeline è funzionante, testata su dati reali (UCI Adult, 30.162 istanze) e produce risultati riproducibili, ma non è integrata in un sistema di produzione con deployment e re-training automatico (requisiti TRL 7+).

Persistono quattro limitazioni strutturali:

| Limitazione | Effetto Critico | TRL | Mitigazione |
|---|---|---|---|
| Obsolescenza Dati (1994) | Forte rischio di *data drift*: pattern socio-economici del 1994 non riflettono la realtà attuale | 5 | Folktables / ACS 2021–2025 |
| Assenza Causalità | Apprendimento di correlazioni storiche, non causali (es. `marital.status` → reddito per correlazione sesso-reddito nel 1994) | 3 | Grafi causali (DoWhy) |
| Analisi non intersezionale | DI ed EOD calcolate separatamente per genere e razza, non per g × r (bias amplificati nascosti) | 4 | Audit fattoriale 2^k |
| Alta Dimensionalità | 108 feature OHE aumentano overfitting su campioni ridotti e complessità interpretativa | 5 | Target Encoding / PCA |

**Obsolescenza Dati (1994) & Forte rischio di *data drift*:** I pattern socio-economici del 1994 non riflettono la realtà attuale. Ad esempio, nel censimento del 1994 solo il 55.7% delle donne lavorava, mentre oggi la partecipazione femminile al mercato del lavoro è salita oltre il 70%. **Soluzione:** Migrazione su dataset moderni tramite *Folktables* / ACS 2021–2025.

**Assenza Causalità:** Il modello impara che `marital.status` predice il reddito perché nel 1994 questa variabile era fortemente correlata al sesso (89.5% degli sposati erano uomini), non perché lo stato civile causi direttamente il reddito. **Soluzione:** Integrazione di grafi causali tramite libreria *DoWhy*.

**Analisi non intersezionale:** Le metriche di fairness sono misurate separatamente per genere e razza, ma non per la combinazione g × r (es. donne non-bianche). **Soluzione:** Audit fattoriale 2^k per valutare tutte le combinazioni di attributi protetti.

**Alta Dimensionalità:** Il One-Hot Encoding delle variabili categoriche produce 108 feature, aumentando il rischio di overfitting. **Soluzione:** Applicare *Target Encoding* o riduzione dimensionale tramite PCA.

### 4.4 Future Works

La roadmap verso un'architettura *Fairness-by-Design* a TRL 7+ si articola in quattro interventi concreti, ciascuno mirato a risolvere una specifica limitazione identificata:

1. **SMOTE-NC per il bilanciamento (Pre-processing):** L'ablation study ha dimostrato che la gestione dello sbilanciamento è critica: senza `class_weight='balanced'` la recall crolla del 32% su SVM. **SMOTE-NC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous data) genera istanze sintetiche della classe minoritaria (>50K) rispettando la natura mista del dataset, permettendo di eliminare la dipendenza dal parametro `class_weight` e migliorare la generalizzazione su distribuzioni più bilanciate.

2. **Fairness Constraints in-processing:** L'attuale approccio misura il bias *dopo* il training (post-hoc). L'evoluzione naturale è integrare il vincolo di equità direttamente nella funzione di perdita durante l'ottimizzazione:

$$\min_\theta \mathcal{L}(\theta) \quad \text{s.t.} \quad \left|P(\hat{Y}=1 \mid G=0) - P(\hat{Y}=1 \mid G=1)\right| \leq \epsilon$$

   Questo approccio, formalizzato da Zafar et al. [5], permetterebbe di superare la soglia EEOC (DI ≥ 0.8) senza rinunciare all'accuracy, bilanciando il trade-off equità-performance tramite il parametro ε.

3. **Monitoraggio continuo con PSI e test KS (Post-processing):** Il dataset del 1994 introduce un rischio concreto di *data drift*. Si prevede l'integrazione di un sistema di monitoraggio basato sul **Population Stability Index (PSI)** e sul test di **Kolmogorov-Smirnov** per rilevare derive statisticamente significative. Al superamento di una soglia critica (PSI > 0.2), il sistema attiverebbe automaticamente una procedura di re-training.

4. **Migrazione a Folktables / ACS 2021–2025 (Scalabilità):** Il framework **Folktables** (Ding et al. [6]) permette di estrarre task di classificazione dai censimenti US ACS aggiornati al 2021–2025, con campioni di oltre 1.2 milioni di istanze. La migrazione consentirebbe: (a) validare la robustezza della pipeline su dati moderni, (b) verificare se il bias strutturale persiste nei dati contemporanei, (c) confrontare i risultati con il benchmark diretto di Ding et al.

---

## Conclusioni Finali

Il presente lavoro ha dimostrato quattro contributi empirici fondamentali:

1. **Validazione Performance:** Accuracy dell'84% con Logistic Regression, coerente con i benchmark della letteratura (Zafar et al. 2019: 0.846; Ding et al. 2021: 0.85) e superiore al majority baseline del 7.6%.

2. **Quantificazione del Bias:** Rilevazione sistematica di violazioni EEOC su entrambi gli attributi protetti (DI Gender = 0.298, DI Race = 0.580 per LR), evidenziando la discriminazione strutturale nei dati del censimento 1994.

3. **Efficacia dell'Ablation:** Dimostrazione empirica che (a) la rimozione di variabili proxy del genere (`marital.status`, `relationship`) peggiora il bias invece di ridurlo (fenomeno di *proxy discrimination*), e (b) il bilanciamento delle classi è condizione necessaria per la correttezza predittiva reale (senza `class_weight='balanced'` la recall crolla del 32%).

4. **Framework TRL 5:** Sviluppo di una pipeline diagnostica end-to-end, deterministica e riproducibile, scalabile e pronta per audit di conformità normativa.

In conclusione, il progetto sancisce la necessità di uno **shift paradigmatico**: dalla pura massimizzazione dell'accuracy a un approccio *fairness-by-design*, dove l'audit etico è parte integrante del ciclo di vita del modello AI.

---

**Supplementary Materials:** Codice/dashboard su GitHub, modelli joblib serializzati.

---

## Riferimenti Bibliografici

[1] Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.

[2] Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.

[3] Mehrabi, N. et al. (2021). A Survey on Bias and Fairness in Machine Learning. *ACM Computing Surveys*, 54(6):1–35.

[4] Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

[5] Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2019). *Fairness Constraints: A Flexible Approach for Fair Classification*. Journal of Machine Learning Research, 20, 1–42.

[6] Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). *Retiring Adult: New Datasets for Fair Machine Learning*. NeurIPS.
