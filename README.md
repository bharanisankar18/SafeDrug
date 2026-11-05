# 💊 SafeDose: Big Data Pharmacovigilance & Safety Signal Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark%204.x-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-brightgreen.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, big data post-market safety analytics platform that ingests raw adverse drug event reports from the **FDA Adverse Event Reporting System (FAERS)**, cleans and standardizes pharmacovigilance data using **Apache Spark**, performs statistical disproportionality signal detection (**PRR**, **ROR**, **$\chi^2$**), mines polypharmacy drug-drug interactions with **Spark MLlib FP-Growth**, flags temporal reporting spikes ($Z \ge 3.0$), and surfaces clinical insights through an interactive **Streamlit + Plotly** dashboard.

---

> [!WARNING]
> **Regulatory & Pharmacovigilance Disclaimer:**  
> Safety signals detected by SafeDose are statistical disproportionality associations derived from spontaneous adverse event reporting systems (FDA FAERS). **Spontaneous reports establish statistical association, not clinical causation.** Every detected signal is an exploratory hypothesis intended to prioritize medications for formal epidemiological and clinical investigation.

---

## 📑 Table of Contents
1. [Clinical & Regulatory Context](#1-clinical--regulatory-context)
2. [System Architecture](#2-system-architecture)
3. [Statistical Methodology](#3-statistical-methodology)
   - [Disproportionality Analysis (Evans Criteria)](#disproportionality-analysis-evans-criteria)
   - [Polypharmacy Mining (FP-Growth)](#polypharmacy-mining-fp-growth)
   - [Time-Series Spike Detection (Z-Score)](#time-series-spike-detection-z-score)
4. [Dataset & Storage Layout](#4-dataset--storage-layout)
5. [Interactive Dashboard Features](#5-interactive-dashboard-features)
6. [Pipeline Execution Guide](#6-pipeline-execution-guide)
7. [Key Empirical Results](#7-key-empirical-results)
8. [Pharmacovigilance Limitations](#8-pharmacovigilance-limitations)

---

## 1. Clinical & Regulatory Context

Pre-approval clinical trials evaluate drug safety in small, homogeneous, highly monitored patient cohorts over restricted timeframes. Consequently, rare adverse drug reactions (ADRs), long-term toxicities, and severe multi-drug interactions frequently remain undetected until post-market deployment.

The **FDA Adverse Event Reporting System (FAERS)** serves as the primary post-market surveillance tool for the FDA, receiving millions of voluntary and mandatory reports from healthcare providers, consumers, and pharmaceutical manufacturers. 

However, FAERS data is voluminous, uncurated, and heavily confounded. **SafeDose** addresses these challenges by:
- Ingesting and deduplicating multi-gigabyte quarterly OpenFDA FAERS JSON records.
- Standardizing drug identities (preferring OpenFDA generic names over brand names) and reactions (MedDRA terminology).
- Computing rigorous disproportionality metrics at scale.
- Identifying dangerous multi-drug synergies across polypharmacy regimens.
- Detecting sudden temporal reporting surges that warrant immediate regulatory inspection.

---

## 2. System Architecture

SafeDose follows a decoupled **Batch Analytics + Fast Serving** architecture:

```
+----------------------------------------------------------------------------------+
|                             DATA INGESTION & PIPELINE                            |
+----------------------------------------------------------------------------------+
  [FDA FAERS JSON Partitions] (1.58 GB Raw)
             │
             ▼
  ┌─────────────────────────────────┐
  │  src/02_download.py             │ Ingests & validates quarterly JSON files
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  src/03_clean.py (PySpark)      │ Deduplicates reports, standardizes names,
  │                                 │ filters primary suspect drugs
  └────────────────┬────────────────┘
                   │
                   ├─────────────────────────────────────────┐
                   ▼                                         ▼
  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
  │  src/04_signals.py (PySpark)    │       │  src/05_interactions.py (MLlib) │
  │  Computes PRR, ROR, Chi-Square  │       │  FP-Growth frequent itemsets &  │
  │  Applies Evans Signal Criteria  │       │  excess reaction rate ratios    │
  └────────────────┬────────────────┘       └────────────────┬────────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │  src/06_trends.py (PySpark)             │
                   │  Monthly time-series & Z-score spikes   │
                   └────────────────────┬────────────────────┘
                                        │
+----------------------------------------------------------------------------------+
|                            ANALYTICAL STORAGE (PARQUET)                          |
+----------------------------------------------------------------------------------+
  data/processed/
    ├── clean_reports.parquet   (377,130 records, ~1.7 MB)
    ├── signals.parquet         (158,414 pairs, 27,860 signals, ~5.1 MB)
    ├── interactions.parquet    (561 frequent pairs, ~30 KB)
    └── trends.parquet          (117,452 monthly series points, ~1.2 MB)
                                        │
+----------------------------------------------------------------------------------+
|                            STREAMLIT + PLOTLY DASHBOARD                          |
+----------------------------------------------------------------------------------+
                   ┌─────────────────────────────────────────┐
                   │  src/app.py                             │
                   │  Sub-second cached reads via PyArrow    │
                   │  6 Interactive Clinical Intelligence    │
                   │  Tabs with Plotly & NetworkX            │
                   └─────────────────────────────────────────┘
```

---

## 3. Statistical Methodology

### Disproportionality Analysis (Evans Criteria)

For each drug $D$ and adverse reaction $R$, a standard $2 \times 2$ contingency matrix is assembled across all FAERS reports:

| | Adverse Reaction $R$ | All Other Reactions ($\neg R$) | Total |
| :--- | :---: | :---: | :---: |
| **Drug $D$** | $a$ | $b$ | $a + b$ |
| **All Other Drugs ($\neg D$)** | $c$ | $d$ | $c + d$ |
| **Total** | $a + c$ | $b + d$ | $N$ |

#### 1. Proportional Reporting Ratio (PRR)
Measures the relative proportion of reports for reaction $R$ among cases receiving drug $D$ versus all other drugs:
$$\text{PRR} = \frac{a / (a + b)}{c / (c + d)}$$

With standard $95\%$ Log-Normal Confidence Interval:
$$\text{SE}(\ln \text{PRR}) = \sqrt{\frac{1}{a} - \frac{1}{a + b} + \frac{1}{c} - \frac{1}{c + d}}$$
$$\text{CI}_{95\%} = \exp\left(\ln(\text{PRR}) \pm 1.96 \cdot \text{SE}(\ln \text{PRR})\right)$$

#### 2. Reporting Odds Ratio (ROR)
Odds of reporting reaction $R$ for drug $D$ relative to other drugs:
$$\text{ROR} = \frac{a \cdot d}{b \cdot c}$$
$$\text{SE}(\ln \text{ROR}) = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$$
$$\text{CI}_{95\%} = \exp\left(\ln(\text{ROR}) \pm 1.96 \cdot \text{SE}(\ln \text{ROR})\right)$$

#### 3. Yates Continuity-Corrected Chi-Square ($\chi^2$)
Tests the null hypothesis that drug $D$ and reaction $R$ are independent:
$$\chi^2 = \frac{N \cdot \left(\max(0, |a \cdot d - b \cdot c| - \frac{N}{2})\right)^2}{(a + b)(c + d)(a + c)(b + d)}$$

#### Evans Decision Rule for Safety Signals
A drug-reaction pair is categorized as a **Validated Safety Signal** if and only if:
1. $a \ge 3$ (at least 3 independent case reports)
2. $\text{PRR} \ge 2.0$
3. $\chi^2 \ge 4.0$ ($p < 0.05$ with 1 degree of freedom)

---

### Polypharmacy Mining (FP-Growth)

Patients taking multiple medications frequently experience synergistic adverse events. SafeDose leverages **Apache Spark MLlib FP-Growth** (Frequent Pattern Growth):
1. **Basket Generation**: Groups all primary suspect and concomitant drugs per unique `safetyreportid`.
2. **Frequent Itemset Mining**: Discovers pairs of medications occurring together above minimum support:
   $$\text{Support}(D_1, D_2) = \frac{\text{Reports}(D_1 \cup D_2)}{N_{\text{multi-drug reports}}}$$
3. **Reaction Rate Ratio (RRR)**: Compares observed adverse reaction intensity against expected single-agent baselines:
   $$\text{Baseline}(D_1, D_2) = \frac{\overline{\text{Rx}}(D_1) + \overline{\text{Rx}}(D_2)}{2}$$
   $$\text{RRR} = \frac{\overline{\text{Rx}}(D_1, D_2)}{\text{Baseline}(D_1, D_2)}$$
4. **High-Risk Interaction Flag**: Combination is flagged high-risk if $\text{RRR} \ge 1.20$ or Serious Outcome Rate $\ge 70\%$.

---

### Time-Series Spike Detection ($Z$-Score)

To identify sudden surges in reporting volume (e.g. batch contamination, media-driven reporting, or emergent safety crises):
1. Aggregates case reports by month $t$ (`year_month`).
2. Calculates historical baseline mean ($\mu$) and standard deviation ($\sigma$) per drug/reaction series.
3. Computes the standardized score:
   $$Z_t = \frac{C_t - \mu}{\sigma} \quad (\text{for } \sigma > 0)$$
4. Flags an **Anomalous Volume Surge** when $Z_t \ge +3.0$ and $C_t \ge 3$.

---

## 4. Dataset & Storage Layout

SafeDose stores intermediate and analytical data using Snappy-compressed Parquet:

```
SafeDose/
├── data/
│   ├── raw/                           # Raw OpenFDA FAERS JSON files (~1.58 GB)
│   │   ├── drug-event-0001-of-0029.json
│   │   ├── drug-event-0002-of-0029.json
│   │   └── drug-event-0003-of-0029.json
│   └── processed/                     # Analytical Parquet tables
│       ├── clean_reports.parquet      # 377,130 rows (Clean drug-reaction records)
│       ├── signals.parquet            # 158,414 pairs (27,860 Evans signals)
│       ├── interactions.parquet       # 561 frequent pairs (560 high-risk)
│       └── trends.parquet             # 117,452 monthly time-series points
├── hadoop/
│   └── bin/                           # winutils.exe and hadoop.dll for Windows
├── src/
│   ├── spark_init.py                  # PySpark environment & JVM memory tuning
│   ├── 02_download.py                 # Ingestion & verification script
│   ├── 03_clean.py                    # PySpark deduplication & standardization
│   ├── 04_signals.py                  # PySpark PRR, ROR, Chi-Square signal mining
│   ├── 05_interactions.py             # Spark MLlib FP-Growth interaction analysis
│   ├── 06_trends.py                   # PySpark time-series & Z-score surge engine
│   ├── 07_hadoop_showcase.py          # Native Hadoop FileSystem & MapReduce workflow
│   └── app.py                         # Streamlit + Plotly interactive UI
└── README.md
```

---

## 5. Interactive Dashboard Features

The dashboard (`src/app.py`) provides 7 interactive pharmacovigilance and Big Data modules:

| Tab | Key Functionality |
| :--- | :--- |
| **📊 Overview** | High-level KPIs, regulatory disclaimer banner, patient demographics (age, sex, seriousness), and top-reported medications and reactions. |
| **🔍 Drug Explorer** | Drug lookup across 3,127 medications; displays case report counts, seriousness %, age histograms, top 15 reactions horizontal bar chart, and co-prescriptions. |
| **🚨 Safety Signals** | Interactive Evans Criteria filtering (PRR, $\chi^2$, counts), formatted table with $95\%$ CIs, and a **Volcano Scatter Plot** ($\log_2(\text{PRR})$ vs $\log_{10}(\chi^2)$). |
| **⚖️ Compare Drugs** | Direct head-to-head safety comparison between any two drugs (e.g. `adalimumab` vs. `methotrexate`), showing side-by-side demographics and comparative reaction rates. |
| **📈 Trends & Spikes** | Monthly adverse event reporting time-series with historical mean baseline ($\mu$), $+3\sigma$ surge threshold line, and diamond markers highlighting anomalous spikes ($Z \ge 3.0$). |
| **🕸️ Network Graph** | **NetworkX + Plotly** graph mapping multi-drug co-prescription networks discovered by FP-Growth with degree-based coloring and customizable thresholds. |
| **🐘 Hadoop Architecture** | Direct inspection of **Apache Hadoop 3.5.0**, `org.apache.hadoop.fs.FileSystem`, block sizes, and live **Hadoop MapReduce** output metrics. |

---

## 6. Pipeline Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Java Development Kit (JDK 17, 21, or 24)
- 4 GB+ RAM

### 1. Install Dependencies
```bash
pip install pyspark pandas pyarrow streamlit plotly networkx requests
```

### 2. Run Data Processing Pipeline (Step-by-Step)
Each pipeline step is modular, self-contained, and writes directly to `data/processed/`:

```powershell
# Step 2: Verify Raw Ingestion
python src/02_download.py

# Step 3: PySpark Data Cleaning & Deduplication
python src/03_clean.py

# Step 4: Disproportionality Signal Detection (PRR, ROR, Chi-Square)
python src/04_signals.py

# Step 5: Multi-Drug Interaction Mining (Spark MLlib FP-Growth)
python src/05_interactions.py

# Step 6: Temporal Trends & Z-Score Surge Detection
python src/06_trends.py

# Optional Showcase: Native Hadoop FileSystem & MapReduce Verification
python src/07_hadoop_showcase.py
```

### 3. Launch the Interactive Dashboard
```powershell
streamlit run src/app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Key Empirical Results

From processing 36,000 raw FAERS adverse event reports (~1.58 GB):

- **Unique Monitored Drugs**: 3,127
- **Clean Drug-Reaction Event Pairs**: 377,130
- **Total Evaluated Drug-Reaction Contingency Pairs**: 158,414
- **Active Evans Safety Signals**: **27,860** (17.6% of evaluated pairs)
- **Frequent Polypharmacy Drug Combinations Mined**: **561** pairs
- **Anomalous Reporting Spikes Detected ($Z \ge 3.0$)**: **179** spikes

### Top Detected High-Risk Drug Interactions

| Drug 1 | Drug 2 | Co-Reports | Avg Rx / Case | Reaction Rate Ratio | Serious % | Top Adverse Reaction |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **acetaminophen** | **methotrexate** | 32 | 52.31 | **6.34x** | 100% | dyspnoea |
| **acetaminophen** | **cosentyx** | 30 | 55.17 | **6.03x** | 100% | hypertension |
| **acetaminophen** | **simponi** | 32 | 56.69 | **5.98x** | 100% | dyspnoea |
| **acetaminophen** | **etanercept** | 42 | 50.26 | **5.30x** | 100% | fatigue |
| **cosentyx** | **rituximab** | 36 | 53.50 | **5.09x** | 100% | fatigue |

### Top Reporting Volume Spikes ($Z$-Scores)

| Drug Name | Reaction Scope | Month | Observed Cases | Historical Mean | $Z$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **sodium oxybate** | ALL | 2024-03 | 16 | 1.68 | **+6.88** |
| **inflectra** | ALL | 2024-01 | 34 | 3.81 | **+5.66** |
| **omalizumab** | ALL | 2024-02 | 581 | 25.07 | **+5.00** |
| **acetaminophen** | ALL | 2024-01 | 190 | 13.09 | **+4.64** |
| **xeljanz** | ALL | 2024-03 | 20 | 2.66 | **+4.61** |

---

## 8. Pharmacovigilance Limitations

1. **Under-Reporting & Reporting Biases**:  
   Spontaneous reporting systems capture only a fraction of real-world adverse events. Severe, unexpected, and novel reactions are reported at significantly higher rates than mild or well-known events (**Weber Effect**).
2. **Publicity & Media Bias**:  
   Regulatory alerts, news media coverage, or litigation can induce artificial surges in report volume unrelated to actual drug risk.
3. **Confounding by Indication**:  
   A disease symptom can be mistakenly flagged as an adverse event of the drug used to treat that very condition (e.g. *rheumatoid arthritis* reported as a reaction for anti-arthritic biologics).
4. **Absence of Denominator Data**:  
   FAERS contains total reported adverse cases, but does not track the total number of patients prescribed the drug. True epidemiological incidence rates cannot be computed from spontaneous reports alone.
5. **Association vs. Causation**:  
   Statistical disproportionality ($PRR \ge 2$, $\chi^2 \ge 4$) indicates higher-than-expected reporting frequency, but **does not establish biological or clinical causality**. Signals serve strictly to trigger formal clinical audits, pharmacoepidemiologic cohort studies, and targeted prospective registries.

---

## 👥 Contributors & Acknowledgements
- Developed using OpenFDA public data endpoints.
- MedDRA terminology utilized under standard FAERS pharmacovigilance conventions.

