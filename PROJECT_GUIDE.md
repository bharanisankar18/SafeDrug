# SafeDose: Beginner's Guide to the Project

This guide explains the whole project in simple words: what it does, which file does what, which tools are used and why, and how to answer common questions from faculty.

---

## 1. The project in one paragraph

When patients take a medicine and something bad happens (a rash, a heart problem, a hospital visit), doctors, patients and companies can send a report to the US FDA. These are called **adverse event reports**. The FDA publishes millions of them for free through **openFDA**.

SafeDose takes these reports (1.7 GB of raw data), **cleans** them with **Apache Spark and Hadoop**, then uses statistics to find **safety signals**: a drug and a reaction that appear together **more often than expected**. The results are shown in an interactive **dashboard** built with Streamlit.

> A signal does **not** prove the drug caused the reaction. It only says "this pair is reported unusually often, a doctor should look at it". This is called *association, not causation*.

---

## 2. The big picture (the workflow)

```
openFDA JSON files (1.7 GB)
        │
        ▼
Hadoop FileSystem  ──  opens and reads the files from disk
        │
        ▼
Apache Spark       ──  reads the JSON, cleans it, removes duplicates
        │
        ▼
Parquet files      ──  small, fast, compressed tables (about 8 MB)
        │
        ├──► Signal scoring      (PRR, ROR, chi-square)
        ├──► Drug interactions   (FP-Growth machine learning)
        ├──► Trends & spikes     (Z-score per month)
        └──► MapReduce counts    (classic Hadoop counting)
                    │
                    ▼
        Streamlit dashboard (what you see in the browser)
```

The **Architecture** page in the dashboard shows this same flow as an **animated diagram**. Press **Play walkthrough** and it explains each stage one by one, using the real numbers from your data.

---

## 3. The dataset

| Item | Value |
| --- | --- |
| Source | openFDA drug adverse event data (FDA FAERS) |
| Files | 3 JSON files in `data/raw/` |
| Raw size | about 1.7 GB |
| Unique reports after cleaning | 35,773 |
| Clean rows (one row = report + drug + reaction) | 377,130 |
| Main months | January to March 2024 |

Each report contains:
- **patient**: sex, age
- **drug** list: every medicine the patient took
- **reaction** list: every side effect reported (written as **MedDRA terms**, a standard medical dictionary)
- **serious**: was it serious (death, hospital, disability...) or not

---

## 4. The tools, and why each one is used

| Tool | What it is (simple words) | Why we use it |
| --- | --- | --- |
| **Python** | The programming language | Everything is written in Python |
| **Apache Hadoop** | A system for storing and reading very large files | Spark uses Hadoop's FileSystem to open the files and write results safely |
| **winutils.exe / hadoop.dll** | Small Windows helper files for Hadoop | Hadoop was built for Linux; these let it run on Windows (in the `hadoop/bin` folder) |
| **Apache Spark (PySpark)** | A big data engine that splits work across many CPU cores | Cleaning 1.7 GB of nested JSON is too heavy for normal pandas |
| **Spark SQL / DataFrames** | Tables inside Spark, like Excel but huge | Used for filtering, grouping, joining |
| **Spark MLlib (FP-Growth)** | Spark's machine learning library | Finds drugs that are often reported together |
| **Parquet** | A compressed, column-based file format | 1.7 GB of JSON becomes about 8 MB, and it loads very fast |
| **pandas** | Python table library | The dashboard reads the Parquet tables with it |
| **Streamlit** | Turns a Python script into a web app | Builds the dashboard pages |
| **Plotly** | Interactive charts | All charts (hover, zoom) |
| **NetworkX** | Graph library | Positions the dots in the Interaction Network |

### Hadoop vs Spark (a common question)
- **Hadoop** = storage and file access (and the original MapReduce idea).
- **Spark** = the fast processing engine. It runs **on top of** Hadoop's file system.
- In this project Spark runs in **local mode** `local[4]`, which means **4 CPU cores on one computer act like 4 workers in a cluster**. The same code could run on a real cluster using HDFS.

---

## 5. Every file, what it does, and its key code

### `src/spark_init.py` — sets up Spark and Hadoop
- Points `HADOOP_HOME` to the project's `hadoop` folder (for winutils).
- Finds Java (Spark needs Java to run).
- Creates the Spark session with 4 cores.

```python
SparkSession.builder.master("local[4]").config("spark.driver.memory", memory)
```

### `src/02_download.py` — Step 1: import the dataset
Downloads the openFDA JSON files into `data/raw/` and checks they are readable.

### `src/03_clean.py` — Step 2: preprocessing (cleaning)
This is the **preprocessing stage**. In order:

1. **Read** the multi-line JSON files with Spark.
2. **Explode `results`**: each file holds a big list of reports, and this turns the list into one row per report.
3. **Remove duplicate reports** using `safetyreportid`.
4. **Explode drugs** and keep only the **primary suspect** drug (`drugcharacterization == "1"`), the drug the reporter thinks caused the problem.
5. **Standardise drug names**: use the generic name if available, lowercase and trim spaces, so "Humira" and "adalimumab" are not counted separately.
6. **Explode reactions** to get one row per reaction.
7. **Select the useful columns**: id, date, seriousness, sex, age, drug, reaction.
8. **Drop empty rows and duplicates.**
9. **Save** as Parquet in `data/processed/clean_reports.parquet`.

```python
df = df.dropDuplicates(["safetyreportid"])
df = df.withColumn("drug", explode(col("patient.drug")))
df = df.filter(col("drug.drugcharacterization") == "1")
df = df.withColumn("reaction", explode(col("patient.reaction")))
clean_df.coalesce(4).write.mode("overwrite").parquet(output_path)
```

> **Result:** 35,773 reports become 377,130 clean rows.

### `src/04_signals.py` — Step 3: find safety signals
For each drug and reaction pair it builds a small 2×2 table:

|  | this reaction | other reactions |
| --- | --- | --- |
| this drug | **a** | b |
| all other drugs | c | d |

Then it calculates:
- **PRR** (Proportional Reporting Ratio) = (a / (a+b)) ÷ (c / (c+d)). It answers "how many times more often is this reaction reported with this drug than with other drugs?"
- **ROR** (Reporting Odds Ratio) = (a × d) ÷ (b × c), with a 95% confidence interval.
- **Chi-square** (with Yates correction) checks whether the difference is real or just chance.

**Evans criteria** (the rule for calling something a signal):
**PRR ≥ 2, chi-square ≥ 4, and at least 3 reports.**

> **Result:** 158,414 pairs scored, 27,860 signals.

### `src/05_interactions.py` — Step 4: drugs reported together
- Treats each report like a **shopping basket** of drugs.
- Runs **FP-Growth** (`minSupport = 0.003`) to find pairs that appear together often.
- Flags a pair as **high risk** if it has 1.2 or more times the reactions of either drug alone, or a serious rate of 70% or more.

> **Result:** 561 pairs, 560 flagged high risk. The rule is very loose, so treat the list as a starting point.

### `src/06_trends.py` — Step 5: monthly trends and spikes
- Counts reports per drug per month.
- Computes each series' **mean** and **standard deviation** with a Spark window function.
- **Z-score** = (this month − mean) ÷ standard deviation.
- A **spike** is Z ≥ 3 with at least 3 reports.

> **Result:** 179 spikes.

### `src/07_hadoop_showcase.py` — Step 6: classic Hadoop MapReduce
- Uses the **Hadoop FileSystem API** directly to read file size, block size and replication.
- Counts drugs the **MapReduce** way:
  - **Map**: turn each row into `(drug, 1)`
  - **Shuffle**: group the same drugs together
  - **Reduce**: add up the 1s

```python
rdd.map(lambda row: (row["drug_name"], 1)).reduceByKey(lambda a, b: a + b)
```

### `src/app.py` — the dashboard
Reads the 5 Parquet tables and shows 7 pages:

| Page | What it shows |
| --- | --- |
| **Overview** | Headline numbers, reporting pulse for the latest month, **"What the data shows"** findings, charts |
| **Drug Explorer** | Pick one drug and see its reactions, age, the drugs it's reported with, and its signals |
| **Safety Signals** | Filter all signals, signal map (scatter plot), download CSV |
| **Compare Drugs** | Two drugs side by side |
| **Trends & Spikes** | Monthly line chart with the spike line |
| **Interaction Network** | Graph of drugs reported together |
| **Architecture** | Animated workflow, tech stack, tables produced |

### `src/assets/architecture_flow.html` — the animated diagram
The moving workflow on the Architecture page. It is plain HTML, SVG and JavaScript, and `app.py` fills it with live numbers.

### `.streamlit/config.toml` — dashboard theme
Dark background and green accent colours.

---

## 6. What we did in this work session

1. **Found why the dashboard wasn't showing.**
   - The Safety Signals page used old column names (`pair_count`, `prr_ci_lower`) that no longer exist in `signals.parquet`. The real names are `report_count` and `ror_lower_ci`.
   - One error stops the whole Streamlit app, so every page after it was missing.
   - The network graph also used a Plotly setting (`titlefont_size`) that the new Plotly version removed.
2. **Fixed both bugs** and tested all 7 pages automatically, with no errors.
3. **Redesigned the dashboard** in a dark theme with green accents, as in your reference screenshots:
   - Top menu bar, with the current page highlighted green
   - Big headline with a "reporting pulse" card
   - Stat strips with coloured status pills and up/down arrows
4. **Added analysis**: the "What the data shows" section on Overview calculates key findings automatically.
5. **Built the animated Architecture page** covering all 10 stages from dataset import to the dashboard.
6. **Kept backups** of the old versions: `src/app_backup_v1.py` (original) and `src/app_backup_v2_light.py` (light theme).

---

## 7. How to run it

```powershell
cd D:\SafeDose

# Only if you want to rebuild the data (needs Java):
python src/03_clean.py
python src/04_signals.py
python src/05_interactions.py
python src/06_trends.py
python src/07_hadoop_showcase.py

# Start the dashboard:
streamlit run src/app.py
```

Then open **http://localhost:8501** in your browser.

The processed tables already exist, so normally you only need the last command.

---

## 8. Questions faculty may ask, with simple answers

**Q: Why use Spark instead of just pandas?**
A: The raw data is 1.7 GB of nested JSON. Spark splits the work across CPU cores and can grow to a cluster of many machines. pandas would struggle with memory on bigger downloads.

**Q: Where is Hadoop used?**
A: Spark reads and writes every file through Hadoop's FileSystem API. Parquet is a Hadoop-ecosystem format. Hadoop's FileOutputCommitter writes results safely, creating the `_SUCCESS` file and `.crc` checksum files. Step 07 also shows classic MapReduce and reads file information through the Hadoop API.

**Q: What is preprocessing in your project?**
A: Step 03. It removes duplicate reports, keeps only the primary suspect drug, standardises drug and reaction names, splits nested lists into rows, removes empty values, and saves the result as Parquet.

**Q: What is a safety signal?**
A: A drug and reaction pair reported more often than expected, using PRR ≥ 2, chi-square ≥ 4 and at least 3 reports (the Evans criteria).

**Q: What does PRR = 23 mean?**
A: The reaction is reported 23 times more often with this drug than with all other drugs.

**Q: Does a signal prove the drug is dangerous?**
A: No. Reports are voluntary and can be biased, for example after news coverage. A signal is a hypothesis that needs clinical study.

**Q: What is FP-Growth?**
A: A machine learning algorithm for finding items that often appear together, like "bread and butter" in shopping. Here the items are drugs in the same report.

**Q: What is a Z-score spike?**
A: A month whose report count is 3 or more standard deviations above that drug's normal month.

**Q: What does `local[4]` mean?**
A: Spark runs on this one computer using 4 CPU cores as 4 parallel workers.

**Q: Why Parquet?**
A: It is compressed and stores data by column, so it is much smaller (about 220 times smaller here) and much faster to read than JSON.

**Q: What is lazy evaluation in Spark?**
A: Spark does not run each step immediately. It builds a plan (a DAG) and only runs it when an action such as `count()` or `write` is called. This lets Spark optimise the whole job.

**Q: What is a shuffle?**
A: When Spark must move rows between workers so that rows with the same key end up together, for example in `dropDuplicates` or `groupBy`. It is the slowest part of a job.

---

## 9. Limitations to mention honestly

- Most reports are from one quarter (January to March 2024), so long-term trends are limited.
- Reports are voluntary: under-reporting, duplicates and media effects are possible.
- Some top "reactions", such as *off label use* or *drug ineffective*, describe how a drug was used, not a side effect.
- The high-risk interaction rule flags 560 of 561 pairs, so it is too loose to rank pairs.
- Spark runs on one machine here, not a real cluster.
