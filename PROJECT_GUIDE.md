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

## 6. Exactly what each tool does, with real examples

This section follows **real records from your data** through every tool, in the order the pipeline uses them.

### 6.1 openFDA JSON: what the raw data looks like

One report from `drug-event-0003-of-0029.json`, shortened:

```json
"patient": {
  "patientonsetage": "20",
  "patientsex": "2",
  "reaction": [
    { "reactionmeddrapt": "Complication of device insertion" }
  ],
  "drug": [
    {
      "drugcharacterization": "1",
      "medicinalproduct": "PARAGARD T 380A",
      "openfda": { "generic_name": ["COPPER"] }
    }
  ]
}
```

**What it means:**
- `patientsex: "2"` → female (1 = male, 2 = female)
- `reaction` → a list (one report can have many reactions)
- `drug` → a list (one report can have many drugs)
- `drugcharacterization: "1"` → this is the **suspect** drug (2 = taken at the same time, 3 = interacting)
- `openfda.generic_name` → the FDA's standard name for the medicine

Everything is **nested** (lists inside lists). You cannot put this straight into a table, so cleaning is needed.

---

### 6.2 Python: the glue

Python is the language every script is written in. It does not process the big data itself; it **gives instructions** to Spark.

**Example:** when `03_clean.py` says `df.dropDuplicates(...)`, Python only sends the instruction. The real work happens inside Spark.

---

### 6.3 Java: the engine room

Spark is written in Scala, which runs on **Java** (the JVM). PySpark uses a bridge called **Py4J** to pass your Python commands to Java.

**Example of what happens when you run `python src/03_clean.py`:**
1. Python starts.
2. `spark_init.py` finds Java (`JAVA_HOME`).
3. A Java process starts in the background. This is the **Spark driver**.
4. Every Spark command in Python is sent to that Java process.

That is why the project needs Java installed, even though you only write Python.

---

### 6.4 Hadoop: opening and saving files

Spark never touches the disk directly. It asks **Hadoop's FileSystem** to do it.

**When reading (example with your files):**

| File | Size |
| --- | --- |
| drug-event-0001-of-0029.json | 707 MB |
| drug-event-0002-of-0029.json | 782 MB |
| drug-event-0003-of-0029.json | 134 MB |

1. Spark asks Hadoop to **list** `data/raw/`, and Hadoop returns the 3 files and their sizes.
2. Hadoop normally cuts big files into **blocks** (32 MB on a local disk), so many workers can read one file in parallel.
3. A multi-line JSON file **cannot be cut** in the middle, because a report would break in half. So Hadoop says "not splittable", and **each file goes to one worker**.

**When writing (example `clean_reports.parquet`):**
1. Spark's 4 workers each write a part file into a hidden `_temporary` folder.
2. When all 4 succeed, Hadoop's **FileOutputCommitter** moves them into the final folder.
3. It writes an empty `_SUCCESS` file, meaning "the job finished properly".
4. Next to each part file it writes a small `.crc` file, a **checksum** used to detect file corruption.

So if your computer crashes halfway, you never get a half-written table.

**winutils.exe and hadoop.dll:** Hadoop expects Linux-style file permissions. On Windows these two files (in `hadoop/bin`) give it that ability. Without them, Spark fails when saving files.

---

### 6.5 Apache Spark: the processing engine

#### a) SparkSession and `local[4]`
```python
SparkSession.builder.master("local[4]")
```
This tells Spark: "use **4 CPU cores** on this computer as **4 workers**". On a real cluster this would be, say, `yarn` with 100 machines, and the code stays the same.

#### b) Tasks and partitions
Spark splits data into **partitions** and gives each one to a worker as a **task**.

**Example:** reading your 3 JSON files gives **3 tasks** (one per file). Core 4 has nothing to read. The 782 MB file takes longest, so it decides how long this stage takes.

#### c) Lazy evaluation
Spark **does not run** each line immediately. It writes a plan first.

```python
df = df.dropDuplicates(["safetyreportid"])   # just added to the plan
df = df.filter(...)                          # just added to the plan
df.count()                                   # NOW Spark runs everything
```
Commands like `count()` and `write` are called **actions**. Everything else is a **transformation**. The plan is called a **DAG** (directed acyclic graph).

#### d) Cleaning one real report, step by step
Report **10236864** (a female patient, aged 52, taking **xeljanz**):

**Step 1: `explode(results)`** turns the file's big list into one row per report.

**Step 2: `dropDuplicates(["safetyreportid"])`** keeps one copy if the same report appears twice.

**Step 3: `explode(patient.drug)`** gives one row per drug in the report.

**Step 4: `filter(drugcharacterization == "1")`** keeps only the suspect drug.

**Step 5: standardise the name.** Use the generic name and make it lowercase:

| Brand name in report | After cleaning |
| --- | --- |
| PARAGARD T 380A | copper |
| HUMIRA | adalimumab |
| ENBREL | etanercept |
| XYREM | sodium oxybate |
| NERLYNX | neratinib |

Without this step, "Humira" and "adalimumab" would be counted as two different drugs.

**Step 6: `explode(patient.reaction)`** gives one row per reaction. Report 10236864 had 7 reactions, so it becomes **7 rows**:

| safetyreportid | sex | age | drug_name | reaction_name |
| --- | --- | --- | --- | --- |
| 10236864 | 2 | 52 | xeljanz | condition aggravated |
| 10236864 | 2 | 52 | xeljanz | covid-19 |
| 10236864 | 2 | 52 | xeljanz | drug ineffective |
| 10236864 | 2 | 52 | xeljanz | off label use |
| 10236864 | 2 | 52 | xeljanz | psoriatic arthropathy |
| 10236864 | 2 | 52 | xeljanz | spinal operation |
| 10236864 | 2 | 52 | xeljanz | staphylococcal infection |

Done for all reports, 35,773 reports become **377,130 rows**.

#### e) Shuffle
`dropDuplicates` and `groupBy` need all rows with the **same key** on the **same worker**. Spark moves rows between workers to make that happen. This is a **shuffle**, and it is the slowest part of a job.

**Example:** to count reports for "xeljanz", every xeljanz row from all 4 workers is sent to one worker, then counted.

---

### 6.6 Parquet + Snappy: saving the clean table

**JSON** stores data row by row, with every column name repeated in every record.
**Parquet** stores data **column by column**:

```
JSON (row by row):            Parquet (column by column):
{"drug":"xeljanz","sex":"2"}  drug: xeljanz, xeljanz, xeljanz ...
{"drug":"xeljanz","sex":"2"}  sex : 2, 2, 2 ...
{"drug":"xeljanz","sex":"2"}
```

Repeated values in a column compress extremely well with **Snappy**.

**Example from your project:** 1.58 GB of raw JSON → **7.4 MB** of Parquet (about 220 times smaller). The dashboard reading only the `drug_name` column skips all the other columns, so it is much faster too.

---

### 6.7 Spark SQL `groupBy` + `join`: signal scoring (`04_signals.py`)

Spark counts:
- reports per **drug** (`groupBy("drug_name")`)
- reports per **reaction** (`groupBy("reaction_name")`)
- reports per **pair** (`groupBy("drug_name", "reaction_name")`)

Then it **joins** them into one 2×2 table per pair.

**Real example: neratinib and diarrhoea**

|  | diarrhoea | other reactions | total |
| --- | --- | --- | --- |
| neratinib | a = 22 | b = 5 | 27 |
| all other drugs | c = 1,246 | d = 34,500 | 35,746 |

**PRR** (how many times more often):
```
PRR = [a / (a+b)] / [c / (c+d)]
    = (22 / 27) / (1,246 / 35,746)
    = 0.815 / 0.0349
    = 23.4
```
In words: diarrhoea is reported **23 times more often** with neratinib than with other drugs.

**ROR** (odds ratio):
```
ROR = (a × d) / (b × c) = (22 × 34,500) / (5 × 1,246) = 121.8
95% range = 46.1 to 322.2
```
The whole range is above 1, so the result is unlikely to be chance.

**Chi-square** (is it chance?):
```
χ² = 457.5   (needs to be ≥ 4)
```

**Evans check:** PRR 23.4 ≥ 2 ✔, χ² 457.5 ≥ 4 ✔, a = 22 ≥ 3 ✔ → **signal** ✔

(Medical note: diarrhoea is a well-known side effect of neratinib, so this shows the method finds real effects.)

Spark does this for all **158,414 pairs at once** and finds **27,860 signals**.

---

### 6.8 Spark MLlib FP-Growth: drugs reported together (`05_interactions.py`)

**Idea:** like a supermarket finding "people who buy bread also buy butter".

1. Make a **basket** per report: the list of its suspect drugs.
   ```
   report A: [prednisone, rituximab, cyclophosphamide]
   report B: [prednisone, rituximab]
   report C: [prednisone, methotrexate]
   ```
2. Keep only reports with **2 or more drugs**. Your data has **7,935** such reports.
3. FP-Growth builds a compact tree of the baskets and finds pairs that appear in at least **0.3%** of them (`minSupport = 0.003`).

**Real result: prednisone + rituximab**
- appear together in **154** reports
- support = 154 ÷ 7,935 = **1.94%** ✔ (above 0.3%)
- reactions per report: **16.6** with both, against **10.0** normally → rate ratio **1.66**
- 1.66 ≥ 1.2 → flagged **high risk**

---

### 6.9 Window functions + Z-score: spikes (`06_trends.py`)

1. `groupBy(drug, month)` counts reports per month.
2. A **window function** calculates, for each drug's own series, its **mean** and **standard deviation**, without collapsing the rows.

```python
w = Window.partitionBy("drug_name", "reaction_name")
mean = avg("report_count").over(w)
std  = stddev("report_count").over(w)
```

**Real example: sodium oxybate (Xyrem), March 2024**
- usual month: mean **1.7** reports
- standard deviation: **2.1**
- March 2024: **16** reports

```
Z = (16 − 1.7) / 2.1 = 6.9
```
6.9 ≥ 3 and 16 ≥ 3, so it is flagged as a **spike**. This is the largest spike in your data.

---

### 6.10 MapReduce: classic Hadoop counting (`07_hadoop_showcase.py`)

**Small example with 5 rows:**
```
Input rows:   repatha, adalimumab, repatha, rituximab, repatha

MAP     → (repatha,1) (adalimumab,1) (repatha,1) (rituximab,1) (repatha,1)
SHUFFLE → repatha: [1,1,1]   adalimumab: [1]   rituximab: [1]
REDUCE  → repatha: 3         adalimumab: 1     rituximab: 1
```

Code:
```python
rdd.map(lambda row: (row["drug_name"], 1)).reduceByKey(lambda a, b: a + b)
```

**Real result:** repatha **5,540**, adalimumab **5,061**, rituximab **5,001** drug–reaction rows. That gives 3,127 drugs counted in total.

This is the original Hadoop idea (from Google, 2004). Spark runs the same pattern, but keeps the data in memory, which makes it much faster.

---

### 6.11 pandas + PyArrow: loading for the dashboard

```python
clean = pd.read_parquet("data/processed/clean_reports.parquet")
```
**PyArrow** reads the 4 Parquet part files and gives **pandas** one table in memory (377,130 rows). Spark is **not** used by the dashboard; the heavy work was already done.

**Example filter in Drug Explorer:**
```python
clean[clean["drug_name"] == "xeljanz"]   # takes milliseconds
```

---

### 6.12 Streamlit: the web app

**How it works:** Streamlit runs `app.py` **from top to bottom every time you click something**.

**Example: you choose "xeljanz" in Drug Explorer**
1. Streamlit reruns `app.py`.
2. `load_tables()` would load the Parquet files, but it has `@st.cache_data`, so Streamlit reuses the copy already in memory.
3. The code filters the table to xeljanz.
4. Streamlit sends the new numbers and charts to your browser.

Without caching, every click would re-read all the files.

---

### 6.13 Plotly: the charts

Python builds a chart description, and Plotly turns it into **JSON** that the browser draws. That is why you can **hover, zoom and drag**.

**Example: the Safety Signals "Signal map"**
- x-axis = log₂(PRR) → neratinib/diarrhoea: log₂(23.4) = **4.5**
- y-axis = log₁₀(χ²) → log₁₀(457.5) = **2.66**
- hover → shows drug, reaction, PRR, ROR

Logarithms are used because PRR ranges from 2 to 35,000, which would not fit on a normal axis.

---

### 6.14 NetworkX: the network picture

NetworkX builds a **graph**: drugs are **nodes**, pairs are **edges**. Then `spring_layout` places them like balls on springs: connected drugs pull together and unconnected ones push apart.

**Example:** prednisone and rituximab are each paired with **47** other drugs, more than any other drug. So they end up **near the centre** as **big dots**.

---

### 6.15 Git + GitHub: saving the project online

| Command | What happens |
| --- | --- |
| `git init` | Makes the folder a Git project (creates a hidden `.git` folder) |
| `.gitignore` | Lists files Git must ignore, such as `data/raw/` (1.7 GB) |
| `git add .` | Selects all changed files for the next save |
| `git commit -m "..."` | Saves a snapshot with a message |
| `git push` | Uploads the snapshots to GitHub |

**Your repo:** https://github.com/bharanisankar18/SafeDrug

---

## 7. What we did in this work session

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

## 8. How to run it

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

## 9. Questions faculty may ask, with simple answers

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

## 10. Limitations to mention honestly

- Most reports are from one quarter (January to March 2024), so long-term trends are limited.
- Reports are voluntary: under-reporting, duplicates and media effects are possible.
- Some top "reactions", such as *off label use* or *drug ineffective*, describe how a drug was used, not a side effect.
- The high-risk interaction rule flags 560 of 561 pairs, so it is too loose to rank pairs.
- Spark runs on one machine here, not a real cluster.
