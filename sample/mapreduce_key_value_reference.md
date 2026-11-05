# SafeDose – MapReduce Key-Value Reference

Reference for the MapReduce job in `src/07_hadoop_showcase.py`. All values below are taken from the real dataset `data/processed/clean_reports.parquet`.

- Total input records: **377,130**
- Unique drug keys: **3,127**
- Unique reaction keys: **5,959**

---
## 1. Input records
Each row is one (drug, reaction) pair from an FDA adverse event report.

| safetyreportid | drug_name | reaction_name |
|---|---|---|
| 10236864 | xeljanz | condition aggravated |
| 10236864 | xeljanz | covid-19 |
| 10236864 | xeljanz | drug ineffective |
| 10236864 | xeljanz | off label use |
| 10236864 | xeljanz | psoriatic arthropathy |
| 10236864 | xeljanz | spinal operation |
| 10236864 | xeljanz | staphylococcal infection |
| 12134121 | alcohol | drug abuse |

---
## 2. Map phase
Mapper emits `(key, 1)` for every record.

**Drug job** – `rdd.map(lambda row: (row["drug_name"], 1))`
```
('xeljanz', 1)
('xeljanz', 1)
('xeljanz', 1)
('xeljanz', 1)
('xeljanz', 1)
('xeljanz', 1)
('xeljanz', 1)
('alcohol', 1)
```
**Reaction job** – `rdd.map(lambda row: (row["reaction_name"], 1))`
```
('condition aggravated', 1)
('covid-19', 1)
('drug ineffective', 1)
('off label use', 1)
('psoriatic arthropathy', 1)
('spinal operation', 1)
('staphylococcal infection', 1)
('drug abuse', 1)
```
---
## 3. Shuffle & sort phase
All values for the same key are grouped together (shown for the sample records above).
```
'xeljanz'                    -> [1, 1, 1, 1, 1, 1, 1]
'alcohol'                    -> [1]
```
---
## 4. Reduce phase
Reducer sums the values per key – `reduceByKey(lambda a, b: a + b)`.

### 4a. Drug counts (top 25)  →  saved to `data/processed/hadoop_mr_counts/`

| # | Key (drug_name) | Value (event_count) |
|---|---|---|
| 1 | repatha | 5,540 |
| 2 | adalimumab | 5,061 |
| 3 | rituximab | 5,001 |
| 4 | methotrexate | 4,823 |
| 5 | prednisone | 4,505 |
| 6 | cosentyx | 4,199 |
| 7 | tirzepatide | 4,185 |
| 8 | acetaminophen | 3,574 |
| 9 | etanercept | 3,569 |
| 10 | tocilizumab | 3,546 |
| 11 | infliximab | 3,464 |
| 12 | dupilumab | 3,022 |
| 13 | leflunomide | 3,021 |
| 14 | certolizumab pegol | 3,005 |
| 15 | abatacept | 2,941 |
| 16 | simponi | 2,836 |
| 17 | alendronate sodium | 2,547 |
| 18 | folic acid | 2,496 |
| 19 | xeljanz | 2,441 |
| 20 | omalizumab | 2,387 |
| 21 | copper | 2,358 |
| 22 | ustekinumab | 2,347 |
| 23 | sulfasalazine | 2,340 |
| 24 | desoximetasone | 2,275 |
| 25 | diclofenac sodium | 2,266 |

### 4b. Reaction counts (top 25)  →  printed only, not saved

| # | Key (reaction_name) | Value (event_count) |
|---|---|---|
| 1 | off label use | 6,357 |
| 2 | drug ineffective | 5,963 |
| 3 | fatigue | 4,167 |
| 4 | nausea | 3,920 |
| 5 | dyspnoea | 3,648 |
| 6 | headache | 3,477 |
| 7 | diarrhoea | 3,405 |
| 8 | pain | 3,261 |
| 9 | rash | 2,826 |
| 10 | pyrexia | 2,744 |
| 11 | arthralgia | 2,739 |
| 12 | vomiting | 2,669 |
| 13 | malaise | 2,510 |
| 14 | condition aggravated | 2,354 |
| 15 | pruritus | 2,301 |
| 16 | hypertension | 2,128 |
| 17 | insomnia | 2,056 |
| 18 | asthenia | 2,049 |
| 19 | dizziness | 1,988 |
| 20 | hypersensitivity | 1,978 |
| 21 | abdominal discomfort | 1,954 |
| 22 | death | 1,949 |
| 23 | pneumonia | 1,877 |
| 24 | rheumatoid arthritis | 1,847 |
| 25 | drug hypersensitivity | 1,834 |

---
## 5. Raw Hadoop-style output (`part-00000`, tab-separated)
```
repatha	5540
adalimumab	5061
rituximab	5001
methotrexate	4823
prednisone	4505
cosentyx	4199
tirzepatide	4185
acetaminophen	3574
etanercept	3569
tocilizumab	3546
```
---
## 6. Output schema
| Column | Type | Meaning |
|---|---|---|
| item_name | string | The key (drug name) |
| event_count | long | The value (number of reports) |

View the full output:
```
python -c "import pandas as pd; print(pd.read_parquet('data/processed/hadoop_mr_counts').sort_values('event_count', ascending=False).to_string(index=False))"
```
