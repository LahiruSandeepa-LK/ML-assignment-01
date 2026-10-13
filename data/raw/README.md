# Raw data

`data.csv` is not committed. Notebook 01 downloads it on first run.

## Source

UCI Machine Learning Repository dataset 697, *Predict Students' Dropout and Academic
Success*. CC BY 4.0.

- <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- <https://doi.org/10.24432/C5MC89>

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student
Dropout and Academic Success. *Data, 7*(11), 146. <https://doi.org/10.3390/data7110146>

## Retrieval

`load_dropout_data()` in notebook 01 reads `data/raw/data.csv`, or fetches and caches it
if the file is absent:

```python
uci = fetch_ucirepo(id=697)
df = pd.concat([uci.data.features, uci.data.targets], axis=1)
df.to_csv("data/raw/data.csv", sep=";", index=False)
```

The cache is semicolon-delimited, matching the original publication.

## Integrity checks

Notebook 01 asserts the following on load.

| check | expected |
|---|---|
| shape | 4,424 x 37 |
| missing cells | 0 |
| duplicate rows | 0 |
| `Target` | Graduate 2,209 / Dropout 1,421 / Enrolled 794 |
| 1st vs 2nd sem (credited) | 0.9448 |
| 1st vs 2nd sem (enrolled) | 0.9426 |
| 1st vs 2nd sem (approved) | 0.9040 |
| 1st vs 2nd sem (grade) | 0.8372 |
| 1st vs 2nd sem (evaluations) | 0.7789 |
| pairs above \|0.7\| | 13 |

## Known discrepancies with the descriptor paper

The nationality column is spelled `Nacionality` in the source file.

Pearson correlations for the nine integer-coded nominal columns differ from the
descriptor paper, because UCI publishes the detailed national registry codes while the
paper's table was computed on a coarser grouping. `Nacionality` against `International`
is 0.7909 here and 0.9117 in the paper; the parents' occupations are 0.9105 and 0.7240.
Notebook 01 section 8.2 covers this.
