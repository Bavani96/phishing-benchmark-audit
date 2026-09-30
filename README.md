# Phishing Benchmark Audit

Code accompanying:

> Bavani K. *Temporal Confounds and Transfer Failure in Phishing E-mail Benchmarks:
> An Empirical Audit of Two Widely Used Corpora.* Submitted to Discover Computing (Springer).

One script (`audit.py`) regenerates every table and figure in Section 5 of the paper.

## 1. Install
```
pip install -r requirements.txt
```

## 2. Get the data (not redistributed here)
Download the curated phishing e-mail corpora released by Champa, Rabbi and Zibran (2024):
- Figshare: https://figshare.com/articles/dataset/Phishing_Email_11_Curated_Datasets/24952503

Place these two files in `data/`:
- `CEAS_08.csv` - CEAS-2008 challenge corpus (39,154 messages)
- `Nazario_5.csv` - Nazario phishing combined with legitimate mail (3,065 messages)

Required columns: `body`, `label` (1 = phishing/spam, 0 = legitimate).
Optional: `subject`, `date`, `urls`.

## 3. Run
```
python audit.py --ceas data/CEAS_08.csv --nazario data/Nazario_5.csv --out outputs
```
Add `--skip-rf` to skip the random-forest run in Table 4 (the slowest step).

## 4. Outputs (in `outputs/`)
| File | Paper item |
|---|---|
| table1_profile.csv | Table 1 |
| results.json | Section 5.1 year-only AUC; Content-Type-only probe |
| table2_nazario_on_ceas.csv | Table 2 |
| table3_probe.csv | Table 3 (plus F1 change with fragment removed) |
| table4_probe_families.csv | Table 4 |
| table5_transfer.csv | Table 5 |
| table6_dedup.csv | Table 6 |
| table7_domain.csv, health_matches_*.csv | Table 7 and matches for manual inspection |
| fig1-fig5 *.png | Figures 1-5 (300 dpi) |

## 5. Protocol settings
- TF-IDF: sublinear TF, min_df = 2, max 60,000 terms; logistic regression, balanced class weights
- Stratified 70/30 split, random seed 42
- Cross-corpus: train on the whole source corpus, evaluate on the whole target corpus, threshold 0.5
- Near-duplicates: MD5 hash of the first 60 normalised tokens of the body
- Implausible Date header: unparsable, or year outside 1990-2026
- Structural probe: Content-Type line + first 40 body characters

## License
MIT
