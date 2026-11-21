# From First Line to First Paper
### *Mastering Python, Statistical Data Science, and Applied Deep Learning*
**Author:** Mezanur Rahman Sharukh — *M.Sc. Statistician | Public Health & Machine Learning Researcher*

A flagship, 1,000+ page zero-to-researcher textbook paired **1-to-1** with interactive **Google Colab** notebooks, automated progressive hints, diagnostic self-checks (`sharukh-tutor`), and realistic Public Health, Epidemiology, and Biostatistics datasets.

---

## Repository Architecture (Single Source of Truth)

```text
first-line-to-first-paper/
├── book/                    # Quarto Book manuscript (PDF + EPUB + Web HTML)
├── master_notebooks/        # AUTHORING SOURCE (solutions + cell markers)
├── colab_drive/             # GENERATED STUDENT + SOLUTION NOTEBOOKS -> GitHub & Google Drive
│   ├── 00_START_HERE.ipynb
│   ├── datasets/
│   ├── part_01_python_foundations/
│   └── solutions/
├── sharukh_tutor/           # Publishable PyPI package powering hints, checks & dataset loading
├── scripts/                 # Zero-drift build, verification, QR generation & sync automation
└── config/                  # Master metadata & Colab URL manifest
```

---

## Core Developer Workflow (Zero-Drift Pipeline)

Every time you author or edit a chapter/notebook, run the automated pipeline from the repository root:

```bash
# 1. Generate realistic Public Health / Epidemiology spine datasets
python3 scripts/generate_spine_datasets.py

# 2. Compile master_notebooks/ -> Student + Solution .ipynb in colab_drive/
python3 scripts/build_notebooks.py

# 3. Generate/update Colab links, colab_manifest.json, and Print PDF QR codes
python3 scripts/inject_colab_links.py

# 4. Run full QA verification (tests all solutions PASS and all student stubs FAIL gracefully)
python3 scripts/verify_all.py

# 5. (When Quarto + LaTeX are installed) Build Print PDF, EPUB, and Web Companion
quarto render book/
```

---

## Connecting GitHub, PyPI, and Google Drive (Option B Dual Mirror)

### 1. Connect to GitHub (Powers Deterministic Colab Links)
1. Update `github_owner` in `config/book_metadata.yml` with your GitHub username.
2. Re-run `python3 scripts/build_notebooks.py && python3 scripts/inject_colab_links.py`.
3. Push this repository to GitHub:
   ```bash
   git remote add origin https://github.com/<YOUR_USERNAME>/first-line-to-first-paper.git
   git push -u origin main
   ```
Every notebook inside `colab_drive/` will immediately open in Google Colab via:
`https://colab.research.google.com/github/<YOUR_USERNAME>/first-line-to-first-paper/blob/main/colab_drive/...`

### 2. Mirror `colab_drive/` to Google Drive
1. Configure `rclone` once: `rclone config` (create a Google Drive remote named `gdrive`).
2. Preview and run the mirror sync:
   ```bash
   python3 scripts/sync_drive.py --dry-run
   python3 scripts/sync_drive.py
   ```
3. Share the `From_First_Line_To_First_Paper` folder in Google Drive as *"Anyone with the link can view"* and paste its folder URL into `config/book_metadata.yml`.

### 3. Publish `sharukh-tutor` to PyPI
```bash
cd sharukh_tutor
python3 -m build
python3 -m twine upload dist/*
```
Once published, Cell 0 in every Google Colab notebook (`!pip install -q sharukh-tutor`) installs in ~2 seconds with zero Google Drive authentication prompts.
