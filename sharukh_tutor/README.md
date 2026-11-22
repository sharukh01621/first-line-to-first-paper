# `sharukh-tutor`

Interactive 3-tier progressive hint engine (**972 hints**), layered diagnostic verification suite (**324 automated checks across Chapters 1–36**), and curated Public Health / Epidemiology / Biostatistics dataset loader for:

> ***From First Line to First Paper: Mastering Python, Statistical Data Science, and Applied Deep Learning***  
> *By Mezanur Rahman Sharukh* (*M.Sc. Statistician | Public Health & Machine Learning Researcher*)

- **Read the Web Book:** https://sharukh01621.github.io/first-line-to-first-paper/
- **Download PDF (860 pp) & EPUB 3:** https://github.com/sharukh01621/first-line-to-first-paper/releases/tag/v1.0.1
- **Launch 182 Google Colab Notebooks:** https://colab.research.google.com/github/sharukh01621/first-line-to-first-paper/blob/main/colab_drive/00_START_HERE.ipynb
- **GitHub Repository:** https://github.com/sharukh01621/first-line-to-first-paper

---

## Installation

```bash
pip install sharukh-tutor
```

Or inside Google Colab:

```python
!pip install -q sharukh-tutor
```

## Quickstart

```python
import sharukh_tutor as tutor

# 1. Activate a chapter session
tutor.activate(chapter="ch01", notebook="guided_todos")

# 2. Request progressive 3-tier hints whenever you get stuck
tutor.hint("ch01_todo_01", level=1)  # Gentle conceptual nudge
tutor.hint("ch01_todo_01", level=2)  # Structural formula / pattern
tutor.hint("ch01_todo_01", level=3)  # Near-complete walkthrough

# 3. Verify your solution with warm diagnostic feedback
tutor.check("ch01_todo_01", my_answer)

# 4. View your live chapter score summary
tutor.summary()

# 5. Load bundled public health spine datasets offline (no Drive auth required!)
print(tutor.list_datasets())
# ['clinic_triage', 'epi_surveillance', 'mch_survey_sample']
df = tutor.load_data("mch_survey_sample", as_pandas=True)
```

## License

Released under the **MIT License**.
