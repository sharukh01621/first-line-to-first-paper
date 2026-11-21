# `sharukh-tutor`

Interactive progressive hint engine, layered diagnostic checker, and curated Public Health / Epidemiology / Biostatistics dataset loader for:

> **From First Line to First Paper: Mastering Python, Statistical Data Science, and Applied Deep Learning**
> *By Mezanur Rahman Sharukh*

## Quickstart (Google Colab or Local)

```python
!pip install -q sharukh-tutor
import sharukh_tutor as tutor

tutor.activate(chapter="ch01", notebook="todos")
tutor.hint("ch01_ex01", level=1)
tutor.check("ch01_ex01", my_answer)
tutor.summary()
```
