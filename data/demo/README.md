# Synthetic demo data

Everything in this directory is fictional and exists only to demonstrate the
project without exposing a real resume, application history, employer record,
or active job posting.

- `synthetic_resume.json` is the source for the generated PDF resume.
- `synthetic_jobs.csv` contains five fictional AI/Agent job descriptions.
- URLs use the reserved `.invalid` top-level domain and are not application links.

Generate the uploadable PDF with:

```bash
python tools/build_demo_resume.py
```

The generated file is written to `output/pdf/synthetic_demo_resume.pdf` and is
ignored by Git.
