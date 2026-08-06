# Data notice

## What is covered by what

| Material | Terms |
|---|---|
| Python scripts, README, docs, annotation guidelines and coding scheme | MIT — see [LICENSE](LICENSE) |
| Manual annotations (the 0/1 labels, `seniority`, `role_family`, notes) | Original research output of this project, MIT |
| Job-posting content — `job_title`, `company`, `job_location`, `job_link`, `job_summary` | **Not licensed by this repository.** See below. |

## The posting data

The corpus derives from a publicly available dataset of LinkedIn job postings. It was
filtered to UK analyst-adjacent roles by `filter_corpus.py`; the raw source files
(`linkedin_job_postings.csv`, `job_summary.csv`) are not included here.

The posting text and metadata are third-party content. This repository does not own
them, does not claim any rights in them, and does not grant you a licence to them.
They are included so that the research is reproducible and the reported metrics can
be checked against the same inputs.

If you reuse this repository:

- The code, the annotation scheme and the annotations themselves are yours to use
  under MIT.
- For the posting data, obtain it from the original source under its own terms rather
  than relying on this copy.
- Postings carry company names and original `job_link` URLs. Treat them as you would
  any scraped web content — in particular, don't redistribute the corpus onward as
  though it were an open dataset.

## Removal requests

If you are a rights holder and want specific postings removed, open an issue and they
will be taken out.
