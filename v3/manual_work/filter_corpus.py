import pandas as pd

# ToR-aligned filter (v2): data and analyst-adjacent roles per Terms of Reference
include = ("data analyst|business analyst|bi analyst|business intelligence|"
           "insight analyst|insights analyst|marketing analyst|finance analyst|"
           "financial analyst|fp&a|commercial analyst|data scientist|"
           "analytics engineer|reporting analyst|mi analyst|analytics|"
           "management information|web analyst")
exclude = "head of|director|manager|bangkok|dubai"

# --- 1. Filter metadata to UK analyst corpus ---
keep = []
for chunk in pd.read_csv("linkedin_job_postings.csv",
                         usecols=["job_link", "job_title", "search_country",
                                  "job_location", "job_level", "first_seen", "company"],
                         chunksize=200_000):
    uk = chunk["search_country"].str.contains("United Kingdom", case=False, na=False)
    inc = chunk["job_title"].str.contains(include, case=False, na=False)
    exc = chunk["job_title"].str.contains(exclude, case=False, na=False)
    keep.append(chunk[uk & inc & ~exc])

corpus = pd.concat(keep, ignore_index=True)
print("corpus size:", len(corpus))
links = set(corpus["job_link"])

# --- 2. Stream summaries, keep matching rows ---
desc = []
done = 0
for chunk in pd.read_csv("job_summary.csv", chunksize=50_000):
    desc.append(chunk[chunk["job_link"].isin(links)])
    done += len(chunk)
    print(f"processed {done:,} rows, matched {sum(len(d) for d in desc)}", end="\r")

descriptions = pd.concat(desc, ignore_index=True)
print()

# --- 3. Merge and save ---
final = corpus.merge(descriptions, on="job_link", how="inner")
print("final size:", len(final))
final.to_csv("uk_analyst_corpus.csv", index=False)
print("saved uk_analyst_corpus.csv")
