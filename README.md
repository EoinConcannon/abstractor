# Abstractor

Summarise research papers into page-cited bullet points, check the summary's numbers against the paper, and ask follow-up questions about it.

## What it does

1. **Reads the PDF properly.** Handles two-column IEEE layouts, strips running headers, footers and page numbers, and keeps the page number attached to every paragraph.
2. **Splits it into sections** (abstract, I. INTRODUCTION, and so on) and cuts off the references, so you can see what the model will be given before it summarises anything.
3. **Summarises into a fixed structure:** Problem, Approach, Key results, Limitations, Takeaway. Every bullet cites a page.
4. **Verifies the summary.** Every number in it is looked up in the paper, on the pages the bullet cites.
5. **Answers follow-up questions** using hybrid retrieval (keyword and vector search). Answers cite pages, say so when the paper doesn't contain the answer, and warn about numbers that aren't in the retrieved text.

It runs as a command-line tool or as a small Streamlit web app.

## How it works

```
PDF
 └─ extract.py   text per page, columns in reading order, headers/footers removed
     └─ sections.py   split at headings, drop references
         ├─ mapreduce.py / summarize.py   notes per section -> final structured summary
         │     └─ pipeline.py   verify numbers and page citations in the summary
         └─ rag.py   ~200-word chunks (with pages) -> BM25 index + local embeddings
               └─ ask.py   hybrid search -> LLM answers from the retrieved text only
```

Every stage is saved to `data/<paper name>/`, so nothing is recomputed or re-requested unless the input, model or prompt changes.

## Design decisions

| Decision | Why |
|---|---|
| **No LLM framework** (no LangChain or LlamaIndex) | The pipeline is a few hundred lines. Writing it by hand means I understand every step and can debug it. |
| **LLM behind a small `LLMClient`** using the OpenAI-compatible API | Switching provider (Google, Groq, OpenRouter, a local Ollama model) is three lines in `.env`. Free tiers change, so this matters. |
| **Retries with backoff** for 429, 503 and connection errors | Free APIs fail regularly. The client waits and retries instead of crashing. |
| **Map-reduce summaries with resumable notes** | A single prompt skimmed the results section. Each section's notes are saved as they arrive, so a failure halfway resumes instead of restarting. |
| **Local embeddings** (`fastembed`, ONNX, `bge-small`) | No API quota used, and no multi-gigabyte PyTorch install on an 8 GB machine. |
| **Plain numpy file instead of a vector database** | One paper is a few hundred chunks. Brute-force cosine similarity is instant. A database would add weight for no benefit at this scale. |
| **Hybrid retrieval** (BM25 and vectors, merged by reciprocal rank fusion) | Keyword search finds exact tokens such as `0.9089` or `UCOSA`. Vector search handles paraphrases. |
| **Saved results keyed by model and prompt hash** | Changing the prompt or model never silently reuses an old summary, and old versions stay for comparison. |

## Setup

Needs Python 3.10 or newer (developed on 3.14, Windows). Commands below are for Windows Command Prompt.

```
git clone https://github.com/<your-username>/abstractor.git
cd abstractor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/api-keys), then:

```
copy .env.example .env
```

Open `.env` and paste your key after `LLM_API_KEY=`. Put your PDFs in a `papers` folder (it is git-ignored, so papers are never committed).

## Usage

**Web app**

```
streamlit run app.py
```

Upload a PDF or pick one from `papers/`. The tabs show the section list, the summary with its verification counts, and a chat box for questions. There is also an exact-word search.

**Command line**

```
python pipeline.py papers\my_paper.pdf             # extract, sections, summary, verification
python pipeline.py papers\my_paper.pdf --no-llm    # stop after the section list (no API key needed)
python pipeline.py papers\my_paper.pdf --refresh   # redo extraction and sections, keep summaries
python pipeline.py papers\my_paper.pdf --method single   # one-prompt summary instead of map-reduce

python ask.py papers\my_paper.pdf "What dataset was used?"   # one question
python ask.py papers\my_paper.pdf                            # interactive
python ask.py papers\my_paper.pdf "..." --chunks             # show retrieved passages only
python ask.py papers\my_paper.pdf --find blockchain          # exact word search, every page

python eval.py                       # retrieval evaluation
python eval.py --max-words 120       # try a different chunk size
```

The first question about a paper embeds it, which takes a minute or two on an old CPU. After that it is saved.

## Evaluation

`eval.py` scores retrieval against 15 questions with known answer pages, taken from two IEEE Access papers (not included, for copyright reasons). A retrieved chunk counts as correct only if it is on a correct page and contains the key fact.

Results of the first run, as hit@6 (a correct chunk is in the top 6) and mean reciprocal rank:

| Chunk size | BM25 only | Vector only | Hybrid |
|---|---|---|---|
| 120 words | 73% / 0.69 | 73% / 0.57 | 73% / 0.65 |
| 200 words | 73% / 0.62 | 67% / 0.50 | 67% / 0.56 |
| 300 words | 80% / 0.63 | 67% / 0.52 | 80% / 0.66 |

What I took from it:

- **Keyword search was at least as good as vector search on this test.** Hybrid only edged ahead of BM25 at 300 words. I would not read much into that. The questions reuse the paper's wording, which favours keyword matching, and the embedding model is a small one.
- **The test is small.** With 15 questions, one question is about 7 percentage points, so the chunk-size differences are within noise.
- **Some "misses" were mistakes in my test, not in the system.** Three questions failed in every configuration. Checking them by hand showed two had gold pages that were too narrow or wrong, and one was a real weakness: **short acronym definitions** (such as "UCOSA stands for...") get buried in long chunks that merely mention the acronym.
- After this run I corrected those three questions and added light stemming and a stopword list to the keyword search. I have not re-run the numbers above.

## Summary quality

Three versions of the summary for the same paper, compared by reading each against the paper:

| Version | Where evidence is cited | Limitations section | Main weakness |
|---|---|---|---|
| One prompt | Mostly the abstract | Reasonable | Skims the experiments section |
| Map-reduce, first prompt | Results pages | Only future work | Lost the real limitations |
| Map-reduce, tuned prompt | Results pages | Real limitations, future work labelled | Some compression artefacts (a leaked LaTeX fragment, one garbled bullet) |

The verification step checks that every number in a summary appears in the paper, on a page the bullet cites. It catches invented figures and wrong page citations. **It cannot tell whether a number is attached to the right claim**, so the summary still has to be read against the paper.

## Known limitations

- **Scanned PDFs are not supported** (no OCR). The app warns when very little text is found.
- **Tables and equations extract as noise**, and algorithm boxes come out with line-number prefixes. The summariser is told to say so rather than guess.
- **Hyphenated compound words lose their hyphen** when they break at a line end ("relevance-greedy" becomes "relevancegreedy"), because rejoining split words and keeping real hyphens can't be told apart without a dictionary.
- **Section detection expects IEEE-style headings** (Roman numerals and capitals, or "Abstract—"). It handles drop caps and headings that wrap onto two lines. I have tested it on two IEEE Access papers only.
- **Retrieval can't answer whole-document questions** ("does the paper mention X?", "how many times?"). A "no" would only mean "not in the retrieved passages". Use `--find` or the exact-word search for those.
- **LLM output varies between runs**, and free-tier models fail or rate-limit regularly. Retries and saved partial results soften this, but a run can still take a few minutes.
- **Privacy:** Google may use free-tier prompts to improve its products. Use public papers only, or switch `.env` to a provider or local model whose terms suit you.
- Chat history in the web app is lost when the browser tab closes.

## Troubleshooting

**404 or "model not found".** Not every model that appears in Google's list works on the free tier, and names change. List what your key can see:

```
python -c "from llm import LLMClient; print('\n'.join(sorted(m.id for m in LLMClient(model='x')._client.models.list())))"
```

Pick a `flash` or `flash-lite` model (drop the `models/` prefix) and set it as `LLM_MODEL` in `.env`.

**503 or timeouts.** Google's servers are overloaded. The client retries for a few minutes. If it still fails, run the same command again: finished sections are saved and are skipped.

**`DLL load failed` when embedding.** Install the Microsoft Visual C++ Redistributable.

**No `fastembed`.** Everything still works, using keyword search only (`pip install numpy rank_bm25` is the minimum).

## Project layout

```
app.py           Streamlit web app
pipeline.py      full pipeline, saved stages, summary verification
extract.py       PDF text extraction (columns, headers/footers, ligatures)
sections.py      heading detection, reference cut-off
summarize.py     one-prompt summary and shared prompt rules
mapreduce.py     section-by-section summary with resumable notes
rag.py           chunking, embeddings, BM25, hybrid search
ask.py           question answering, exact-word search
eval.py          retrieval evaluation
llm.py           provider-agnostic LLM client with retries
eval/questions.json   evaluation questions
.env.example     settings template
```

`papers/`, `data/`, `models/` and `.env` are git-ignored.

## What I would do next

- Automated tests and a CI workflow.
- Re-run the evaluation after the question fixes, with more questions written in my own words rather than the paper's.
- Clean up page-1 licence footnotes and equation debris that leak into chunks.
- Deploy a demo where visitors supply their own API key.
