---
layout: post
title: "Is RAG Dead? What My Own Numbers Say"
description: "A controlled test of RAG vs. long-context on the same corpus and the same model — accuracy, retrieval quality, latency, and real dollar cost, measured rather than assumed."
redirect_from:
  - /2026/09/01/is-rag-dead/
---

<div class="video-embed">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/uj6IrJib1fs" title="Is RAG Dead? What My Own Numbers Say" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

*Companion post to the video above. This is the deeper reference version — the methodology, the exact numbers, and the sources the video didn't have time for. If you just want the verdict, watch the video first; come back here for the receipts. Full notebook and code: [github.com/kabirrajsingh/notebooks](https://github.com/kabirrajsingh/notebooks/tree/master/is-rag-dead).*

## The promise from the last video

Building a "production-ready" RAG pipeline is only half the argument these days. The other half — the one people actually bring up in the comments — is whether you needed RAG at all. Context windows on frontier models now run into the hundreds of thousands, sometimes millions, of tokens. If the model can just read everything, why build a retrieval pipeline?

That's a fair question, and it deserves a tested answer instead of a hot take. So this post is the test: the same corpus, the same model generating the final answer, two different ways of getting information in front of it — chunk-and-retrieve versus dump-everything-in — measured on accuracy, retrieval quality, latency, and real dollar cost.

## What was actually tested

The corpus: a set of Wikipedia articles on space exploration missions — Apollo 11, Apollo 13, the ISS, Voyager 1/2, Mars rovers, Hubble, JWST, SpaceX — chosen for genuine topical overlap between articles, since overlapping content is what makes retrieval a real test rather than a trivial one. 12 articles, 509 chunks, 24 questions generated from the corpus itself with known reference answers.

Two paths, same model for the final answer generation in both:

- **RAG**: chunk the corpus (1500 characters, 200-character overlap — a plain baseline, not a tuned one), embed each chunk, embed the question, retrieve the top 5 chunks by cosine similarity, answer from those chunks only.
- **Long-context**: skip retrieval entirely. Send the whole corpus as context on every single query, and let the model find what it needs.

![RAG pipeline versus long-context pipeline: RAG chunks, embeds, and retrieves a targeted slice; long-context sends the entire corpus as context on every query with no retrieval step. Same corpus, same model generates the final answer in both paths — the only difference is whether the model reads a targeted slice or everything.](/assets/images/rag-vs-longcontext-architecture.png)

The two prompts sent to the model make the difference concrete. RAG's prompt only ever contains the top-5 retrieved chunks:

```python
def answer_with_rag(question):
    retrieved = retrieve(question)  # top-5 chunks by cosine similarity
    context = "\n\n---\n\n".join(f'[{c["article_title"]}]\n{c["text"]}' for c in retrieved)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    return client.models.generate_content(model=GEN_MODEL, contents=prompt)
```

Long-context's prompt is the same template, but `context` is the *entire corpus*, built once and reused on every single question:

```python
FULL_CONTEXT = "\n\n===\n\n".join(f'[{a["title"]}]\n{a["text"]}' for a in corpus)

def answer_with_long_context(question):
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"CONTEXT:\n{FULL_CONTEXT}\n\nQUESTION: {question}\nANSWER:"
    )
    return client.models.generate_content(model=GEN_MODEL, contents=prompt)
```

Same model, same instruction template, same question — the only variable is how much of the corpus is in `CONTEXT`. That's what "isolating retrieval strategy as the only difference" actually looks like in code, not just in prose.

Scoring used an LLM-as-judge (comparing each generated answer against the reference answer), which is convenient but not infallible — a sample of the judged answers was cross-checked by hand before trusting the aggregate numbers below.

One honest methodology note: retrieval was tracked at two levels, not one. "Was the right *article* retrieved" is a weaker claim than "was the exact *chunk* containing the answer retrieved" — a model can pull the correct article and still miss the specific paragraph the answer lives in. Collapsing those into one number hides exactly the kind of failure this whole question is about.

## Finding 1: the model really is capable

Long-context answered 96% of questions correctly (23/24). That's the part where the skeptics have a point — modern long-context models are genuinely good at pulling a specific fact out of a large pile of text. This corpus ran to about 145K tokens per query, which sounds like a lot until you check it against the model's actual window: `gemini-3.1-flash-lite` supports up to **1,000,000 input tokens**. This test used roughly **14.5%** of that window. Nothing here came close to stress-testing where these models actually start to struggle.

RAG, once the retrieval pipeline was actually working correctly, answered 88% correctly (21/24) — close enough to long-context that accuracy alone doesn't settle the argument. That's the real headline: at this scale, on single-fact questions, both approaches basically work. If your mental model of RAG is "necessary because the model can't otherwise find the answer," that mental model is out of date.

## Finding 2: capable isn't the same as free

Here's where the two paths actually diverge. Real cost, pulled from the API's own reported token usage, not estimated from character counts:

```python
def usage_from_response(resp, fallback_prompt_text=""):
    """Pull REAL token counts from the API response instead of guessing."""
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None:
        return {
            "input_tokens": meta.prompt_token_count or est_tokens(fallback_prompt_text),
            "output_tokens": meta.candidates_token_count or 0,
        }
    return {"input_tokens": est_tokens(fallback_prompt_text), "output_tokens": 0}

PRICING_PER_M = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},  # USD per 1M tokens
    # ...re-verify at ai.google.dev/pricing before publishing — these move
}

def call_cost_usd(model, input_tokens, output_tokens):
    p = PRICING_PER_M[model]
    return (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]
```

Every cost figure below comes from that function, applied to the token counts each API call actually reported — not a guess.

| | RAG | Long-context |
|---|---|---|
| Answer accuracy | 88% (21/24) | 96% (23/24) |
| Correct article retrieved | 100% | n/a (whole corpus is "retrieved" every time) |
| Correct exact chunk retrieved | 87% | n/a |
| Avg. latency | 2.5s | 3.0s |
| Avg. cost per query | $0.00049 | $0.03639 |

![Bar chart comparing RAG and long-context on accuracy, latency, and average cost per query — the accuracy and latency gaps are small, the cost gap is not.](/assets/images/recall-comparison.png)

The cost gap is real and it's structural, not incidental: long-context resends the *entire corpus* as input on every single query, while RAG sends only the handful of chunks that are actually relevant. That difference compounds with every query you run — it's not a fixed cost, it scales with usage. At this test's scale, long-context averaged **75x** more per query than RAG.

Latency is the number worth being honest about, because it didn't behave the way a lot of the "is RAG dead" discourse online claims. Some widely-cited comparisons put long-context latency at 20-60+ seconds against roughly 1 second for RAG. This test's measured gap was much smaller — 2.5s vs 3.0s, a 0.5s difference. Don't inherit someone else's number when you've measured your own; report what you actually saw.

## Finding 3: retrieval succeeding isn't the same as the answer being right

The chunk-level tracking mentioned above exists specifically to catch this. Each generated question is matched back to the one chunk that actually contains its answer:

```python
def find_source_chunk(article_title, reference_answer):
    """Locate which chunk of the article actually contains the reference answer text.
    'The right article was retrieved' and 'the paragraph with the actual fact was
    retrieved' are different claims — conflating them hides where a failure happened."""
    candidates = chunks_by_article.get(article_title, [])
    needle = reference_answer.strip().rstrip(".").strip()
    for c in candidates:
        if needle and needle.lower() in c["text"].lower():
            return c["chunk_id"]
    return None  # couldn't localize it exactly — falls back to article-level tracking only
```

Then, per question: `rag_retrieved_exact_chunk = source_chunk_id in rag["retrieved_chunk_ids"]`. That's the flag that separates "RAG retrieved the exact chunk containing the answer, and still got it wrong" from "RAG never had the answer in front of it to begin with." The first is a generation failure — the right information was in the prompt — the second is a retrieval failure.

One case in this run: a question asking for a specific dollar amount (reference answer: $86 billion). RAG retrieved the exact chunk containing that figure — `rag_retrieved_exact_chunk` was `True` — but the model's answer hedged, saying the provided context contained "conflicting information," rather than committing to the number that was right there in front of it. That's not a retrieval failure. The retriever did its job; the reader didn't trust what it found.

This matters for how you think about debugging a RAG system in production: a wrong answer doesn't automatically mean "fix the retriever." Sometimes the retriever did its job and the reader dropped the ball.

## Where this test can't speak — and what the literature actually says

Everything above is this test's own data, at this test's own scale. There are structural arguments for RAG that this experiment is too small to demonstrate directly, but that are backed by actual research rather than just practitioner intuition. Worth naming explicitly rather than blurring into "and also RAG is good for X" without a source.

**Position within the context matters, independent of whether the fact is technically present.** Liu et al., ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) (2023), found that model accuracy on long-context tasks follows a U-shaped curve — performance is strongest when the relevant information sits at the very start or end of the context, and measurably worse when it's buried in the middle, even though the information is technically within the context window the whole time. This test's corpus (~145K tokens) didn't push far enough to reliably surface this effect — the paper's findings were on larger and more adversarial context arrangements than what was tested here. It's cited as literature context for why "the fact is in the window" doesn't guarantee "the model will use it," not as something this experiment independently confirmed.

![Illustrative accuracy-vs-position curve based on Liu et al.'s "Lost in the Middle" findings — not measured in this experiment.](/assets/images/lost-in-the-middle-illustrative.png)

**There's no universal winner, and it depends on more than accuracy.** Li et al., ["LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs — No Silver Bullet for LC or RAG Routing"](https://arxiv.org/abs/2502.09977) (2025), ran a much larger and more rigorous version of this same question — 11 LLMs, 2,326 test cases, four QA task types, three long-context types — and found that which approach wins depends on the model, the context length, the task type, and the retrieval characteristics involved. Their conclusion tracks with this post's own finding at a smaller scale: neither approach is categorically better, and the deciding factors (in their case and in this test) shift toward things other than raw accuracy once both approaches are "good enough."

Two structural points that follow logically from how each approach works, rather than from a specific paper, worth naming as reasoning rather than dressing up as findings: a corpus larger than the model's context window makes long-context not just expensive but literally impossible — RAG doesn't have that ceiling. And updating one document means re-sending the entire corpus on every future query with long-context, versus re-indexing just that one document with RAG.

## The verdict

RAG isn't dead. It stopped being an accuracy problem and became a cost-and-scale problem. Modern long-context models are legitimately capable readers now — that's not a myth, this test's own numbers back it up. But "capable" was never the same claim as "affordable" or "scales." RAG's job shifted from compensating for a weak reader to keeping a strong, expensive reader affordable and viable as usage and corpus size grow. That's a narrower claim than either "RAG is obsolete" or "RAG is unquestionably still necessary" — and it's the one the data actually supports.

## Code

The full notebook — corpus fetching, chunking, embeddings, both answer paths, scoring, and every chart above — is on GitHub: [github.com/kabirrajsingh/notebooks](https://github.com/kabirrajsingh/notebooks/tree/master/is-rag-dead).

## References

- Liu, N. F., Lin, K., Hewitt, J., et al. — [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (arXiv:2307.03172, 2023)
- Li, K., et al. — [LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs — No Silver Bullet for LC or RAG Routing](https://arxiv.org/abs/2502.09977) (arXiv:2502.09977, 2025)
- Google — [Gemini API pricing](https://ai.google.dev/pricing)
