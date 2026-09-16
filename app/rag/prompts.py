"""
Prompts.

The prompt is where grounding happens. The model gets the CV and the
retrieved postings as text, and is told to reason only from them and to
cite job ids. Anything it says should be traceable to something it was
given.
"""

SYSTEM = """You are a careful careers advisor. You will be given a candidate's CV and a
small set of job postings that a search system found for it. Assess the fit of
each posting using ONLY the text provided. Do not invent requirements, skills,
or experience that are not written in the CV or the posting.

Write in the third person, about "the candidate". Never speak as the candidate.
Distinguish what the CV states from what you infer: a company's city is not
the candidate's city, and a tool used at one job is not a tool listed under
skills unless it is. If something is unclear, say so rather than guess."""


def build_match_prompt(cv_text: str, jobs: list[dict]) -> str:
    postings = "\n\n".join(
        f"[{j['job_id']}] {j['title']} at {j['company']} ({j['location']})\n"
        f"Skills: {j['skills']}\n"
        f"{j['description']}"
        for j in jobs
    )
    return f"""CV:
\"\"\"
{cv_text}
\"\"\"

Job postings, best retrieval score first:

{postings}

For each posting, in the same order, write:

[job id] Title at Company
- Fit: one sentence on why this posting matches the CV, quoting or closely paraphrasing the CV.
- Gaps: the skills or experience the posting asks for that the CV does not show. Say "none evident" if there are none.
- Verdict: strong / possible / weak.

Then one final line: which single posting you would recommend applying to first, and why, in one sentence.
"""
