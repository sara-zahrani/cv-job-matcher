"""
Quality gate.

Every job posting passes through here before it is stored. A posting either
comes out clean or is quarantined with a list of reasons. Nothing is
silently dropped: a rejected posting is evidence, not garbage.
"""

from datetime import date

# The fields a posting must have, and must not have empty.
REQUIRED_FIELDS = ("id", "title", "company", "description")


def check_required_fields(job: dict) -> list[str]:
    """One reason per missing or empty required field."""
    reasons = []
    for field in REQUIRED_FIELDS:
        value = job.get(field)  # .get returns None instead of raising if the key is absent
        if value is None:
            reasons.append(f"missing field: {field}")
        elif isinstance(value, str) and not value.strip():
            reasons.append(f"empty field: {field}")
    return reasons


def check_posted_date(job: dict) -> list[str]:
    """A posting dated in the future is not a real posting."""
    raw = job.get("posted_at")
    if raw is None:
        return ["missing field: posted_at"]
    try:
        posted = date.fromisoformat(raw)
    except ValueError:
        return [f"invalid date: {raw}"]
    if posted > date.today():
        return [f"posted in the future: {raw}"]
    return []


def fingerprint(job: dict) -> str:
    """What makes two postings 'the same', regardless of their ids."""
    parts = (job.get("title", ""), job.get("company", ""), job.get("description", ""))
    # Lowercase and strip so "Data Engineer " and "data engineer" collide.
    return "|".join(p.strip().lower() for p in parts)


class QualityGate:
    """Runs all checks on a posting, remembering what it has already seen
    so it can catch duplicates. One gate per pipeline run."""

    def __init__(self, known_ids: set[str] | None = None) -> None:
        # Ids already in storage. Seeing one of these again is not a data
        # problem, just a re-run, so they are kept apart from duplicates
        # found within the source itself.
        self.known_ids: set[str] = set(known_ids or ())
        self.seen_ids: set[str] = set()
        self.seen_fingerprints: set[str] = set()

    def already_stored(self, job: dict) -> bool:
        """True if this id came from storage. Still records the fingerprint,
        so a copy of a stored posting under a new id is caught later."""
        if job.get("id") in self.known_ids:
            self.seen_fingerprints.add(fingerprint(job))
            return True
        return False

    def check_duplicates(self, job: dict) -> list[str]:
        reasons = []
        job_id = job.get("id")
        fp = fingerprint(job)

        if job_id in self.seen_ids:
            reasons.append(f"duplicate id: {job_id}")
        elif fp in self.seen_fingerprints:
            reasons.append("duplicate posting under a different id")

        # Remember it either way, so a third copy is also caught.
        if job_id is not None:
            self.seen_ids.add(job_id)
        self.seen_fingerprints.add(fp)
        return reasons

    def check(self, job: dict) -> list[str]:
        """Run every check. Empty list means the posting is clean."""
        return (
            check_required_fields(job)
            + check_posted_date(job)
            + self.check_duplicates(job)
        )


if __name__ == "__main__":
    samples = [
        {"id": "ok-1", "title": "Data Engineer", "company": "Elm",
         "description": "Build pipelines.", "posted_at": "2026-09-01"},
        {"id": "bad-1", "title": "Data Engineer", "company": "Elm",
         "description": "", "posted_at": "2026-09-01"},
        {"id": "bad-2", "company": "Elm",
         "description": "No title here.", "posted_at": "2027-01-15"},
        {"id": "ok-1", "title": "Data Engineer", "company": "Elm",
         "description": "Build pipelines.", "posted_at": "2026-09-01"},
        {"id": "ok-9", "title": "data engineer", "company": "Elm",
         "description": "Build pipelines.", "posted_at": "2026-09-03"},
    ]
    gate = QualityGate()
    for job in samples:
        reasons = gate.check(job)
        verdict = "CLEAN" if not reasons else "QUARANTINE"
        print(f"{job['id']:<6} {verdict:<11} {reasons}")
