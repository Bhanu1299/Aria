"""tests/test_dashboard_answers.py — the My answers tab: cards deduped by meaning, one Save,
read-only info groups with Edit, Check these, history collapsed."""
from __future__ import annotations

import json
import re

import pytest

import aria.features.answer_bank as ab
from aria.core import paths
from aria.dashboard import answers
from aria.dashboard.sources import Source, parse_review

SCALE = """# Review: Software Engineer - New Grad at Scale AI

- **Form:** Greenhouse, https://job-boards.greenhouse.io/scaleai/jobs/1
- **Sheet row:** 3

## Needs you (4)

| Field | Value | Required | Note |
|---|---|---|---|
| Are you legally authorized to work in the United States? |  | yes | no work authorization in your bank yet |
| Start date year |  | yes | no education start year in your bank yet |
| Resume/CV |  | yes | file upload |
| School |  | yes | no school in your bank yet |

## Guessed by the LLM: check these (2)

| Field | Value | Required | Note |
|---|---|---|---|
| Years of Python | 3 | yes |  |
| What school are you currently attending? | University at Buffalo, SUNY | yes |  |
"""

ZIP = """# Review: Software Engineer - New Grad at ZipRecruiter

- **Form:** Greenhouse, https://job-boards.greenhouse.io/ziprecruiter/jobs/2
- **Sheet row:** 5

## Needs you (3)

| Field | Value | Required | Note |
|---|---|---|---|
| Are you legally authorized to work in the United States for our Company? |  | yes | no work authorization in your bank yet |
| What archetype are you most closely aligned with? |  | yes | not answerable from your profile |
| question_68569459 |  | yes | still empty |
"""

PALANTIR = """# Review: Software Engineer at Palantir

- **Form:** Greenhouse, https://x.example/p

## Needs you (1)

| Field | Value | Required | Note |
|---|---|---|---|
| cards[dbe30cab-7c88][field1] |  | yes | still empty |
"""


@pytest.fixture(autouse=True)
def bank(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IDENTITY_JSON", tmp_path / "identity.json")
    monkeypatch.setattr(ab, "BANK_PATH", tmp_path / "answer_bank.json")
    (tmp_path / "answer_bank.json").write_text(json.dumps({"answers": {
        "first_name": "Alex", "school": "University at Buffalo, SUNY", "work_authorization": None,
        "edu_start_year": None, "sponsorship": None, "relocation": "Yes"}}))
    return tmp_path / "answer_bank.json"


@pytest.fixture
def reviews(tmp_path):
    crop = tmp_path / "zip_field0.png"
    crop.write_bytes(b"png")
    out = {}
    for name, text, unl, mtime in (("scale.md", SCALE, [], 3.0),
                                   ("zip.md", ZIP, [{"name": "question_68569459", "tag": "input", "type": "text",
                                                     "nearby": "Preferred pronunciation of your name",
                                                     "crop": str(crop)}], 2.0),
                                   ("palantir.md", PALANTIR, None, 1.0)):     # ran before field context existed
        parsed = parse_review(text)
        parsed.update(path=str(tmp_path / name), mtime=mtime, unlabeled=unl)
        out[str(tmp_path / name)] = parsed
    return out


def test_cards_dedupe_by_meaning_and_list_the_jobs(reviews):
    cards, _skipped = answers.cards(ab.load(), reviews)
    by_title = {c["title"]: c for c in cards}
    work = [c for c in cards if c["key"] == "work_authorization"]
    assert len(work) == 1 and len(work[0]["questions"]) == 2
    assert [j["company"] for j in work[0]["jobs"]] == ["Scale AI", "ZipRecruiter"]
    assert work[0]["yes_no"] is True
    assert "School" not in by_title                                           # the bank answers it now
    assert any(c["key"] == "edu_start_year" for c in cards)
    assert any(c["questions"] == ["What archetype are you most closely aligned with?"] and not c["yes_no"]
               for c in cards)


def test_unlabeled_fields_only_with_context(reviews):
    cards, _ = answers.cards(ab.load(), reviews)
    unl = [c for c in cards if c["unlabeled"]]
    assert len(unl) == 1 and unl[0]["nearby"] == "Preferred pronunciation of your name"
    assert unl[0]["crop"] == "zip_field0.png" and unl[0]["jobs"][0]["company"] == "ZipRecruiter"
    assert not any("cards[" in q for c in cards for q in c["questions"])     # pre-context Palantir rows dropped


def test_answer_card_sets_the_field_and_saves_every_wording(bank):
    qs = ["Are you legally authorized to work in the United States?",
          "Are you legally authorized to work in the United States for our Company?"]
    answers.act({"op": "answer_cards", "cards": [{"key": "work_authorization", "questions": qs, "value": "Yes",
                                                  "scope": "all"}]})
    data = json.loads(bank.read_text())
    assert data["answers"]["work_authorization"] == "Yes"
    assert "are you legally authorized to work in the united states for our company" in data["aliases"]["work_authorization"]


def test_answer_card_only_for_one_job(bank):
    answers.act({"op": "answer_cards", "cards": [{"key": "", "questions": ["What archetype are you most closely aligned with?"],
                                                  "value": "Full Stack", "scope": "job",
                                                  "job": "ziprecruiter | software engineer new grad"}]})
    assert ab.lookup("What archetype are you most closely aligned with?",
                     job={"company": "ZipRecruiter", "role": "Software Engineer - New Grad"}).value == "Full Stack"
    assert json.loads(bank.read_text())["answers"].get("archetype") is None


def test_answer_cards_skip_empty_and_reject_all_empty(bank):
    with pytest.raises(answers.AnswerError):
        answers.act({"op": "answer_cards", "cards": [{"key": "sponsorship", "questions": ["x"], "value": " "}]})


def test_set_group_saves_values_and_always_ask(bank):
    answers.act({"op": "set_group", "values": {"relocation": "No", "sponsorship": "No"},
                 "always_ask": {"relocation": True}})
    data = json.loads(bank.read_text())
    assert data["answers"]["relocation"] == "No" and data["answers"]["sponsorship"] == "No"
    assert "relocation" in data["always_ask"]


def test_check_these_skips_guesses_the_bank_now_answers(reviews):
    rows = answers.check_these(ab.load(), reviews)
    assert [(r["field"], r["value"], r["company"]) for r in rows] == [("Years of Python", "3", "Scale AI")]


def test_info_groups_put_unset_fields_first(bank):
    groups = dict((g["group"], g["rows"]) for g in answers.info_groups(ab.load(), json.loads(bank.read_text())))
    work = groups["Work preferences"]
    first_set = next(i for i, r in enumerate(work) if r["value"])
    assert all(not r["value"] for r in work[:first_set]) and any(r["key"] == "work_authorization" for r in work[:first_set])
    assert [r["label"] for r in groups["Education"]][:1] != ["School"] or groups["Education"][0]["value"]


def test_render_layout(reviews):
    html = answers.render(Source(True, reviews))
    order = [html.index(t) for t in ("Needs your answer", "Check these", "Your info", "<details class=\"history")]
    assert order == sorted(order)
    assert html.count('id="saveall"') == 1 and "only for ZipRecruiter" in html
    assert 'class="yn"' in html                                                # yes/no buttons
    assert "Work authorization" in html and 'class="kv unset"' in html and "not set" in html
    view = re.sub(r'<div class="edit".*?</div><!--/edit-->', "", html, flags=re.S)
    assert "always ask" not in view.lower()                                    # only in the edit view
    assert "always ask" in html.lower()
    shown = re.sub(r"<[^>]+>", " ", html)                                       # what you read, not attributes
    for hidden in ("first_name", "work_authorization", "maps to", "answer bank", "identity.json"):
        assert hidden not in shown, hidden
    assert "cards[dbe30cab" not in html
    assert "Preferred pronunciation of your name" in html and "/review-img?f=zip_field0.png" in html


def test_short_field_wordings_take_the_field_name(reviews):
    cards, _ = answers.cards(ab.load(), reviews)
    start = next(c for c in cards if c["key"] == "edu_start_year")
    assert start["title"] == "Education start year" and start["questions"] == ["Start date year"]
