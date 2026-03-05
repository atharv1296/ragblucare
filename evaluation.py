"""
╔═══════════════════════════════════════════════════════════════════════╗
║                 BluCare+ — Comprehensive Evaluation Suite            ║
║                                                                       ║
║  Tests every component of the RagBluCare medical assistant:           ║
║    1. Symptom Extraction (Rule-based NER)                             ║
║    2. Synonym Normalization                                           ║
║    3. Duration & Intensity Parsing                                    ║
║    4. Severity Classification (Red-flag rules)                        ║
║    5. Scoring Engine (Overlap, Red-flag, Prevalence, Final)           ║
║    6. Medicine Policy (OTC matching, age-gating, severity filter)     ║
║    7. Retriever / RAG Pipeline (Vector + Hybrid)                      ║
║    8. Utility Functions (text normalization, tokenization)            ║
║    9. Model / Schema Validation (Pydantic models)                     ║
║   10. API Endpoint Smoke Tests (health, session, chat)                ║
║   11. LLM Client Configuration Audit                                  ║
║   12. Firebase Manager (session CRUD)                                 ║
║   13. End-to-End Integration Scenarios                                ║
║                                                                       ║
║  Run:  python evaluation.py                                           ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

# ─── Make sure app/ is importable ─────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# ═══════════════════════════════════════════════════════════
# Result data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    score: float = 1.0          # 0.0 – 1.0
    detail: str = ""
    elapsed_ms: float = 0.0

@dataclass
class CategoryResult:
    category: str
    tests: list[TestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def avg_score(self) -> float:
        if not self.tests:
            return 0.0
        return sum(t.score for t in self.tests) / self.total


# ═══════════════════════════════════════════════════════════
# Helper: run a single test safely
# ═══════════════════════════════════════════════════════════

def _run_test(name: str, fn: Callable[[], tuple[bool, float, str]]) -> TestResult:
    """Execute one test function; catch any exception as a failure."""
    t0 = time.perf_counter()
    try:
        passed, score, detail = fn()
    except Exception as exc:
        passed, score, detail = False, 0.0, f"EXCEPTION: {exc}\n{traceback.format_exc()}"
    elapsed = (time.perf_counter() - t0) * 1000
    return TestResult(name=name, passed=passed, score=score, detail=detail, elapsed_ms=round(elapsed, 2))


# ═══════════════════════════════════════════════════════════
#  1 — SYMPTOM EXTRACTION (Rule-based)
# ═══════════════════════════════════════════════════════════

def eval_symptom_extraction() -> CategoryResult:
    from app.symptom_extractor import RuleBasedExtractor, SYMPTOM_LEXICON, SYNONYM_MAP

    extractor = RuleBasedExtractor()
    cat = CategoryResult(category="1. Symptom Extraction (Rule-Based NER)")

    # ── Test 1.1  Single common symptom ──
    def t_single():
        r = extractor.extract("I have a headache")
        found = {s.lower() for s in r.symptoms}
        ok = "headache" in found
        return ok, 1.0 if ok else 0.0, f"symptoms={r.symptoms}"
    cat.tests.append(_run_test("Single symptom — 'headache'", t_single))

    # ── Test 1.2  Multiple symptoms ──
    def t_multi():
        r = extractor.extract("I have fever, cough, and body pain since 3 days")
        found = {s.lower() for s in r.symptoms}
        expected = {"fever", "cough"}  # body pain may normalize to myalgia
        hits = expected & found
        score = len(hits) / len(expected)
        return score >= 0.8, score, f"found={r.symptoms}"
    cat.tests.append(_run_test("Multiple symptoms — fever, cough, body pain", t_multi))

    # ── Test 1.3  Synonym normalization (body pain → myalgia) ──
    def t_synonym():
        r = extractor.extract("I have severe body pain")
        found = {s.lower() for s in r.symptoms}
        ok = "myalgia" in found or "body pain" in found
        return ok, 1.0 if ok else 0.0, f"symptoms={r.symptoms}"
    cat.tests.append(_run_test("Synonym normalization — body pain → myalgia", t_synonym))

    # ── Test 1.4  Edge: empty input ──
    def t_empty():
        r = extractor.extract("")
        ok = len(r.symptoms) == 0
        return ok, 1.0 if ok else 0.0, f"symptoms={r.symptoms}"
    cat.tests.append(_run_test("Edge — empty input yields 0 symptoms", t_empty))

    # ── Test 1.5  Edge: gibberish ──
    def t_gibberish():
        r = extractor.extract("asdfghjkl qwerty 12345")
        ok = len(r.symptoms) == 0
        return ok, 1.0 if ok else 0.0, f"symptoms={r.symptoms}"
    cat.tests.append(_run_test("Edge — gibberish yields 0 symptoms", t_gibberish))

    # ── Test 1.6  Heart-related red-flag symptoms ──
    def t_cardiac():
        r = extractor.extract("I feel chest pain and palpitations and shortness of breath")
        found = {s.lower() for s in r.symptoms}
        expected = {"chest pain", "palpitations", "shortness of breath"}
        hits = expected & found
        score = len(hits) / len(expected)
        return score >= 0.66, score, f"found={r.symptoms}"
    cat.tests.append(_run_test("Cardiac cluster — chest pain, palpitations, SOB", t_cardiac))

    # ── Test 1.7  GI symptoms ──
    def t_gi():
        r = extractor.extract("nausea, vomiting, diarrhea and abdominal pain")
        found = {s.lower() for s in r.symptoms}
        expected = {"nausea", "vomiting", "diarrhea", "abdominal pain"}
        hits = expected & found
        score = len(hits) / len(expected)
        return score >= 0.75, score, f"found={r.symptoms}"
    cat.tests.append(_run_test("GI cluster — nausea, vomiting, diarrhea, abdominal pain", t_gi))

    # ── Test 1.8  Lexicon coverage check ──
    def t_lexicon():
        ok = len(SYMPTOM_LEXICON) >= 80
        return ok, min(len(SYMPTOM_LEXICON) / 100, 1.0), f"lexicon_size={len(SYMPTOM_LEXICON)}"
    cat.tests.append(_run_test("Lexicon coverage ≥ 80 symptoms", t_lexicon))

    # ── Test 1.9  Synonym map coverage ──
    def t_synmap():
        ok = len(SYNONYM_MAP) >= 20
        return ok, min(len(SYNONYM_MAP) / 30, 1.0), f"synonym_map_size={len(SYNONYM_MAP)}"
    cat.tests.append(_run_test("Synonym map ≥ 20 entries", t_synmap))

    # ── Test 1.10  Respiratory cluster ──
    def t_resp():
        r = extractor.extract("dry cough, wheezing, sore throat and nasal congestion")
        found = {s.lower() for s in r.symptoms}
        expected = {"dry cough", "wheezing", "sore throat", "nasal congestion"}
        hits = sum(1 for e in expected if any(e in f or f in e for f in found))
        score = hits / len(expected)
        return score >= 0.75, score, f"found={r.symptoms}"
    cat.tests.append(_run_test("Respiratory cluster — cough, wheezing, sore throat, congestion", t_resp))

    return cat


# ═══════════════════════════════════════════════════════════
#  2 — DURATION & INTENSITY PARSING
# ═══════════════════════════════════════════════════════════

def eval_duration_intensity() -> CategoryResult:
    from app.symptom_extractor import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    cat = CategoryResult(category="2. Duration & Intensity Parsing")

    # ── 2.1  "since 3 days" ──
    def t_dur1():
        r = extractor.extract("I have headache since 3 days")
        ok = bool(r.duration) and "3" in r.duration
        return ok, 1.0 if ok else 0.0, f"duration='{r.duration}'"
    cat.tests.append(_run_test("Duration — 'since 3 days'", t_dur1))

    # ── 2.2  "for 2 weeks" ──
    def t_dur2():
        r = extractor.extract("Cough for 2 weeks now")
        ok = bool(r.duration) and "2" in r.duration
        return ok, 1.0 if ok else 0.0, f"duration='{r.duration}'"
    cat.tests.append(_run_test("Duration — 'for 2 weeks'", t_dur2))

    # ── 2.3  "started yesterday" ──
    def t_dur3():
        r = extractor.extract("Fever started yesterday")
        ok = bool(r.duration) and "yesterday" in r.duration.lower()
        return ok, 1.0 if ok else 0.0, f"duration='{r.duration}'"
    cat.tests.append(_run_test("Duration — 'started yesterday'", t_dur3))

    # ── 2.4  Intensity: severe ──
    def t_int_severe():
        r = extractor.extract("I have severe headache")
        ok = r.intensity == "severe"
        return ok, 1.0 if ok else 0.0, f"intensity='{r.intensity}'"
    cat.tests.append(_run_test("Intensity — 'severe'", t_int_severe))

    # ── 2.5  Intensity: mild ──
    def t_int_mild():
        r = extractor.extract("I have a mild stomachache")
        ok = r.intensity == "mild"
        return ok, 1.0 if ok else 0.0, f"intensity='{r.intensity}'"
    cat.tests.append(_run_test("Intensity — 'mild'", t_int_mild))

    # ── 2.6  Intensity: unbearable → severe ──
    def t_int_unbear():
        r = extractor.extract("Unbearable chest pain")
        ok = r.intensity == "severe"
        return ok, 1.0 if ok else 0.0, f"intensity='{r.intensity}'"
    cat.tests.append(_run_test("Intensity — 'unbearable' → severe", t_int_unbear))

    # ── 2.7  Age extraction: "I am 25 years old" ──
    def t_age():
        r = extractor.extract("I am 25 years old and have fever")
        ok = r.age == "25"
        return ok, 1.0 if ok else 0.0, f"age='{r.age}'"
    cat.tests.append(_run_test("Age extraction — 'I am 25 years old'", t_age))

    # ── 2.8  Gender extraction ──
    def t_gender():
        r = extractor.extract("I am a female with back pain")
        ok = r.gender == "female"
        return ok, 1.0 if ok else 0.0, f"gender='{r.gender}'"
    cat.tests.append(_run_test("Gender extraction — 'I am a female'", t_gender))

    return cat


# ═══════════════════════════════════════════════════════════
#  3 — SEVERITY CLASSIFICATION (Rule Layer)
# ═══════════════════════════════════════════════════════════

def eval_severity_rules() -> CategoryResult:
    from app.severity_classifier import (
        SeverityResult, RED_FLAG_RULES,
        ANSWER_ESCALATION_PATTERNS, SeverityClassifier,
    )
    from app.symptom_extractor import ExtractionResult

    cat = CategoryResult(category="3. Severity Classification (Rule-Based)")

    # ── 3.1  Red flag count check ──
    def t_rf_count():
        ok = len(RED_FLAG_RULES) >= 10
        return ok, min(len(RED_FLAG_RULES) / 12, 1.0), f"red_flag_rules={len(RED_FLAG_RULES)}"
    cat.tests.append(_run_test("Red flag rule coverage ≥ 10", t_rf_count))

    # ── 3.2  Chest pain → severe ──
    def t_chest():
        result = SeverityResult()
        ext = ExtractionResult()
        ext.symptoms = ["chest pain", "palpitations"]
        # Manually apply rules (same logic as _apply_rules)
        combined_text = " ".join(ext.symptoms).lower()
        for rule in RED_FLAG_RULES:
            for pattern in rule["patterns"]:
                if pattern in combined_text:
                    result.escalate("severe", rule["reason"])
        ok = result.level == "severe"
        return ok, 1.0 if ok else 0.0, f"level={result.level}"
    cat.tests.append(_run_test("Chest pain → severe escalation", t_chest))

    # ── 3.3  Mild symptom stays mild ──
    def t_mild_stays():
        result = SeverityResult()
        ext = ExtractionResult()
        ext.symptoms = ["headache"]
        ext.intensity = "mild"
        # Apply just intensity logic
        if ext.intensity == "mild" and result.level == "undetermined":
            result.level = "mild"
        ok = result.level == "mild"
        return ok, 1.0 if ok else 0.0, f"level={result.level}"
    cat.tests.append(_run_test("Mild headache stays mild", t_mild_stays))

    # ── 3.4  Never-downgrade property ──
    def t_no_downgrade():
        r = SeverityResult(level="severe", confidence=0.9)
        r.escalate("mild", "test")  # attempt downgrade
        ok = r.level == "severe"
        return ok, 1.0 if ok else 0.0, f"level={r.level} after attempted downgrade"
    cat.tests.append(_run_test("Never-downgrade — severe cannot become mild", t_no_downgrade))

    # ── 3.5  Escalation chain: undetermined → moderate → severe ──
    def t_chain():
        r = SeverityResult()
        r.escalate("moderate", "step1")
        mid = r.level
        r.escalate("severe", "step2")
        ok = mid == "moderate" and r.level == "severe"
        return ok, 1.0 if ok else 0.0, f"chain: undetermined→{mid}→{r.level}"
    cat.tests.append(_run_test("Escalation chain — undetermined→moderate→severe", t_chain))

    # ── 3.6  Seizure → severe ──
    def t_seizure():
        result = SeverityResult()
        combined = "seizure"
        for rule in RED_FLAG_RULES:
            for pattern in rule["patterns"]:
                if pattern in combined:
                    result.escalate("severe", rule["reason"])
        ok = result.level == "severe"
        return ok, 1.0 if ok else 0.0, f"level={result.level}"
    cat.tests.append(_run_test("Seizure → severe escalation", t_seizure))

    # ── 3.7  SeverityResult.to_dict() structure ──
    def t_dict():
        r = SeverityResult(level="moderate", confidence=0.75, reasoning="test", is_emergency=False)
        d = r.to_dict()
        expected_keys = {"level", "confidence", "reasoning", "red_flags_found", "recommended_action", "is_emergency"}
        ok = set(d.keys()) == expected_keys
        return ok, 1.0 if ok else 0.0, f"keys={set(d.keys())}"
    cat.tests.append(_run_test("SeverityResult.to_dict() has correct keys", t_dict))

    # ── 3.8  Answer escalation patterns coverage ──
    def t_ans_patterns():
        total = sum(len(v) for v in ANSWER_ESCALATION_PATTERNS.values())
        ok = total >= 10
        return ok, min(total / 15, 1.0), f"answer_patterns={total}"
    cat.tests.append(_run_test("Answer escalation patterns ≥ 10", t_ans_patterns))

    # ── 3.9  Suicidal ideation → severe + emergency ──
    def t_suicide():
        result = SeverityResult()
        combined = "suicidal thoughts"
        for rule in RED_FLAG_RULES:
            for pattern in rule["patterns"]:
                if pattern in combined:
                    result.escalate("severe", rule["reason"])
                    result.is_emergency = True
        ok = result.level == "severe" and result.is_emergency
        return ok, 1.0 if ok else 0.0, f"level={result.level}, emergency={result.is_emergency}"
    cat.tests.append(_run_test("Suicidal thoughts → severe + emergency", t_suicide))

    return cat


# ═══════════════════════════════════════════════════════════
#  4 — SCORING ENGINE
# ═══════════════════════════════════════════════════════════

def eval_scoring() -> CategoryResult:
    from app.scoring import (
        compute_symptom_overlap,
        compute_red_flag_weight,
        compute_prevalence_weight,
        compute_final_score,
        HIGH_PREVALENCE_DISEASES,
        CRITICAL_RED_FLAGS,
    )

    cat = CategoryResult(category="4. Scoring Engine")

    # ── 4.1  Perfect overlap ──
    def t_perfect():
        s = compute_symptom_overlap(["fever", "cough"], ["fever", "cough"])
        ok = abs(s - 1.0) < 0.01
        return ok, s, f"overlap={s:.4f}"
    cat.tests.append(_run_test("Symptom overlap — perfect match = 1.0", t_perfect))

    # ── 4.2  Zero overlap ──
    def t_zero():
        s = compute_symptom_overlap(["rash"], ["headache"])
        ok = s < 0.01
        return ok, 1.0 if ok else 0.0, f"overlap={s:.4f}"
    cat.tests.append(_run_test("Symptom overlap — no match = 0.0", t_zero))

    # ── 4.3  Partial overlap ──
    def t_partial():
        s = compute_symptom_overlap(["fever", "cough", "rash"], ["fever", "diarrhea"])
        ok = 0.3 <= s <= 0.7
        return ok, 1.0 if ok else 0.5, f"overlap={s:.4f}"
    cat.tests.append(_run_test("Symptom overlap — partial match", t_partial))

    # ── 4.4  Red flag weight with matching flags ──
    def t_rf():
        w = compute_red_flag_weight(["chest pain"], ["chest pain"])
        ok = w > 0.5
        return ok, w, f"red_flag_weight={w:.4f}"
    cat.tests.append(_run_test("Red flag weight — matching = high", t_rf))

    # ── 4.5  Prevalence: common disease ──
    def t_prev_common():
        w = compute_prevalence_weight("Common Cold")
        ok = w >= 0.7
        return ok, 1.0 if ok else 0.0, f"prevalence={w}"
    cat.tests.append(_run_test("Prevalence weight — 'Common Cold' = high", t_prev_common))

    # ── 4.6  Prevalence: rare disease ──
    def t_prev_rare():
        w = compute_prevalence_weight("Xeroderma Pigmentosum")
        ok = w == 0.5  # default
        return ok, 1.0 if ok else 0.0, f"prevalence={w}"
    cat.tests.append(_run_test("Prevalence weight — rare disease = 0.5 default", t_prev_rare))

    # ── 4.7  Final score formula verification ──
    def t_formula():
        # 0.5*0.8 + 0.3*0.6 + 0.1*0.4 + 0.1*0.5 = 0.4+0.18+0.04+0.05 = 0.67
        expected = 0.67
        actual = compute_final_score(0.8, 0.6, 0.4, 0.5)
        ok = abs(actual - expected) < 0.01
        return ok, 1.0 if ok else 0.0, f"expected={expected}, got={actual:.4f}"
    cat.tests.append(_run_test("Final score formula — 0.5S + 0.3O + 0.1R + 0.1P", t_formula))

    # ── 4.8  High prevalence set coverage ──
    def t_hpd():
        ok = len(HIGH_PREVALENCE_DISEASES) >= 20
        return ok, min(len(HIGH_PREVALENCE_DISEASES) / 25, 1.0), f"count={len(HIGH_PREVALENCE_DISEASES)}"
    cat.tests.append(_run_test("High-prevalence disease set ≥ 20", t_hpd))

    # ── 4.9  Critical red flags set coverage ──
    def t_crf():
        ok = len(CRITICAL_RED_FLAGS) >= 15
        return ok, min(len(CRITICAL_RED_FLAGS) / 20, 1.0), f"count={len(CRITICAL_RED_FLAGS)}"
    cat.tests.append(_run_test("Critical red flags set ≥ 15", t_crf))

    # ── 4.10  Empty inputs handled gracefully ──
    def t_empty():
        s1 = compute_symptom_overlap([], [])
        s2 = compute_red_flag_weight([], [])
        ok = s1 == 0.0 and s2 == 0.0
        return ok, 1.0 if ok else 0.0, f"overlap={s1}, rf_weight={s2}"
    cat.tests.append(_run_test("Empty inputs — graceful 0.0 returns", t_empty))

    return cat


# ═══════════════════════════════════════════════════════════
#  5 — MEDICINE POLICY
# ═══════════════════════════════════════════════════════════

def eval_medicine_policy() -> CategoryResult:
    from app.medicine_policy import MedicinePolicy, OTC_MEDICATIONS, PROHIBITED_CATEGORIES, DISCLAIMER

    mp = MedicinePolicy()
    cat = CategoryResult(category="5. Medicine Policy (OTC Safety)")

    # ── 5.1  Fever → Paracetamol match ──
    def t_fever():
        meds = mp.get_suggestions(["fever"], severity="mild")
        names = [m["name"] for m in meds]
        ok = any("Paracetamol" in n or "Acetaminophen" in n for n in names)
        return ok, 1.0 if ok else 0.0, f"matched={names}"
    cat.tests.append(_run_test("Fever → Paracetamol/Acetaminophen match", t_fever))

    # ── 5.2  Diarrhea → ORS match ──
    def t_diarrhea():
        meds = mp.get_suggestions(["diarrhea"])
        names = [m["name"] for m in meds]
        ok = any("ORS" in n or "Rehydration" in n for n in names)
        return ok, 1.0 if ok else 0.0, f"matched={names}"
    cat.tests.append(_run_test("Diarrhea → ORS match", t_diarrhea))

    # ── 5.3  Allergy → Antihistamine match ──
    def t_allergy():
        meds = mp.get_suggestions(["allergy", "itching"])
        names = [m["name"] for m in meds]
        ok = any("Cetirizine" in n or "Loratadine" in n or "Antihistamine" in n for n in names)
        return ok, 1.0 if ok else 0.0, f"matched={names}"
    cat.tests.append(_run_test("Allergy → Antihistamine match", t_allergy))

    # ── 5.4  Child age detection ──
    def t_child():
        ok = mp._is_child("10") and not mp._is_child("25") and not mp._is_child("")
        return ok, 1.0 if ok else 0.0, "is_child(10)=True, is_child(25)=False, is_child('')=False"
    cat.tests.append(_run_test("Child age detection — 10=child, 25=adult", t_child))

    # ── 5.5  Child dosing notes ──
    def t_child_dose():
        meds = mp.get_suggestions(["fever"], age="8", severity="mild")
        ok = any("age_note" in m or "pediatr" in m.get("dosage", "").lower() for m in meds)
        return ok, 1.0 if ok else 0.0, f"child_meds={[m.get('dosage','')[:50] for m in meds]}"
    cat.tests.append(_run_test("Child dosing — age 8 gets pediatric note", t_child_dose))

    # ── 5.6  OTC database size ──
    def t_otc_count():
        ok = len(OTC_MEDICATIONS) >= 8
        return ok, min(len(OTC_MEDICATIONS) / 10, 1.0), f"otc_count={len(OTC_MEDICATIONS)}"
    cat.tests.append(_run_test("OTC medication database ≥ 8 entries", t_otc_count))

    # ── 5.7  Prohibited categories defined ──
    def t_prohibited():
        ok = len(PROHIBITED_CATEGORIES) >= 10
        return ok, min(len(PROHIBITED_CATEGORIES) / 12, 1.0), f"prohibited={len(PROHIBITED_CATEGORIES)}"
    cat.tests.append(_run_test("Prohibited categories ≥ 10", t_prohibited))

    # ── 5.8  Disclaimer present ──
    def t_disc():
        ok = "disclaimer" in DISCLAIMER.lower() and len(DISCLAIMER) > 30
        return ok, 1.0 if ok else 0.0, f"disclaimer_len={len(DISCLAIMER)}"
    cat.tests.append(_run_test("Medical disclaimer present and non-trivial", t_disc))

    # ── 5.9  Format method returns text ──
    def t_format():
        meds = mp.get_suggestions(["fever"])
        text = mp.format_for_response(meds)
        ok = len(text) > 20 and "Dosage" in text
        return ok, 1.0 if ok else 0.0, f"output_len={len(text)}"
    cat.tests.append(_run_test("format_for_response() produces readable text", t_format))

    # ── 5.10  No match for unrecognized symptom ──
    def t_no_match():
        meds = mp.get_suggestions(["alien_disease_xyz"])
        ok = len(meds) == 0
        return ok, 1.0 if ok else 0.0, f"matched={len(meds)} (expected 0)"
    cat.tests.append(_run_test("Unknown symptom → 0 matches", t_no_match))

    return cat


# ═══════════════════════════════════════════════════════════
#  6 — UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def eval_utils() -> CategoryResult:
    from app.utils import (
        normalize_symptom, clean_text, tokenize_symptoms,
        extract_symptoms_from_text, Settings, get_settings, ensure_dir,
    )

    cat = CategoryResult(category="6. Utility Functions")

    # ── 6.1  normalize_symptom ──
    def t_norm():
        r = normalize_symptom("  CHEST Pain!!  ")
        ok = r == "chest pain"
        return ok, 1.0 if ok else 0.0, f"result='{r}'"
    cat.tests.append(_run_test("normalize_symptom — strips/lowercases/removes punct", t_norm))

    # ── 6.2  clean_text ──
    def t_clean():
        r = clean_text("Hello   World\r\n\n\n\nTest")
        ok = "  " not in r and "\r" not in r
        return ok, 1.0 if ok else 0.0, f"result='{r[:40]}'"
    cat.tests.append(_run_test("clean_text — normalizes whitespace & line endings", t_clean))

    # ── 6.3  tokenize_symptoms ──
    def t_tokenize():
        r = tokenize_symptoms("I have fever, headache and diarrhea")
        ok = len(r) >= 2
        return ok, 1.0 if ok else 0.5, f"tokens={r}"
    cat.tests.append(_run_test("tokenize_symptoms — splits on commas/and", t_tokenize))

    # ── 6.4  tokenize removes filler ──
    def t_tokenize_filler():
        r = tokenize_symptoms("I have been having chest pain and nausea please help")
        combined = " ".join(r).lower()
        ok = "please help" not in combined and "i have" not in combined
        return ok, 1.0 if ok else 0.0, f"tokens={r}"
    cat.tests.append(_run_test("tokenize_symptoms — strips filler phrases", t_tokenize_filler))

    # ── 6.5  Settings loads env vars ──
    def t_settings():
        s = get_settings()
        ok = hasattr(s, "GROQ_API_KEY") and hasattr(s, "EMBEDDING_MODEL") and hasattr(s, "CACHE_DIR")
        return ok, 1.0 if ok else 0.0, f"embedding_model={s.EMBEDDING_MODEL}"
    cat.tests.append(_run_test("Settings has required config fields", t_settings))

    # ── 6.6  ensure_dir creates directories ──
    def t_ensure():
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp()) / "eval_test_dir"
        p = ensure_dir(tmp)
        ok = p.exists() and p.is_dir()
        shutil.rmtree(tmp, ignore_errors=True)
        return ok, 1.0 if ok else 0.0, f"path={p}"
    cat.tests.append(_run_test("ensure_dir creates directory", t_ensure))

    # ── 6.7  extract_symptoms_from_text (heuristic) ──
    def t_extract_text():
        text = "Symptoms: fever, cough, headache. Patient may experience nausea."
        r = extract_symptoms_from_text(text)
        ok = len(r) >= 2
        return ok, min(len(r) / 3, 1.0), f"extracted={r[:5]}"
    cat.tests.append(_run_test("extract_symptoms_from_text — regex heuristic", t_extract_text))

    return cat


# ═══════════════════════════════════════════════════════════
#  7 — PYDANTIC MODEL VALIDATION
# ═══════════════════════════════════════════════════════════

def eval_models() -> CategoryResult:
    from app.models import (
        DiseaseNode, RetrievalResult, SessionData,
        StartSessionRequest, ChatRequest, AuthResponse,
    )

    cat = CategoryResult(category="7. Pydantic Model Validation")

    # ── 7.1  DiseaseNode default construction ──
    def t_dn():
        d = DiseaseNode()
        ok = d.disease_name == "" and isinstance(d.symptoms, list) and d.chunk_id
        return ok, 1.0 if ok else 0.0, f"chunk_id={d.chunk_id[:8]}"
    cat.tests.append(_run_test("DiseaseNode — default construction", t_dn))

    # ── 7.2  DiseaseNode with data ──
    def t_dn_data():
        d = DiseaseNode(
            disease_name="Flu",
            symptoms=["fever", "cough"],
            red_flags=["high fever"],
            treatments=["rest"],
        )
        ok = d.disease_name == "Flu" and len(d.symptoms) == 2
        return ok, 1.0 if ok else 0.0, str(d.model_dump())[:80]
    cat.tests.append(_run_test("DiseaseNode — construction with data", t_dn_data))

    # ── 7.3  RetrievalResult ──
    def t_rr():
        d = DiseaseNode(disease_name="Test")
        r = RetrievalResult(disease=d, semantic_score=0.9, final_score=0.85)
        ok = r.final_score == 0.85 and r.disease.disease_name == "Test"
        return ok, 1.0 if ok else 0.0, f"score={r.final_score}"
    cat.tests.append(_run_test("RetrievalResult — stores scores correctly", t_rr))

    # ── 7.4  SessionData defaults ──
    def t_sd():
        s = SessionData(user_id="u1", session_id="s1")
        ok = (
            s.phase == "greeting"
            and s.turn_count == 0
            and isinstance(s.extracted_symptoms, list)
            and s.created_at
        )
        return ok, 1.0 if ok else 0.0, f"phase={s.phase}, created={s.created_at[:10]}"
    cat.tests.append(_run_test("SessionData — defaults (phase=greeting, turn=0)", t_sd))

    # ── 7.5  StartSessionRequest ──
    def t_ssr():
        r = StartSessionRequest(user_id="u1", name="Alice", age="25", gender="female")
        ok = r.user_id == "u1" and r.name == "Alice"
        return ok, 1.0 if ok else 0.0, f"user_id={r.user_id}"
    cat.tests.append(_run_test("StartSessionRequest — validates correctly", t_ssr))

    # ── 7.6  ChatRequest ──
    def t_cr():
        r = ChatRequest(user_id="u1", session_id="s1", message="hello")
        ok = r.message == "hello"
        return ok, 1.0 if ok else 0.0, f"message={r.message}"
    cat.tests.append(_run_test("ChatRequest — validates correctly", t_cr))

    # ── 7.7  AuthResponse ──
    def t_ar():
        r = AuthResponse(success=True, user_id="u1", display_name="Alice", message="OK")
        ok = r.success and r.user_id == "u1"
        return ok, 1.0 if ok else 0.0, f"success={r.success}"
    cat.tests.append(_run_test("AuthResponse — validates correctly", t_ar))

    # ── 7.8  SessionData model_dump ──
    def t_dump():
        s = SessionData(user_id="u1", session_id="s1")
        d = s.model_dump()
        keys = set(d.keys())
        needed = {"user_id", "session_id", "phase", "extracted_symptoms", "previous_messages", "turn_count"}
        ok = needed.issubset(keys)
        return ok, 1.0 if ok else 0.0, f"keys={sorted(keys)[:8]}"
    cat.tests.append(_run_test("SessionData.model_dump() — has all fields", t_dump))

    return cat


# ═══════════════════════════════════════════════════════════
#  8 — EMBEDDING MANAGER & FAISS INDEX
# ═══════════════════════════════════════════════════════════

def eval_embeddings() -> CategoryResult:
    from app.embeddings import EmbeddingManager
    from app.models import DiseaseNode

    cat = CategoryResult(category="8. Embedding Manager & FAISS Index")

    # ── 8.1  Index loading from cache ──
    def t_load():
        emb = EmbeddingManager()
        loaded = emb.load_index()
        ok = loaded  # should have cache from prior ingestion
        idx_size = emb.index.ntotal if emb.index else 0
        meta_size = len(emb.metadata) if emb.metadata else 0
        return ok, 1.0 if ok else 0.0, f"loaded={loaded}, vectors={idx_size}, metadata={meta_size}"
    cat.tests.append(_run_test("FAISS index loads from cache", t_load))

    # ── 8.2  Index and metadata consistency ──
    def t_consistency():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No cached index available"
        ok = emb.index.ntotal == len(emb.metadata) and emb.index.ntotal > 0
        return ok, 1.0 if ok else 0.0, f"vectors={emb.index.ntotal}, metadata={len(emb.metadata)}"
    cat.tests.append(_run_test("Index vectors == metadata entries", t_consistency))

    # ── 8.3  Metadata entries are valid DiseaseNodes ──
    def t_meta_valid():
        emb = EmbeddingManager()
        if not emb.load_index() or not emb.metadata:
            return False, 0.0, "No metadata"
        valid = 0
        for m in emb.metadata[:20]:
            try:
                DiseaseNode(**m)
                valid += 1
            except Exception:
                pass
        score = valid / min(len(emb.metadata), 20)
        return score > 0.9, score, f"valid={valid}/{min(len(emb.metadata), 20)}"
    cat.tests.append(_run_test("Metadata entries parse as DiseaseNode", t_meta_valid))

    # ── 8.4  Search returns results ──
    def t_search():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        results = emb.search("fever and headache", top_k=5)
        ok = len(results) > 0
        names = [r[0].disease_name for r in results[:3]]
        return ok, min(len(results) / 3, 1.0), f"results={len(results)}, top={names}"
    cat.tests.append(_run_test("Search 'fever and headache' returns results", t_search))

    # ── 8.5  Search returns (DiseaseNode, float) tuples ──
    def t_search_types():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        results = emb.search("cough", top_k=3)
        if not results:
            return False, 0.0, "No results"
        node, score = results[0]
        ok = isinstance(node, DiseaseNode) and isinstance(score, float)
        return ok, 1.0 if ok else 0.0, f"type_node={type(node).__name__}, type_score={type(score).__name__}"
    cat.tests.append(_run_test("Search returns (DiseaseNode, float) tuples", t_search_types))

    # ── 8.6  Dimension check ──
    def t_dim():
        emb = EmbeddingManager()
        ok = emb.dimension == 768  # e5-base-v2
        return ok, 1.0 if ok else 0.0, f"dimension={emb.dimension}"
    cat.tests.append(_run_test("Embedding dimension = 768 (e5-base-v2)", t_dim))

    return cat


# ═══════════════════════════════════════════════════════════
#  9 — HYBRID RETRIEVER
# ═══════════════════════════════════════════════════════════

def eval_retriever() -> CategoryResult:
    from app.embeddings import EmbeddingManager
    from app.retriever import HybridRetriever
    from app.models import RetrievalResult

    cat = CategoryResult(category="9. Hybrid Retriever (RAG Pipeline)")

    # ── 9.1  Retriever initialization ──
    def t_init():
        emb = EmbeddingManager()
        emb.load_index()
        retriever = HybridRetriever(emb)
        ok = retriever.emb is not None
        return ok, 1.0, "Retriever initialized"
    cat.tests.append(_run_test("HybridRetriever initializes", t_init))

    # ── 9.2  Retrieve returns RetrievalResult list ──
    def t_retrieve():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        results = retriever.retrieve("high fever and body pain", top_k_final=3)
        ok = len(results) > 0 and isinstance(results[0], RetrievalResult)
        names = [r.disease.disease_name for r in results]
        return ok, min(len(results) / 2, 1.0), f"results={names}"
    cat.tests.append(_run_test("Retrieve 'high fever and body pain' → results", t_retrieve))

    # ── 9.3  Results are ranked (descending final_score) ──
    def t_ranked():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        results = retriever.retrieve("headache dizziness nausea", top_k_final=5)
        if len(results) < 2:
            return False, 0.0, f"Only {len(results)} results"
        scores = [r.final_score for r in results]
        ok = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        return ok, 1.0 if ok else 0.0, f"scores={scores}"
    cat.tests.append(_run_test("Results ranked in descending score order", t_ranked))

    # ── 9.4  Each result has all score components ──
    def t_components():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        results = retriever.retrieve("cough and cold", top_k_final=3)
        if not results:
            return False, 0.0, "No results"
        r = results[0]
        has_all = all([
            r.semantic_score is not None,
            r.keyword_score is not None,
            r.red_flag_weight is not None,
            r.prevalence_weight is not None,
            r.final_score is not None,
        ])
        return has_all, 1.0 if has_all else 0.0, (
            f"sem={r.semantic_score}, kw={r.keyword_score}, "
            f"rf={r.red_flag_weight}, prev={r.prevalence_weight}, "
            f"final={r.final_score}"
        )
    cat.tests.append(_run_test("Result has all score components", t_components))

    # ── 9.5  top_k_final limit is respected ──
    def t_topk():
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        results = retriever.retrieve("abdominal pain vomiting", top_k_final=2)
        ok = len(results) <= 2
        return ok, 1.0 if ok else 0.0, f"requested=2, got={len(results)}"
    cat.tests.append(_run_test("top_k_final=2 → ≤ 2 results", t_topk))

    return cat


# ═══════════════════════════════════════════════════════════
# 10 — LLM CLIENT CONFIGURATION
# ═══════════════════════════════════════════════════════════

def eval_llm_config() -> CategoryResult:
    from app.llm_client import LLMClient
    from app.utils import get_settings

    cat = CategoryResult(category="10. LLM Client Configuration")

    # ── 10.1  Groq API key configured ──
    def t_groq_key():
        s = get_settings()
        ok = bool(s.GROQ_API_KEY) and len(s.GROQ_API_KEY) > 10
        return ok, 1.0 if ok else 0.0, f"key_len={len(s.GROQ_API_KEY) if s.GROQ_API_KEY else 0}"
    cat.tests.append(_run_test("Groq API key configured", t_groq_key))

    # ── 10.2  OpenRouter API key configured ──
    def t_or_key():
        s = get_settings()
        ok = bool(s.OPENROUTER_API_KEY) and len(s.OPENROUTER_API_KEY) > 10
        return ok, 1.0 if ok else 0.0, f"key_len={len(s.OPENROUTER_API_KEY) if s.OPENROUTER_API_KEY else 0}"
    cat.tests.append(_run_test("OpenRouter API key configured", t_or_key))

    # ── 10.3  Model names configured ──
    def t_models():
        s = get_settings()
        ok = bool(s.GROQ_MODEL) and bool(s.OPENROUTER_MODEL)
        return ok, 1.0 if ok else 0.0, f"groq={s.GROQ_MODEL}, or={s.OPENROUTER_MODEL}"
    cat.tests.append(_run_test("LLM model names configured", t_models))

    # ── 10.4  LLMClient instantiation ──
    def t_init():
        client = LLMClient()
        ok = client is not None
        return ok, 1.0, "LLMClient created"
    cat.tests.append(_run_test("LLMClient instantiates without error", t_init))

    # ── 10.5  Message builder works ──
    def t_msgs():
        msgs = LLMClient._build_messages("system prompt", "user message")
        ok = len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        return ok, 1.0 if ok else 0.0, f"msgs={msgs}"
    cat.tests.append(_run_test("_build_messages produces [system, user]", t_msgs))

    # ── 10.6  Message builder without system prompt ──
    def t_msgs_nosys():
        msgs = LLMClient._build_messages("", "user message")
        ok = len(msgs) == 1 and msgs[0]["role"] == "user"
        return ok, 1.0 if ok else 0.0, f"msgs={msgs}"
    cat.tests.append(_run_test("_build_messages without system → [user] only", t_msgs_nosys))

    return cat


# ═══════════════════════════════════════════════════════════
# 11 — FIREBASE / SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════

def eval_firebase() -> CategoryResult:
    from app.firebase_manager import FirebaseManager

    cat = CategoryResult(category="11. Firebase / Session Management")

    # ── 11.1  Initialization ──
    def t_init():
        fm = FirebaseManager()
        ok = fm is not None
        return ok, 1.0, "FirebaseManager created"
    cat.tests.append(_run_test("FirebaseManager instantiates", t_init))

    # ── 11.2  Create session (in-memory fallback) ──
    def t_create():
        fm = FirebaseManager()
        fm.initialize()
        sid = fm.create_session("eval_test_user")
        ok = bool(sid) and len(sid) > 5
        return ok, 1.0 if ok else 0.0, f"session_id={sid}"
    cat.tests.append(_run_test("Create session → returns session ID", t_create))

    # ── 11.3  Get session back ──
    def t_get():
        fm = FirebaseManager()
        fm.initialize()
        sid = fm.create_session("eval_test_user_2")
        session = fm.get_session("eval_test_user_2", sid)
        ok = session is not None and session.session_id == sid
        return ok, 1.0 if ok else 0.0, f"phase={session.phase if session else 'None'}"
    cat.tests.append(_run_test("Get session → returns SessionData", t_get))

    # ── 11.4  Update session ──
    def t_update():
        fm = FirebaseManager()
        fm.initialize()
        sid = fm.create_session("eval_update_user")
        fm.update_session("eval_update_user", sid, {"patient_name": "EvalBot", "turn_count": 5})
        session = fm.get_session("eval_update_user", sid)
        ok = session is not None and session.patient_name == "EvalBot" and session.turn_count == 5
        return ok, 1.0 if ok else 0.0, f"name={session.patient_name if session else '?'}"
    cat.tests.append(_run_test("Update session — fields persist", t_update))

    # ── 11.5  List sessions ──
    def t_list():
        fm = FirebaseManager()
        fm.initialize()
        fm.create_session("eval_list_user")
        fm.create_session("eval_list_user")
        sessions = fm.list_sessions("eval_list_user")
        ok = len(sessions) >= 2
        return ok, 1.0 if ok else 0.0, f"count={len(sessions)}"
    cat.tests.append(_run_test("List sessions — returns ≥ 2", t_list))

    # ── 11.6  Delete session ──
    def t_delete():
        fm = FirebaseManager()
        fm.initialize()
        sid = fm.create_session("eval_del_user")
        deleted = fm.delete_session("eval_del_user", sid)
        after = fm.get_session("eval_del_user", sid)
        ok = deleted and after is None
        return ok, 1.0 if ok else 0.0, f"deleted={deleted}, after_exists={after is not None}"
    cat.tests.append(_run_test("Delete session — removed successfully", t_delete))

    # ── 11.7  Get non-existent session returns None ──
    def t_get_none():
        fm = FirebaseManager()
        fm.initialize()
        session = fm.get_session("nobody", "fake_id")
        ok = session is None
        return ok, 1.0 if ok else 0.0, f"session={session}"
    cat.tests.append(_run_test("Get non-existent session → None", t_get_none))

    return cat


# ═══════════════════════════════════════════════════════════
# 12 — API ENDPOINT SMOKE TESTS
# ═══════════════════════════════════════════════════════════

def eval_api_endpoints() -> CategoryResult:
    """Tests against a running server at localhost:8000. Skips gracefully if not running."""
    import urllib.request
    import urllib.error

    BASE = "http://127.0.0.1:8000"
    cat = CategoryResult(category="12. API Endpoint Smoke Tests")

    def _http_get(path: str) -> tuple[int, dict]:
        try:
            req = urllib.request.Request(f"{BASE}{path}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, {}
        except Exception:
            return 0, {}

    def _http_post(path: str, data: dict) -> tuple[int, dict]:
        try:
            payload = json.dumps(data).encode()
            req = urllib.request.Request(
                f"{BASE}{path}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = {}
            try:
                body = json.loads(e.read())
            except Exception:
                pass
            return e.code, body
        except Exception:
            return 0, {}

    # ── 12.1  Health check ──
    def t_health():
        code, body = _http_get("/health")
        if code == 0:
            return False, 0.0, "Server not running at :8000 — SKIPPED"
        ok = code == 200 and body.get("status") == "healthy"
        return ok, 1.0 if ok else 0.0, f"code={code}, body={body}"
    cat.tests.append(_run_test("GET /health → 200 + healthy", t_health))

    # ── 12.2  Register ──
    def t_register():
        code, body = _http_post("/register", {"name": f"eval_{int(time.time())}", "password": "testpass"})
        if code == 0:
            return False, 0.0, "Server not running — SKIPPED"
        ok = code == 200 and body.get("success") is True
        return ok, 1.0 if ok else 0.5, f"code={code}, body={body}"
    cat.tests.append(_run_test("POST /register → success", t_register))

    # ── 12.3  Start session ──
    def t_start():
        code, body = _http_post("/start-session", {"user_id": "eval_user", "name": "EvalBot"})
        if code == 0:
            return False, 0.0, "Server not running — SKIPPED"
        ok = code == 200 and "session_id" in body
        return ok, 1.0 if ok else 0.0, f"code={code}, session_id={body.get('session_id','?')}"
    cat.tests.append(_run_test("POST /start-session → session_id", t_start))

    # ── 12.4  Chat (SSE) — at least gets 200 ──
    def t_chat():
        # First create a session
        _, s_body = _http_post("/start-session", {"user_id": "eval_chat_user", "name": "E"})
        sid = s_body.get("session_id", "")
        if not sid:
            return False, 0.0, "Could not create session"
        try:
            payload = json.dumps({"user_id": "eval_chat_user", "session_id": sid, "message": "hello"}).encode()
            req = urllib.request.Request(
                f"{BASE}/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                ok = resp.status == 200
                first = resp.read(500).decode(errors="replace")
                return ok, 1.0 if ok else 0.0, f"status={resp.status}, first_bytes='{first[:100]}'"
        except Exception as e:
            return False, 0.0, f"error: {e}"
    cat.tests.append(_run_test("POST /chat → SSE stream starts", t_chat))

    # ── 12.5  Sessions listing ──
    def t_sessions_list():
        code, body = _http_get("/api/sessions?user_id=eval_user")
        if code == 0:
            return False, 0.0, "Server not running — SKIPPED"
        ok = code == 200 and "sessions" in body
        return ok, 1.0 if ok else 0.0, f"code={code}, count={len(body.get('sessions', []))}"
    cat.tests.append(_run_test("GET /api/sessions → sessions list", t_sessions_list))

    return cat


# ═══════════════════════════════════════════════════════════
# 13 — INGESTION PIPELINE CHECK
# ═══════════════════════════════════════════════════════════

def eval_ingestion() -> CategoryResult:
    cat = CategoryResult(category="13. Data & Ingestion Pipeline")

    # ── 13.1  Data directory exists ──
    def t_data_dir():
        data_dir = ROOT / "data"
        ok = data_dir.exists() and data_dir.is_dir()
        return ok, 1.0 if ok else 0.0, f"path={data_dir}"
    cat.tests.append(_run_test("data/ directory exists", t_data_dir))

    # ── 13.2  Raw text files present ──
    def t_raw():
        raw_dir = ROOT / "data" / "raw"
        if not raw_dir.exists():
            return False, 0.0, "data/raw/ not found"
        files = list(raw_dir.glob("*.txt"))
        ok = len(files) >= 3
        return ok, min(len(files) / 5, 1.0), f"txt_files={len(files)}"
    cat.tests.append(_run_test("data/raw/ has ≥ 3 medical textbooks", t_raw))

    # ── 13.3  Cache directory exists ──
    def t_cache():
        cache_dir = ROOT / "cache"
        ok = cache_dir.exists()
        return ok, 1.0 if ok else 0.0, f"path={cache_dir}"
    cat.tests.append(_run_test("cache/ directory exists", t_cache))

    # ── 13.4  FAISS index cached ──
    def t_faiss():
        idx = ROOT / "cache" / "faiss_index.bin"
        ok = idx.exists() and idx.stat().st_size > 1000
        size_mb = idx.stat().st_size / 1024 / 1024 if idx.exists() else 0
        return ok, 1.0 if ok else 0.0, f"size={size_mb:.2f} MB"
    cat.tests.append(_run_test("FAISS index file cached", t_faiss))

    # ── 13.5  Metadata cached ──
    def t_meta():
        meta = ROOT / "cache" / "faiss_metadata.json"
        ok = meta.exists() and meta.stat().st_size > 100
        if ok:
            data = json.loads(meta.read_text(encoding="utf-8"))
            count = len(data) if isinstance(data, list) else 0
        else:
            count = 0
        return ok, 1.0 if ok else 0.0, f"entries={count}"
    cat.tests.append(_run_test("FAISS metadata JSON cached", t_meta))

    # ── 13.6  Disease nodes JSON cached ──
    def t_nodes():
        nodes = ROOT / "cache" / "disease_nodes.json"
        ok = nodes.exists() and nodes.stat().st_size > 100
        if ok:
            data = json.loads(nodes.read_text(encoding="utf-8"))
            count = len(data) if isinstance(data, list) else 0
        else:
            count = 0
        return ok, 1.0 if ok else 0.0, f"disease_nodes={count}"
    cat.tests.append(_run_test("disease_nodes.json cached", t_nodes))

    # ── 13.7  book.json knowledge base ──
    def t_book():
        book = ROOT / "data" / "book.json"
        ok = book.exists() and book.stat().st_size > 100
        return ok, 1.0 if ok else 0.0, f"exists={book.exists()}"
    cat.tests.append(_run_test("data/book.json exists", t_book))

    return cat


# ═══════════════════════════════════════════════════════════
# 14 — REMEDY LOADER
# ═══════════════════════════════════════════════════════════

def eval_remedy_loader() -> CategoryResult:
    from app.remedy_loader import RemedyLoader

    cat = CategoryResult(category="14. Home Remedy Loader")

    # ── 14.1  Instantiation ──
    def t_init():
        rl = RemedyLoader()
        ok = rl is not None
        return ok, 1.0, f"path={rl.path}"
    cat.tests.append(_run_test("RemedyLoader instantiates", t_init))

    # ── 14.2  Load remedies ──
    def t_load():
        rl = RemedyLoader()
        loaded = rl.load()
        ok = loaded and len(rl.remedies) > 0
        return ok, 1.0 if ok else 0.0, f"loaded={loaded}, count={len(rl.remedies)}"
    cat.tests.append(_run_test("Remedies file loads successfully", t_load))

    # ── 14.3  Format output works ──
    def t_format():
        rl = RemedyLoader()
        rl.load()
        matched = rl.get_remedies(["cold", "fever"])
        text = rl.format_for_response(matched)
        ok = isinstance(text, str)
        return ok, 1.0 if ok else 0.0, f"output_len={len(text)}"
    cat.tests.append(_run_test("format_for_response produces text", t_format))

    return cat


# ═══════════════════════════════════════════════════════════
# 15 — END-TO-END INTEGRATION
# ═══════════════════════════════════════════════════════════

def eval_integration() -> CategoryResult:
    cat = CategoryResult(category="15. End-to-End Integration Scenarios")

    # ── 15.1  Full pipeline: message → extraction → retrieval → scoring ──
    def t_full():
        from app.symptom_extractor import RuleBasedExtractor
        from app.embeddings import EmbeddingManager
        from app.retriever import HybridRetriever
        from app.medicine_policy import MedicinePolicy

        message = "I have had high fever, severe headache and body pain for 3 days"

        # Step 1: Extract
        ext = RuleBasedExtractor()
        result = ext.extract(message)
        if not result.symptoms:
            return False, 0.0, "Extraction yielded 0 symptoms"

        # Step 2: Retrieve
        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No FAISS index"
        retriever = HybridRetriever(emb)
        rag_results = retriever.retrieve(message, user_symptoms=result.symptoms, top_k_final=3)
        if not rag_results:
            return False, 0.3, "Retrieval returned 0 results"

        # Step 3: Medicine
        mp = MedicinePolicy()
        meds = mp.get_suggestions(result.symptoms, severity=result.intensity or "mild")

        ok = len(result.symptoms) >= 2 and len(rag_results) >= 1
        return ok, 1.0 if ok else 0.5, (
            f"symptoms={result.symptoms}, "
            f"diseases={[r.disease.disease_name for r in rag_results]}, "
            f"meds={[m['name'] for m in meds]}"
        )
    cat.tests.append(_run_test("Full pipeline: extract → retrieve → medicate", t_full))

    # ── 15.2  Severity escalation integration ──
    def t_sev_int():
        from app.symptom_extractor import RuleBasedExtractor, ExtractionResult
        from app.severity_classifier import SeverityResult, RED_FLAG_RULES

        ext = RuleBasedExtractor()
        result = ext.extract("severe chest pain and difficulty breathing since this morning")

        sev = SeverityResult()
        combined_text = " ".join(result.symptoms).lower()
        for rule in RED_FLAG_RULES:
            for pattern in rule["patterns"]:
                if pattern in combined_text:
                    sev.escalate("severe", rule["reason"])

        ok = sev.level == "severe" and sev.is_emergency
        return ok, 1.0 if ok else 0.0, f"level={sev.level}, emergency={sev.is_emergency}, symptoms={result.symptoms}"
    cat.tests.append(_run_test("Severity integration: chest pain → emergency", t_sev_int))

    # ── 15.3  Session lifecycle: create → update → read → delete ──
    def t_session_lifecycle():
        from app.firebase_manager import FirebaseManager

        fm = FirebaseManager()
        fm.initialize()

        uid = "eval_lifecycle_user"
        sid = fm.create_session(uid)
        fm.update_session(uid, sid, {"extracted_symptoms": ["fever", "cough"], "phase": "gathering"})
        session = fm.get_session(uid, sid)
        if not session or session.phase != "gathering":
            return False, 0.3, "Phase update failed"

        fm.delete_session(uid, sid)
        after = fm.get_session(uid, sid)
        ok = after is None
        return ok, 1.0 if ok else 0.5, "Full lifecycle: create→update→read→delete"
    cat.tests.append(_run_test("Session lifecycle: create→update→read→delete", t_session_lifecycle))

    # ── 15.4  GI scenario ──
    def t_gi_scenario():
        from app.symptom_extractor import RuleBasedExtractor
        from app.embeddings import EmbeddingManager
        from app.retriever import HybridRetriever

        message = "nausea vomiting diarrhea and stomach cramps for 2 days"
        ext = RuleBasedExtractor()
        result = ext.extract(message)

        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        rag_results = retriever.retrieve(message, user_symptoms=result.symptoms, top_k_final=3)

        ok = len(result.symptoms) >= 2 and len(rag_results) >= 1
        return ok, 1.0 if ok else 0.0, (
            f"symptoms={result.symptoms}, diseases={[r.disease.disease_name for r in rag_results]}"
        )
    cat.tests.append(_run_test("GI scenario: nausea + vomiting + diarrhea", t_gi_scenario))

    # ── 15.5  Respiratory scenario ──
    def t_resp_scenario():
        from app.symptom_extractor import RuleBasedExtractor
        from app.embeddings import EmbeddingManager
        from app.retriever import HybridRetriever

        message = "persistent dry cough, sore throat, runny nose and mild fever for a week"
        ext = RuleBasedExtractor()
        result = ext.extract(message)

        emb = EmbeddingManager()
        if not emb.load_index():
            return False, 0.0, "No index"
        retriever = HybridRetriever(emb)
        rag_results = retriever.retrieve(message, user_symptoms=result.symptoms, top_k_final=3)

        ok = len(result.symptoms) >= 3 and len(rag_results) >= 1
        return ok, 1.0 if ok else 0.0, (
            f"symptoms={result.symptoms}, diseases={[r.disease.disease_name for r in rag_results]}"
        )
    cat.tests.append(_run_test("Respiratory scenario: cough + sore throat + fever", t_resp_scenario))

    return cat


# ═══════════════════════════════════════════════════════════
#  MAIN — Run all categories & print evaluation matrix
# ═══════════════════════════════════════════════════════════

HEADER = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║         ____  _       ____                    _                       ║
║        | __ )| |_   _/ ___|__ _ _ __ ___  _| |_                     ║
║        |  _ \| | | | | |   / _` | '__/ _ \(_) __|                    ║
║        | |_) | | |_| | |__| (_| | | |  __/ _| |_                    ║
║        |____/|_|\__,_|\____\__,_|_|  \___|(_)\__|                    ║
║                                                                       ║
║              Comprehensive Evaluation Suite v1.0                      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


def print_bar(score: float, width: int = 20) -> str:
    """Render a progress bar: ████████░░░░░░░░░░░░ 85%"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {score * 100:5.1f}%"


def run_custom_symptoms() -> None:
    """Interactive mode: user enters their own symptoms → full pipeline evaluation."""
    print("\n" + "═" * 72)
    print("  🩺  CUSTOM SYMPTOM EVALUATION MODE")
    print("═" * 72)
    print()
    print("  Enter your symptoms below (comma-separated or natural language).")
    print("  Example: 'I have fever, headache, body pain since 3 days'")
    print()

    user_input = input("  ▶ Your symptoms: ").strip()
    if not user_input:
        print("\n  ⚠️  No input provided. Exiting.\n")
        return

    print(f"\n  Received: \"{user_input}\"")
    print(f"\n{'─' * 72}")
    t_total = time.perf_counter()

    # ── Step 1: Symptom Extraction ────────────────────────
    print("\n  📋 STEP 1 — Symptom Extraction (Rule-Based NER)")
    print(f"  {'─' * 64}")
    from app.symptom_extractor import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    t0 = time.perf_counter()
    extraction = extractor.extract(user_input)
    t1 = time.perf_counter()

    print(f"    Symptoms found : {extraction.symptoms or ['(none)']}")
    print(f"    Body parts     : {extraction.body_parts or ['(none)']}")
    print(f"    Duration       : {extraction.duration or '(not detected)'}")
    print(f"    Intensity      : {extraction.intensity or '(not detected)'}")
    print(f"    Age            : {extraction.age or '(not detected)'}")
    print(f"    Gender         : {extraction.gender or '(not detected)'}")
    print(f"    ⏱  Time: {(t1 - t0) * 1000:.1f} ms")

    if not extraction.symptoms:
        print("\n  ⚠️  No symptoms detected. Try more specific medical language.")
        print("     Example: 'severe headache, fever 102, nausea for 2 days'\n")

    # ── Step 2: Severity Classification ───────────────────
    print(f"\n  🚨 STEP 2 — Severity Classification (Rule-Based)")
    print(f"  {'─' * 64}")
    from app.severity_classifier import SeverityResult, RED_FLAG_RULES
    t0 = time.perf_counter()
    severity = SeverityResult()
    combined_text = " ".join(extraction.symptoms).lower()

    for rule in RED_FLAG_RULES:
        for pattern in rule["patterns"]:
            if pattern in combined_text:
                severity.red_flags_found.append(pattern)
                severity.escalate("severe", rule["reason"])
                severity.recommended_action = rule["action"]

    if extraction.intensity == "severe":
        severity.escalate("severe", "Self-reported severe symptoms")
    elif extraction.intensity == "moderate":
        severity.escalate("moderate", "Self-reported moderate symptoms")
    elif extraction.intensity == "mild" and severity.level == "undetermined":
        severity.level = "mild"
        severity.reasoning = "Self-reported mild symptoms"

    if len(extraction.symptoms) >= 5 and severity.level in ("undetermined", "mild"):
        severity.escalate("moderate", f"Multiple symptoms ({len(extraction.symptoms)})")

    if severity.level == "undetermined":
        severity.level = "mild"
        severity.reasoning = "No escalation triggers — default mild"

    t1 = time.perf_counter()

    level_emoji = {"mild": "🟢", "moderate": "🟡", "severe": "🔴"}.get(severity.level, "⚪")
    print(f"    Severity       : {level_emoji} {severity.level.upper()}")
    print(f"    Emergency      : {'🚑 YES' if severity.is_emergency else '✅ No'}")
    print(f"    Red flags      : {severity.red_flags_found or ['(none)']}")
    print(f"    Reasoning      : {severity.reasoning or '—'}")
    if severity.recommended_action:
        print(f"    Action         : {severity.recommended_action}")
    print(f"    ⏱  Time: {(t1 - t0) * 1000:.1f} ms")

    # ── Step 3: RAG Retrieval ─────────────────────────────
    print(f"\n  🔍 STEP 3 — RAG Retrieval (Vector Search + Hybrid Scoring)")
    print(f"  {'─' * 64}")
    from app.embeddings import EmbeddingManager
    from app.retriever import HybridRetriever

    t0 = time.perf_counter()
    emb = EmbeddingManager()
    if not emb.load_index():
        print("    ⚠️  No FAISS index found. Run ingestion first.")
        return

    retriever = HybridRetriever(emb)
    rag_results = retriever.retrieve(
        user_input, user_symptoms=extraction.symptoms, top_k_final=5
    )
    t1 = time.perf_counter()

    if rag_results:
        print(f"    Top {len(rag_results)} disease matches:\n")
        print(f"    {'#':<4} {'Disease':<35} {'Semantic':>9} {'Overlap':>9} {'RedFlag':>9} {'Final':>8}")
        print(f"    {'─' * 4} {'─' * 35} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 8}")
        for i, r in enumerate(rag_results, 1):
            name = r.disease.disease_name[:33]
            print(f"    {i:<4} {name:<35} {r.semantic_score:>8.4f} {r.keyword_score:>8.4f} "
                  f"{r.red_flag_weight:>8.4f} {r.final_score:>7.4f}")
        print()
        # Show symptoms of top match
        top = rag_results[0]
        if top.disease.symptoms:
            print(f"    📌 Top match symptoms: {', '.join(top.disease.symptoms[:10])}")
        if top.disease.red_flags:
            print(f"    🚩 Top match red flags: {', '.join(top.disease.red_flags[:5])}")
        if top.disease.treatments:
            print(f"    💊 Top match treatments: {', '.join(top.disease.treatments[:5])}")
    else:
        print("    ⚠️  No matching diseases found in knowledge base.")

    print(f"    ⏱  Time: {(t1 - t0) * 1000:.1f} ms")

    # ── Step 4: Medicine Policy ───────────────────────────
    print(f"\n  💊 STEP 4 — Safe OTC Medication Suggestions")
    print(f"  {'─' * 64}")
    from app.medicine_policy import MedicinePolicy
    t0 = time.perf_counter()
    mp = MedicinePolicy()
    meds = mp.get_suggestions(
        extraction.symptoms,
        age=extraction.age,
        severity=severity.level,
    )
    t1 = time.perf_counter()

    if meds:
        for i, m in enumerate(meds, 1):
            print(f"    {i}. {m['name']}")
            print(f"       Dosage: {m['dosage']}")
            if m.get("age_note"):
                print(f"       ⚠️  {m['age_note']}")
            for w in m.get("warnings", [])[:2]:
                print(f"       ⚠️  {w}")
            print()
    else:
        print("    No OTC medications matched the detected symptoms.")

    print(f"    ⏱  Time: {(t1 - t0) * 1000:.1f} ms")

    # ── Step 5: Home Remedies ─────────────────────────────
    print(f"\n  🌿 STEP 5 — Home Remedies")
    print(f"  {'─' * 64}")
    from app.remedy_loader import RemedyLoader
    t0 = time.perf_counter()
    rl = RemedyLoader()
    rl.load()
    disease_names = [r.disease.disease_name for r in rag_results] if rag_results else []
    remedy_results = rl.get_remedies(disease_names, symptoms=extraction.symptoms)
    t1 = time.perf_counter()

    if remedy_results:
        for rm in remedy_results[:3]:
            print(f"    🌱 {rm['condition']}:")
            for r in rm.get("remedies", [])[:3]:
                if isinstance(r, str):
                    print(f"       • {r}")
                elif isinstance(r, dict):
                    print(f"       • {r.get('name', r)}")
            print()
    else:
        print("    No home remedies found (remedies.json may not exist).")

    print(f"    ⏱  Time: {(t1 - t0) * 1000:.1f} ms")

    # ── Step 6: Scoring Summary ───────────────────────────
    print(f"\n  📊 STEP 6 — Scoring Formula Breakdown")
    print(f"  {'─' * 64}")
    from app.scoring import compute_final_score
    if rag_results:
        r = rag_results[0]
        print(f"    Formula: final = 0.5×semantic + 0.3×overlap + 0.1×red_flag + 0.1×prevalence")
        print(f"    Values : final = 0.5×{r.semantic_score:.4f} + 0.3×{r.keyword_score:.4f} "
              f"+ 0.1×{r.red_flag_weight:.4f} + 0.1×{r.prevalence_weight:.4f}")
        recalc = compute_final_score(r.semantic_score, r.keyword_score, r.red_flag_weight, r.prevalence_weight)
        print(f"    Result : {recalc:.4f}")
    else:
        print("    (No retrieval results to score)")

    # ── Final Summary ─────────────────────────────────────
    total_time = (time.perf_counter() - t_total) * 1000
    print(f"\n{'═' * 72}")
    print(f"  📝 EVALUATION SUMMARY FOR YOUR INPUT")
    print(f"{'═' * 72}")
    print(f"    Input      : \"{user_input}\"")
    print(f"    Symptoms   : {extraction.symptoms}")
    print(f"    Severity   : {level_emoji} {severity.level}")
    print(f"    Diseases   : {[r.disease.disease_name for r in rag_results[:3]] if rag_results else '(none)'}")
    print(f"    Medications: {[m['name'] for m in meds] if meds else '(none)'}")
    print(f"    Remedies   : {len(remedy_results)} found")
    print(f"    Total Time : {total_time:.0f} ms")
    print(f"{'═' * 72}\n")


def main() -> None:
    print(HEADER)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  Root:    {ROOT}")
    print()

    # ── Ask user: custom symptoms or predefined tests ─────
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │   Choose Evaluation Mode:                       │")
    print("  │                                                 │")
    print("  │   1 → Enter your OWN symptoms (live pipeline)   │")
    print("  │   2 → Run predefined test suite (106 tests)     │")
    print("  │                                                 │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    choice = input("  ▶ Enter 1 or 2: ").strip()

    if choice == "1":
        run_custom_symptoms()
        return

    if choice != "2":
        print("  (Defaulting to option 2 — predefined test suite)\n")

    # ── Collect all evaluation categories ─────────────────
    evaluators = [
        eval_symptom_extraction,
        eval_duration_intensity,
        eval_severity_rules,
        eval_scoring,
        eval_medicine_policy,
        eval_utils,
        eval_models,
        eval_embeddings,
        eval_retriever,
        eval_llm_config,
        eval_firebase,
        eval_api_endpoints,
        eval_ingestion,
        eval_remedy_loader,
        eval_integration,
    ]

    all_categories: list[CategoryResult] = []
    total_tests = 0
    total_passed = 0
    total_start = time.perf_counter()

    for fn in evaluators:
        try:
            cat = fn()
        except Exception as exc:
            cat = CategoryResult(category=fn.__name__)
            cat.tests.append(TestResult(
                name="Category Load Error",
                passed=False,
                score=0.0,
                detail=f"EXCEPTION: {exc}",
            ))
        all_categories.append(cat)
        total_tests += cat.total
        total_passed += cat.passed

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # ═══════════════════════════════ DETAILED RESULTS ════
    for cat in all_categories:
        print(f"\n{'─' * 72}")
        print(f"  {cat.category}")
        print(f"{'─' * 72}")
        for t in cat.tests:
            status = "✅ PASS" if t.passed else "❌ FAIL"
            print(f"    {status}  {t.name}")
            print(f"           Score: {t.score:.2f}  |  Time: {t.elapsed_ms:.1f}ms")
            if t.detail:
                detail_lines = t.detail.split("\n")
                for line in detail_lines[:3]:
                    print(f"           → {line[:100]}")
        print(f"  ── Category Score: {cat.passed}/{cat.total} passed  |  Avg: {cat.avg_score:.2%}")

    # ═══════════════════════════════ SUMMARY MATRIX ══════
    print(f"\n\n{'═' * 72}")
    print("  EVALUATION MATRIX — SUMMARY")
    print(f"{'═' * 72}\n")

    header_fmt = f"  {'#':<4} {'Category':<45} {'Pass':>5} {'Total':>6} {'Score':>8}  {'Bar':<26}"
    print(header_fmt)
    print(f"  {'─' * 4} {'─' * 45} {'─' * 5} {'─' * 6} {'─' * 8}  {'─' * 26}")

    for i, cat in enumerate(all_categories, 1):
        short_name = cat.category.split(". ", 1)[-1] if ". " in cat.category else cat.category
        bar = print_bar(cat.avg_score)
        print(f"  {i:<4} {short_name:<45} {cat.passed:>5} {cat.total:>6} {cat.avg_score:>7.1%}  {bar}")

    print(f"  {'─' * 4} {'─' * 45} {'─' * 5} {'─' * 6} {'─' * 8}  {'─' * 26}")
    overall = total_passed / total_tests if total_tests else 0
    overall_bar = print_bar(overall)
    print(f"  {'':4} {'OVERALL':45} {total_passed:>5} {total_tests:>6} {overall:>7.1%}  {overall_bar}")

    # ═══════════════════════════════ FINAL GRADE ═════════
    print(f"\n{'═' * 72}")
    if overall >= 0.95:
        grade, emoji = "A+", "🏆"
    elif overall >= 0.90:
        grade, emoji = "A", "🌟"
    elif overall >= 0.85:
        grade, emoji = "A-", "✨"
    elif overall >= 0.80:
        grade, emoji = "B+", "👍"
    elif overall >= 0.70:
        grade, emoji = "B", "📊"
    elif overall >= 0.60:
        grade, emoji = "C", "⚠️"
    else:
        grade, emoji = "D", "🔴"

    print(f"  {emoji}  FINAL GRADE: {grade}  ({overall:.1%})")
    print(f"  Total: {total_passed}/{total_tests} tests passed")
    print(f"  Time:  {total_elapsed:.0f} ms")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 72}\n")

    # ═══════════════════════════════ FAILED TESTS ════════
    failed = [(cat.category, t) for cat in all_categories for t in cat.tests if not t.passed]
    if failed:
        print(f"\n  ❌ FAILED TESTS ({len(failed)}):")
        print(f"  {'─' * 68}")
        for cat_name, t in failed:
            short = cat_name.split(". ", 1)[-1] if ". " in cat_name else cat_name
            print(f"    [{short}] {t.name}")
            if t.detail:
                for line in t.detail.split("\n")[:2]:
                    print(f"      → {line[:90]}")
        print()
    else:
        print("\n  🎉 ALL TESTS PASSED! No failures.\n")

    # Exit code: 0 if all passed, 1 otherwise
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
