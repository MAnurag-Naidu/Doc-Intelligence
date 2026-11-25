import re
from src.schema import Finding

# a number like 11.8, 9,000, <5, .00
NUM_RE = re.compile(r"^\s*([<>]?\s*\d[\d,]*(?:\.\d+)?|\.\d+)\s*(.*)$")
# a range like 40-70, 4000-11000, .00-01
RANGE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?|\.\d+)\s*-\s*(\d[\d,]*(?:\.\d+)?|\.\d+)")


def to_float(text):
    return float(text.replace(",", "").replace("<", "").replace(">", "").strip())


def looks_like_range(text):
    return bool(text and RANGE_RE.search(text))


def clean_unit(unit):
    """Fix small OCR slips in units: '/%' -> '%', 'Mill/Cumm..' -> 'Mill/Cumm'."""
    if not unit:
        return None
    unit = unit.strip().rstrip(".").replace("/%", "%")
    return unit or None


def pick_range(ref, sex):
    """For (M)/(F) ranges, pick the one matching the patient's sex."""
    if not ref:
        return None
    if "(M)" in ref or "(F)" in ref:
        if not sex:
            return None                      # sex unknown -> can't decide
        key = "(M)" if sex.strip().upper().startswith("M") else "(F)"
        for part in ref.split(";"):
            if key in part:
                return part
        return None
    return ref


def compute_flag(value, ref, sex):
    part = pick_range(ref, sex)
    if not part:
        return None
    m = RANGE_RE.search(part)
    if not m:
        return None
    try:
        v, low, high = to_float(value), to_float(m.group(1)), to_float(m.group(2))
    except ValueError:
        return None
    if v < low:
        return "low"
    if v > high:
        return "high"
    return "normal"


def postprocess(report):
    tests, findings = [], list(report.other_findings)

    for t in report.tests:
        m = NUM_RE.match(t.value or "")

        # 1. Text results (e.g. "ADEQUATE") are findings, not numeric tests
        if not m:
            findings.append(Finding(name=t.test_name, result=t.value))
            continue

        number, rest = m.group(1).strip(), m.group(2).strip()

        # 2. Fix shifted cells: a range sitting in 'unit'
        if looks_like_range(t.unit) and not t.reference_range:
            t.reference_range, t.unit = t.unit, None

        # 3. Unit stuck to the value: "2.80 Lakhs/cumm"
        if rest and not t.unit:
            t.unit = rest

        t.value = number
        t.unit = clean_unit(t.unit)

        # 4. Calculate the flag in Python (reliable maths)
        t.flag = compute_flag(t.value, t.reference_range, report.sex)
        tests.append(t)

    report.tests, report.other_findings = tests, findings
    return report