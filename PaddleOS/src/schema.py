from typing import List, Literal, Optional
from pydantic import BaseModel


class LabTest(BaseModel):
    test_name: str
    value: str                              # REQUIRED now – model can't skip it
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[Literal["high", "low", "normal"]] = None


class Finding(BaseModel):
    name: str
    result: str


class LabReport(BaseModel):
    report_title: Optional[str] = None
    patient_name: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    report_date: Optional[str] = None
    sample_type: Optional[str] = None
    method: Optional[str] = None
    tests: List[LabTest] = []
    other_findings: List[Finding] = []