"""
types.py - data structures for de-identification.

PhiSpan offsets point into the ORIGINAL report text and use the same type
names as data/truth/phi, so scoring compares like with like.
MaskedText and DeidResult are the only objects common/spark.py will accept;
they are built inside deid/ and carry an HMAC seal (filled in on Day 2).
"""
from dataclasses import asdict, dataclass, field

PHI_TYPES = (
    "NAME", "ID", "DOB", "DATE", "MRN", "ACCESSION",
    "DOCTOR", "HOSPITAL", "PHONE", "ADDRESS", "EMAIL", "ACCOUNT",
)
STATUSES = ("proposed", "removed", "kept")  # kept = user unmarked a false alarm


@dataclass(frozen=True)
class PhiSpan:
    id: str
    type: str
    start: int
    end: int
    text: str            # original text: contains PHI, never leaves the laptop
    source: str          # "pattern:<recogniser>" | "spark" | "user" | "profile"
    score: float = 1.0
    status: str = "proposed"

    def __post_init__(self):
        if self.type not in PHI_TYPES:
            raise ValueError(f"unknown PHI type {self.type!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if not 0 <= self.start < self.end:
            raise ValueError(f"bad span {self.start}-{self.end}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PhiSpan":
        return cls(**d)


@dataclass(frozen=True)
class TagSpan:
    """One replacement: where a surrogate sits in the masked or de-identified text."""
    id: str
    type: str
    source: str
    surrogate: str       # e.g. "[PATIENT-A]", "[ID-1]"
    orig_start: int
    orig_end: int
    start: int           # offsets in the output text
    end: int


@dataclass(frozen=True)
class MaskedText:
    """Pattern hits replaced by type tags; names may still be present. Input to the name pass."""
    deid_id: str
    text: str
    tags: tuple[TagSpan, ...]
    seal: str = ""


@dataclass(frozen=True)
class DeidResult:
    deid_id: str
    report_id: str | None
    status: str          # "proposed" | "confirmed"
    text: str            # surrogate text (empty until confirmed)
    tags: tuple[TagSpan, ...]
    summary: dict
    profile_version: int
    vault_ref: str | None = None
    seal: str = ""

    def to_dict(self) -> dict:
        """Public view: contains no original identifier text."""
        return asdict(self)


@dataclass
class PrivacyProfile:
    version: int = 1
    remove: dict = field(default_factory=lambda: {k: v[3] for k, v in CHECKLIST.items()})
    replace_with_tags: bool = True
    shift_dates: bool = False
    allow_list: list[str] = field(default_factory=list)
    always_remove: list[str] = field(default_factory=list)


# First-time checklist: key -> (label, PHI types it covers, warn if unticked, ticked by default).
# Clinical dates (collected, reported, diagnosis) are kept by default; only DOB is masked.
CHECKLIST = {
    "patient_names": ("Patient names", ("NAME",), True, True),
    "relative_names": ("Names of relatives and others", ("NAME",), True, True),
    "doctor_names": ("Doctor names", ("DOCTOR",), False, True),
    "national_ids": ("NRIC/FIN, NHS and SSN numbers", ("ID",), True, True),
    "dob": ("Date of birth", ("DOB",), True, True),
    "other_dates": ("Other dates (collected, reported, diagnosis)", ("DATE",), False, False),
    "mrn_accession": ("MRN, accession and account numbers", ("MRN", "ACCESSION", "ACCOUNT"), False, True),
    "phone_email_address": ("Phone, email and address", ("PHONE", "EMAIL", "ADDRESS"), True, True),
    "hospital": ("Hospital name", ("HOSPITAL",), False, True),
}
