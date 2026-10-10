"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real env vars win)

Contract: openapi.yaml (HTTP paths, status codes, business rules R1-R10).
Storage:  DATA_MODEL.md (collections, field types, nine sample documents).

All application code lives in this single file by design.

Design notes
------------
* Instants cross the API boundary as integer epoch *milliseconds*; MongoDB stores
  them as BSON datetimes (UTC, timezone-aware here). We convert at the boundary.
* Calendar dates ("YYYY-MM-DD"), months ("YYYY-MM") and shift times ("HH:MM") are
  plain strings in IST, exactly as stored.
* Derived fields (work_hours, late_minutes, overtime_minutes, half_day) are computed
  by this API and stored; analytics read the stored values, never recomputing them.
* All times are truncated to whole seconds before any calculation (R1).
"""
import os
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()

# ``tz_aware=True`` makes PyMongo return timezone-aware (UTC) datetimes instead of
# naive ones, so our IST conversions are unambiguous.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# Epoch-ms bounds mirrored from openapi.yaml EpochMillis (seconds-looking,
# out-of-range, floats and strings are all rejected with 422).
EPOCH_MS_MIN = 100_000_000_000
EPOCH_MS_MAX = 4_102_444_800_000

PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = {"PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"}

@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    ensure_indexes()
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan
)


# --------------------------------------------------------------------------- #
# Time / rounding helpers
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds -> timezone-aware UTC datetime, truncated to whole
    seconds (R1). Milliseconds are intentionally dropped before any math."""
    whole_seconds = ms // 1000
    return datetime.fromtimestamp(whole_seconds, tz=UTC)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Timezone-aware (or naive-as-UTC) datetime -> epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def ist_date_str(dt: datetime) -> str:
    """IST calendar date (YYYY-MM-DD) of an instant."""
    return dt.astimezone(IST).strftime("%Y-%m-%d")


def round_half_up(value: float, places: int) -> float:
    """Decimal half-up rounding (R4/R8), avoiding float/banker's rounding."""
    quant = Decimal(1).scaleb(-places)  # 10^-places
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def parse_hhmm(hhmm: str) -> tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)


def shift_start_dt(date_str: str, shift_start: str) -> datetime:
    """UTC datetime of the shift start on the record's attendance date (IST)."""
    h, m = parse_hhmm(shift_start)
    naive = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    return naive.replace(tzinfo=IST).astimezone(UTC)


def shift_end_dt(date_str: str, shift_start: str, shift_end: str) -> datetime:
    """UTC datetime of the shift end. For an overnight shift (shift_end <=
    shift_start) the end falls on the next calendar day (R1)."""
    sh, sm = parse_hhmm(shift_start)
    eh, em = parse_hhmm(shift_end)
    base = datetime.strptime(date_str, "%Y-%m-%d")
    overnight = (eh, em) <= (sh, sm)
    if overnight:
        base = base + timedelta(days=1)
    naive = base.replace(hour=eh, minute=em)
    return naive.replace(tzinfo=IST).astimezone(UTC)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return parse_hhmm(shift_end) <= parse_hhmm(shift_start)


def attendance_date_for_punch(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: the record's date is the IST calendar date of punch_in, except for an
    overnight shift where a punch-in earlier (IST) than shift_end belongs to the
    shift that started the previous day."""
    ist = punch_in.astimezone(IST)
    if is_overnight(shift_start, shift_end):
        eh, em = parse_hhmm(shift_end)
        if (ist.hour, ist.minute, ist.second) < (eh, em, 0):
            ist = ist - timedelta(days=1)
    return ist.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Business rules R2-R5
# --------------------------------------------------------------------------- #
def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2: late only if punch_in is strictly more than 10:00 after shift_start;
    then floor(minutes since shift_start), not since the grace period ended."""
    start = shift_start_dt(date_str, shift_start)
    delta = (punch_in - start).total_seconds()
    if delta <= 600:  # within 10:00 grace, inclusive
        return 0
    return int(delta // 60)


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (punch_out - punch_in) seconds / 3600, rounded to 2 dp half-up."""
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: floor(minutes from shift_end to punch_out), counted only if >= 30."""
    end = shift_end_dt(date_str, shift_start, shift_end)
    delta = (punch_out - end).total_seconds()
    if delta < 0:
        return 0
    minutes = int(delta // 60)
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    """R5: half_day true when rounded work_hours < 4.50."""
    return work_hours is not None and work_hours < 4.50


# --------------------------------------------------------------------------- #
# Serialization: stored document -> API response
# --------------------------------------------------------------------------- #
def serialize_history(history: list[dict]) -> list[dict]:
    out = []
    time_fields = {"punch_in", "punch_out"}
    for entry in history or []:
        changes = {}
        for field, change in (entry.get("changes") or {}).items():
            if field in time_fields:
                changes[field] = {
                    "from": dt_to_ms(change.get("from")),
                    "to": dt_to_ms(change.get("to")),
                }
            else:
                changes[field] = {"from": change.get("from"), "to": change.get("to")}
        out.append(
            {
                "at": dt_to_ms(entry.get("at")),
                "by": entry.get("by"),
                "reason": entry.get("reason"),
                "changes": changes,
            }
        )
    return out


def serialize_attendance(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0) or 0,
        "overtime_minutes": doc.get("overtime_minutes", 0) or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": serialize_history(doc.get("history", [])),
    }


def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc.get("created_at")),
    }


# --------------------------------------------------------------------------- #
# Pydantic models (validation = 422 on bad input)
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v

    @field_validator("shift_end")
    @classmethod
    def _distinct_shift(cls, v: str, info) -> str:
        start = info.data.get("shift_start")
        if start is not None and v == start:
            raise ValueError("shift_start must differ from shift_end")
        return v


def validate_epoch_ms(value: Any, field: str) -> int:
    """Enforce EpochMillis: integer (not bool/float/str) within range -> else 422."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(422, f"{field} must be an integer epoch millisecond value")
    if value < EPOCH_MS_MIN or value > EPOCH_MS_MAX:
        raise HTTPException(422, f"{field} is out of the accepted epoch-millisecond range")
    return value


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = None
    status: str = "PRESENT"

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in PRESENCE_STATUSES:
            raise ValueError("status must be one of PRESENT, WFH, ON_DUTY")
        return v


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = None


class RegularizeIn(BaseModel):
    status: Optional[str] = None
    punch_in: Optional[StrictInt] = None
    punch_out: Optional[StrictInt] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @field_validator("status")
    @classmethod
    def _status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALL_STATUSES:
            raise ValueError("invalid status")
        return v


# --------------------------------------------------------------------------- #
# Indexes - created at startup, idempotently
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    # Unique employee code; enforces race-safe creation.
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    # Department filter + emp_code sort for list and analytics joins.
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_empcode"
    )
    # joined_on used by headcount (R9).
    db.employees.create_index([("joined_on", ASCENDING)], name="joined_on")

    # One record per (emp_code, date); enforces race-safe punch-in.
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date"
    )
    # List endpoint: sort by date desc, emp_code asc, filterable by date/status.
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)], name="date_empcode"
    )
    # Analytics by employee+date range (monthly, leaderboard, explain).
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], name="empcode_date"
    )
    # Status filter on the list endpoint.
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING)], name="status_date"
    )


# --------------------------------------------------------------------------- #
# Month / date helpers
# --------------------------------------------------------------------------- #
def month_bounds(month: str) -> tuple[str, str]:
    """Return inclusive (first_day, last_day) date strings for YYYY-MM."""
    year, mon = int(month[:4]), int(month[5:7])
    first = f"{year:04d}-{mon:02d}-01"
    if mon == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, mon + 1, 1)
    last = (nxt - timedelta(days=1)).strftime("%Y-%m-%d")
    return first, last


def validate_month(month: str) -> None:
    import re

    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", month):
        raise HTTPException(422, "month must be YYYY-MM")


def validate_date(value: str, field: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, f"{field} must be YYYY-MM-DD")


def working_days_in_range(start: str, end: str, joined_on: Optional[str]) -> int:
    """R7: count Mon-Fri days in [start, end], not before joined_on."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if joined_on:
        j = datetime.strptime(joined_on, "%Y-%m-%d")
        if j > s:
            s = j
    count = 0
    cur = s
    while cur <= e:
        if cur.weekday() < 5:  # Mon-Fri
            count += 1
        cur += timedelta(days=1)
    return count


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        db.command("ping")
    except Exception:
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Employees
# --------------------------------------------------------------------------- #
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(tz=UTC).replace(microsecond=0)
    try:
        db.employees.insert_one(dict(doc))
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: dict = {}
    if department:
        q["department"] = department
    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.employees.find(q)
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_employee(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance: punch-in
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")

    if body.punched_at is None:
        punched = datetime.now(tz=UTC).replace(microsecond=0)
    else:
        validate_epoch_ms(body.punched_at, "punched_at")
        punched = ms_to_dt(body.punched_at)

    date_str = attendance_date_for_punch(punched, emp["shift_start"], emp["shift_end"])
    late = compute_late_minutes(punched, date_str, emp["shift_start"])

    doc = {
        "emp_code": body.emp_code,
        "date": date_str,
        "status": body.status,
        "punch_in": punched,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(dict(doc))
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return serialize_attendance(doc)


# --------------------------------------------------------------------------- #
# Attendance: punch-out
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")

    if body.punched_at is None:
        punched = datetime.now(tz=UTC).replace(microsecond=0)
    else:
        validate_epoch_ms(body.punched_at, "punched_at")
        punched = ms_to_dt(body.punched_at)

    # Most recent open record whose punch_in is at or before punched_at. Matching
    # on the punch_in instant (not the date) is what makes this work across
    # midnight for overnight shifts.
    candidate = db.attendance_logs.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$ne": None, "$lte": punched},
            "punch_out": None,
        },
        sort=[("punch_in", DESCENDING)],
    )
    if not candidate:
        # No open record at/before punched_at. Decide which error to raise.
        latest_at_or_before = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": punched}},
            sort=[("punch_in", DESCENDING)],
        )
        if latest_at_or_before and latest_at_or_before.get("punch_out") is not None:
            raise HTTPException(409, "record already punched out")
        # An open record exists but its punch_in is AFTER punched_at -> the client
        # is trying to punch out before punching in: that is a 422, not a 404.
        open_after = db.attendance_logs.find_one(
            {
                "emp_code": body.emp_code,
                "punch_in": {"$ne": None, "$gt": punched},
                "punch_out": None,
            },
            sort=[("punch_in", DESCENDING)],
        )
        if open_after:
            raise HTTPException(422, "punched_at must be after punch_in")
        raise HTTPException(404, "no open punch-in found")

    punch_in_dt = candidate["punch_in"]
    delta = (punched - punch_in_dt).total_seconds()
    if delta <= 0:
        raise HTTPException(422, "punched_at must be after punch_in")
    if delta > 24 * 3600:
        raise HTTPException(422, "punched_at must be within 24 hours of punch_in")

    work_hours = compute_work_hours(punch_in_dt, punched)
    overtime = compute_overtime(
        punched, candidate["date"], emp["shift_start"], emp["shift_end"]
    )
    half_day = compute_half_day(work_hours)

    # Atomic close: only succeeds if still open (race-safe for twin punch-outs).
    updated = db.attendance_logs.find_one_and_update(
        {"_id": candidate["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punched,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(409, "record already punched out")
    return serialize_attendance(updated)


# --------------------------------------------------------------------------- #
# Attendance: list
# --------------------------------------------------------------------------- #
@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = build_attendance_query(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_attendance(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


def build_attendance_query(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[str],
) -> dict:
    if date_from:
        validate_date(date_from, "date_from")
    if date_to:
        validate_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    if status and status not in ALL_STATUSES:
        raise HTTPException(422, "invalid status")
    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return q


# --------------------------------------------------------------------------- #
# Attendance: regularize (PATCH)
# --------------------------------------------------------------------------- #
@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    validate_date(date, "date")
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(404, "no record for that date")

    # Validate supplied instants up-front (422 on bad values).
    new_pi = None
    new_po = None
    if body.punch_in is not None:
        validate_epoch_ms(body.punch_in, "punch_in")
        new_pi = ms_to_dt(body.punch_in)
    if body.punch_out is not None:
        validate_epoch_ms(body.punch_out, "punch_out")
        new_po = ms_to_dt(body.punch_out)

    target_status = body.status if body.status is not None else record["status"]

    # Rule: ABSENT / LEAVE clear both punch times; supplying either is 422.
    if target_status in {"ABSENT", "LEAVE"}:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT/LEAVE records cannot carry punch times")
        punch_in_dt = None
        punch_out_dt = None
    else:
        # Presence status: resolve final punch_in / punch_out.
        punch_in_dt = new_pi if body.punch_in is not None else record.get("punch_in")
        punch_out_dt = new_po if body.punch_out is not None else record.get("punch_out")
        # Changing away from ABSENT/LEAVE without a punch_in is invalid.
        if punch_in_dt is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        # punch_in must stay on the record's attendance date (R1).
        if attendance_date_for_punch(
            punch_in_dt, emp["shift_start"], emp["shift_end"]
        ) != date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if punch_out_dt is not None:
            delta = (punch_out_dt - punch_in_dt).total_seconds()
            if delta <= 0:
                raise HTTPException(422, "punch_out must be after punch_in")
            if delta > 24 * 3600:
                raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

    # Recompute derived fields from the final times (R2-R5).
    if target_status in {"ABSENT", "LEAVE"}:
        new_work_hours = None
        new_late = 0
        new_overtime = 0
        new_half = False
    else:
        new_late = compute_late_minutes(punch_in_dt, date, emp["shift_start"])
        if punch_out_dt is not None:
            new_work_hours = compute_work_hours(punch_in_dt, punch_out_dt)
            new_overtime = compute_overtime(
                punch_out_dt, date, emp["shift_start"], emp["shift_end"]
            )
            new_half = compute_half_day(new_work_hours)
        else:
            new_work_hours = None
            new_overtime = 0
            new_half = False

    # Build the final field set and compute what actually changed.
    final = {
        "status": target_status,
        "punch_in": punch_in_dt,
        "punch_out": punch_out_dt,
        "work_hours": new_work_hours,
        "late_minutes": new_late,
        "overtime_minutes": new_overtime,
        "half_day": new_half,
    }
    current = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0) or 0,
        "overtime_minutes": record.get("overtime_minutes", 0) or 0,
        "half_day": bool(record.get("half_day", False)),
    }

    changes: dict[str, dict] = {}
    for field in (
        "status",
        "punch_in",
        "punch_out",
        "work_hours",
        "late_minutes",
        "overtime_minutes",
        "half_day",
    ):
        if current[field] != final[field]:
            changes[field] = {"from": current[field], "to": final[field]}

    if not changes:
        raise HTTPException(422, "the request changes nothing")

    history_entry = {
        "at": datetime.now(tz=UTC).replace(microsecond=0),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Concurrency guard: only apply if the record still looks exactly as we read
    # it (optimistic lock). A concurrent edit changes one of these fields and the
    # update matches nothing -> the loser gets 409. $push preserves prior history.
    expected = {
        "_id": record["_id"],
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "history": record.get("history", []),
    }
    updated = db.attendance_logs.find_one_and_update(
        expected,
        {"$set": final, "$push": {"history": history_entry}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(409, "record was modified concurrently")
    return serialize_attendance(updated)


# --------------------------------------------------------------------------- #
# Analytics: employee monthly
# --------------------------------------------------------------------------- #
def employee_monthly_pipeline(emp_code: str, month: str) -> list[dict]:
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {
            "$addFields": {
                # Day-of-week in IST: date string parsed as UTC midnight, +5:30
                # keeps it on the same calendar day, so isoDayOfWeek is correct.
                "_dow": {
                    "$isoDayOfWeek": {
                        "$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}
                    }
                },
                "_present_unit": {
                    "$cond": [
                        {"$in": ["$status", list(PRESENCE_STATUSES)]},
                        {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                        0,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [{"$lte": ["$_dow", 5]}, "$_present_unit", 0]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(...)):
    validate_month(month)
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "unknown employee")

    first, last = month_bounds(month)
    working_days = working_days_in_range(first, last, emp.get("joined_on"))

    agg = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, month)))
    row = agg[0] if agg else {}
    present_days = round_half_up(float(row.get("present_days", 0) or 0), 2)
    leave_days = int(row.get("leave_days", 0) or 0)
    late_count = int(row.get("late_count", 0) or 0)
    total_late = int(row.get("total_late_minutes", 0) or 0)
    total_ot = int(row.get("total_overtime_minutes", 0) or 0)

    if working_days > 0:
        attendance_pct = round_half_up(present_days / working_days * 100, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


# --------------------------------------------------------------------------- #
# Analytics: department summary
# --------------------------------------------------------------------------- #
def department_summary_pipeline(month: str, department: Optional[str]) -> list[dict]:
    first, last = month_bounds(month)
    emp_match: dict = {"joined_on": {"$lte": last}}  # R9 headcount
    if department:
        emp_match["department"] = department
    present_list = list(PRESENCE_STATUSES)
    return [
        {"$match": emp_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"ec": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$ec"]},
                                    {"$gte": ["$date", first]},
                                    {"$lte": ["$date", last]},
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "_dow": {
                                "$isoDayOfWeek": {
                                    "$dateFromString": {
                                        "dateString": "$date",
                                        "format": "%Y-%m-%d",
                                    }
                                }
                            }
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$addFields": {
                "_present_days": {
                    "$reduce": {
                        "input": "$logs",
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {
                                    "$cond": [
                                        {
                                            "$and": [
                                                {"$in": ["$$this.status", present_list]},
                                                {"$lte": ["$$this._dow", 5]},
                                            ]
                                        },
                                        {
                                            "$cond": [
                                                {"$eq": ["$$this.half_day", True]},
                                                0.5,
                                                1,
                                            ]
                                        },
                                        0,
                                    ]
                                },
                            ]
                        },
                    }
                },
                "_work_sum": {
                    "$reduce": {
                        "input": {
                            "$filter": {
                                "input": "$logs",
                                "as": "l",
                                "cond": {
                                    "$and": [
                                        {"$in": ["$$l.status", present_list]},
                                        {"$ne": ["$$l.work_hours", None]},
                                    ]
                                },
                            }
                        },
                        "initialValue": {"sum": 0, "count": 0},
                        "in": {
                            "sum": {"$add": ["$$value.sum", "$$this.work_hours"]},
                            "count": {"$add": ["$$value.count", 1]},
                        },
                    }
                },
                "_late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]},
                        }
                    }
                },
                "_total_late": {
                    "$reduce": {
                        "input": "$logs",
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {"$ifNull": ["$$this.late_minutes", 0]},
                            ]
                        },
                    }
                },
                "_leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]},
                        }
                    }
                },
                "_on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "ON_DUTY"]},
                        }
                    }
                },
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$_present_days"},
                "work_sum": {"$sum": "$_work_sum.sum"},
                "work_count": {"$sum": "$_work_sum.count"},
                "late_count": {"$sum": "$_late_count"},
                "total_late_minutes": {"$sum": "$_total_late"},
                "leave_count": {"$sum": "$_leave_count"},
                "on_duty_count": {"$sum": "$_on_duty_count"},
            }
        },
        {"$sort": {"_id": 1}},
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(...), department: Optional[str] = None):
    validate_month(month)
    rows = list(db.employees.aggregate(department_summary_pipeline(month, department)))
    items = []
    for r in rows:
        if r["headcount"] == 0:
            continue
        avg = (
            round_half_up(r["work_sum"] / r["work_count"], 2)
            if r["work_count"] > 0
            else None
        )
        items.append(
            {
                "department": r["_id"],
                "headcount": int(r["headcount"]),
                "present_days": round_half_up(float(r["present_days"]), 2),
                "avg_work_hours": avg,
                "late_count": int(r["late_count"]),
                "total_late_minutes": int(r["total_late_minutes"]),
                "leave_count": int(r["leave_count"]),
                "on_duty_count": int(r["on_duty_count"]),
            }
        )
    return {"month": month, "items": items}


# --------------------------------------------------------------------------- #
# Analytics: late leaderboard
# --------------------------------------------------------------------------- #
def late_leaderboard_pipeline(month: str, department: Optional[str]) -> list[dict]:
    first, last = month_bounds(month)
    pipeline: list[dict] = [
        {
            "$match": {
                "date": {"$gte": first, "$lte": last},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$match": {"emp": {"$ne": []}}},  # ignore logs with no employee
        {"$addFields": {"emp": {"$arrayElemAt": ["$emp", 0]}}},
    ]
    if department:
        pipeline.append({"$match": {"emp.department": department}})
    pipeline += [
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {"rank": {"$rank": {}}},  # standard competition rank
            }
        },
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    validate_month(month)
    rows = list(db.attendance_logs.aggregate(late_leaderboard_pipeline(month, department)))
    items = []
    for r in rows:
        if r["rank"] > limit:  # limit applied AFTER ranking (ties may exceed limit)
            continue
        emp = r["emp"]
        items.append(
            {
                "rank": int(r["rank"]),
                "emp_code": r["_id"],
                "name": emp.get("name"),
                "department": emp.get("department"),
                "total_late_minutes": int(r["total_late_minutes"]),
                "late_count": int(r["late_count"]),
            }
        )
    return {"month": month, "items": items}


# --------------------------------------------------------------------------- #
# Analytics: department trend
# --------------------------------------------------------------------------- #
@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
):
    validate_date(date_from, "from")
    validate_date(date_to, "to")
    if date_to < date_from:
        raise HTTPException(422, "to must not be before from")
    span = (
        datetime.strptime(date_to, "%Y-%m-%d")
        - datetime.strptime(date_from, "%Y-%m-%d")
    ).days + 1
    if span > 92:
        raise HTTPException(422, "range exceeds 92 days")

    dept_emps = list(
        db.employees.find({"department": department}, {"emp_code": 1, "joined_on": 1})
    )
    if not dept_emps:
        raise HTTPException(404, "unknown department")

    emp_codes = [e["emp_code"] for e in dept_emps]

    rows = list(
        db.attendance_logs.aggregate(
            build_trend_pipeline(emp_codes, date_from, date_to)
        )
    )
    # rows: one per day in range with present_count & late_count (gap-filled).
    by_date = {r["date"]: r for r in rows}

    # Precompute joined_on sorted for headcount-by-day.
    joined = sorted(
        datetime.strptime(e["joined_on"], "%Y-%m-%d").date() for e in dept_emps
    )

    items = []
    cur = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()
    rates_window: list[Optional[float]] = []
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        is_wd = cur.weekday() < 5
        headcount = sum(1 for j in joined if j <= cur)  # R9
        r = by_date.get(ds, {})
        present_count = round_half_up(float(r.get("present_count", 0) or 0), 2)
        late_count = int(r.get("late_count", 0) or 0)

        if is_wd and headcount > 0:
            attendance_rate = round_half_up(present_count / headcount, 4)
        else:
            attendance_rate = None

        rates_window.append(attendance_rate)
        if len(rates_window) > 7:
            rates_window.pop(0)
        non_null = [x for x in rates_window if x is not None]
        moving_avg = round_half_up(sum(non_null) / len(non_null), 4) if non_null else None

        items.append(
            {
                "date": ds,
                "is_working_day": is_wd,
                "headcount": headcount,
                "present_count": present_count,
                "late_count": late_count,
                "attendance_rate": attendance_rate,
                "moving_avg_7d": moving_avg,
            }
        )
        cur += timedelta(days=1)

    return {"department": department, "items": items}


def build_trend_pipeline(emp_codes: list[str], date_from: str, date_to: str) -> list[dict]:
    """Aggregate daily present_count (half day = 0.5) and late_count for the
    department's employees across the range. Gap filling to one row per calendar
    day is finished in the handler (headcount depends on per-employee joined_on,
    which is cheaper to resolve in Python than to densify server-side)."""
    present_list = list(PRESENCE_STATUSES)
    return [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": date_from, "$lte": date_to},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", present_list]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
            }
        },
        {"$project": {"_id": 0, "date": "$_id", "present_count": 1, "late_count": 1}},
    ]


# --------------------------------------------------------------------------- #
# Admin: explain
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    date_from_alias: Optional[str] = Query(None, alias="from"),
    date_to_alias: Optional[str] = Query(None, alias="to"),
):
    valid = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }
    if endpoint not in valid:
        raise HTTPException(422, "unknown endpoint")

    def run_find_explain(collection, query, sort=None, skip=0, lim=0):
        cmd = {
            "explain": {
                "find": collection,
                "filter": query,
            },
            "verbosity": "executionStats",
        }
        if sort:
            cmd["explain"]["sort"] = dict(sort)
        if skip:
            cmd["explain"]["skip"] = skip
        if lim:
            cmd["explain"]["limit"] = lim
        return db.command(cmd)

    def run_agg_explain(collection, pipeline):
        return db.command(
            {
                "explain": {
                    "aggregate": collection,
                    "pipeline": pipeline,
                    "cursor": {},
                },
                "verbosity": "executionStats",
            }
        )

    if endpoint == "attendance_list":
        q = build_attendance_query(emp_code, date_from, date_to, status)
        skip = (page - 1) * page_size
        explain = run_find_explain(
            "attendance_logs",
            q,
            sort=[("date", DESCENDING), ("emp_code", ASCENDING)],
            skip=skip,
            lim=page_size,
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(422, "emp_code and month are required")
        validate_month(month)
        explain = run_agg_explain(
            "attendance_logs", employee_monthly_pipeline(emp_code, month)
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "department_summary":
        if not month:
            raise HTTPException(422, "month is required")
        validate_month(month)
        explain = run_agg_explain(
            "employees", department_summary_pipeline(month, department)
        )
        return {"endpoint": endpoint, "collection": "employees", "explain": explain}

    if endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, "month is required")
        validate_month(month)
        explain = run_agg_explain(
            "attendance_logs", late_leaderboard_pipeline(month, department)
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    # department_trend
    frm = date_from_alias or date_from
    to = date_to_alias or date_to
    if not department or not frm or not to:
        raise HTTPException(422, "department, from and to are required")
    validate_date(frm, "from")
    validate_date(to, "to")
    dept_emps = list(db.employees.find({"department": department}, {"emp_code": 1}))
    emp_codes = [e["emp_code"] for e in dept_emps]
    explain = run_agg_explain(
        "attendance_logs", build_trend_pipeline(emp_codes, frm, to)
    )
    return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}
