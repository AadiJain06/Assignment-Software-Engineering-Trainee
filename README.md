# Employee Attendance & Analytics API

FastAPI + MongoDB implementation of the HROne attendance assignment. All application code is in
`app/main.py`. The HTTP contract is `openapi.yaml`; stored shapes are in `DATA_MODEL.md`.

## Run

```bash
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017"   # or your Atlas SRV URI
export MONGO_DB="attendance_db"
uvicorn app.main:app --port 8000
```

`MONGO_URI` and `MONGO_DB` are read from the environment (a local `.env` is loaded for convenience,
but real environment variables take priority). All indexes are created at startup, idempotently, so
the app can be pointed straight at a pre-seeded database. `GET /health` returns 200 once MongoDB
answers a ping, 503 otherwise. Requires MongoDB 6.0+ (uses `$setWindowFields`/`$rank`).

Load the nine sample documents with `python sample_seed.py` (reads the same env vars).

## What is implemented

Every endpoint in the contract:

- `GET /health`, `POST /employees`, `GET /employees`
- `POST /attendance/punch-in`, `POST /attendance/punch-out`, `GET /attendance`
- `PATCH /attendance/{emp_code}/{date}`
- `GET /analytics/employees/{emp_code}/monthly`
- `GET /analytics/departments/summary`
- `GET /analytics/leaderboard/late`
- `GET /analytics/departments/{department}/trend`
- `GET /admin/explain/{endpoint}`

Notes worth knowing:

- Instants cross the API as integer **epoch milliseconds** and are stored as BSON datetimes (UTC);
  the client (`tz_aware=True`) returns UTC-aware datetimes. Values are truncated to whole seconds (R1).
- Business rules R1–R10 live in helper functions (`compute_late_minutes`, `compute_work_hours`,
  `compute_overtime`, `compute_half_day`, `attendance_date_for_punch`, `working_days_in_range`) and
  the aggregation pipelines. Analytics read the **stored** derived values, never recomputing them.
- Concurrency is handled with unique indexes (`emp_code`; `(emp_code, date)`), atomic
  `find_one_and_update` for punch-out, and an optimistic-lock match for regularization (the loser of a
  concurrent edit gets 409, so no history entry is silently lost).
- The three analytics endpoints are MongoDB aggregation pipelines (`$lookup`, `$reduce`,
  `$setWindowFields`/`$rank`). The leaderboard uses standard competition ranking and applies `limit`
  after ranking. The department trend returns one row per calendar day (weekends and empty days
  included) with a trailing 7-day moving average.
- `GET /admin/explain/{endpoint}` runs `explain` with `executionStats` on each endpoint's main
  query/pipeline, built the same way the real endpoint builds it. Verified on a 100k-record dataset:
  every plan uses `IXSCAN` with no `COLLSCAN`.

## Nothing is missing

All contract endpoints and rules R1–R10 are implemented and were exercised locally against MongoDB
7.0 with the sample data and a synthetic ~100k-record dataset (status codes, R1–R5, pagination,
races, ties, no-log employees, mid-month joiners, overnight shifts, gap-filled trend).
