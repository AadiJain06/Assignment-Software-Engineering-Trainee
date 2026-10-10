# DECISIONS.md

1. **Indexes.** `employees`: unique `emp_code` (identity + race-safe creation), `(department, emp_code)` (list/filter), `joined_on` (R9 headcount). `attendance_logs`: unique `(emp_code, date)` (one record per day + race-safe punch-in), `(date desc, emp_code asc)` (list sort/range), `(emp_code, date)` (monthly/leaderboard scans), `(status, date)` (status filter). I rejected a `punch_in` index: punch-out already finds the open record by the unique `(emp_code,date)` path, so it earns nothing.

2. **Punch-in race.** Both requests build the document and call `insert_one`. The unique `(emp_code, date)` index lets exactly one commit; the other raises `DuplicateKeyError`, which I map to 409. No check-then-insert, so there is no window to lose.

3. **Ties.** Ranking is `$setWindowFields` with `$rank` (standard competition ranking: 1,2,2,4). `limit` is applied **after** ranking by keeping every row with `rank <= limit`. So two employees tied at the cutoff both appear, and the response can hold more than `limit` rows. Tied rows are ordered by `emp_code` ascending.

4. **Headcount.** The summary pipeline starts from `employees` (filtered `joined_on <= month end`, R9) and `$lookup`s logs, so a zero-log employee still contributes 1 to `headcount`.

5. **At 100x.** I would shard `attendance_logs` on `emp_code` (hashed) and pre-aggregate daily department rollups so analytics read summaries instead of scanning raw logs.
