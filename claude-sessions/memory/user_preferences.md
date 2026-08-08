---
name: User engineering style and preferences
description: User's working style on hd_ann_test.cpp tuning — catches inefficiencies, prefers concise technical French, thinks in terms of I/O economics.
type: user
originSessionId: b722493c-5024-46d5-a669-8ab2348ac60d
---
- User speaks and writes in French, often informal/terse. Reply in French with concise technical prose.
- User thinks in terms of I/O economics and resource constraints (RAM, page cache, cgroup limits) — frame optimization trade-offs accordingly.
- User catches redundant work and inefficient experimental protocols quickly (flagged running the same forest traversal 7× for a pool sweep instead of evaluating cutoffs from one pass). Always think about whether expensive work can be factored out before proposing a sweep.
- User prefers exploratory results in compact tables, then a short lessons-learned summary at the end.
- Not interested in optimizations that give only ~10% gain if they require non-trivial code changes. Prefers interventions with clear, measurable impact.
