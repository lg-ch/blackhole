#ifndef HOT_COMPACT_BG_H
#define HOT_COMPACT_BG_H

/* Background compaction thread. Owns a pthread that loops through trees
 * round-robin, compacting when HOT for a given tree exceeds a threshold.
 * Throttled by a per-iteration sleep to bound SSD bandwidth impact.
 *
 * Serialization : the C compaction call obtains HOT snapshot under tree
 * mutex (via hot_snapshot_and_clear), does the rewrite lock-free, then
 * reopens the Forest fd. Concurrent queries : safe because
 *   - HOT snapshot & clear is atomic w.r.t. new appends (mutex)
 *   - forest_reopen_tree may briefly break in-flight query reads on that
 *     tree ; caller-side design must ensure queries do not span compaction.
 *
 * MVP hard rule : callers MUST NOT issue a query on the forest while
 * forest_reopen_tree runs. Simplest guarantee : a single reader/writer
 * lock at the query entry point, with compaction taking the writer lock
 * only during the fd swap. Left to caller for now (SDK-level).
 */

#include <stdint.h>
#include "query_tree.h"    /* Forest */
#include "hot_store.h"     /* HotOverlay */

typedef struct HotCompactBg HotCompactBg;

/* Create + start a background thread compacting `forest` from `hot` into
 * `index_dir`, one tree per iteration.
 *
 * threshold_docs   : per-tree HOT size to trigger compaction. 0 = always.
 * sleep_ms         : sleep between round-robin passes (throttle).
 * out_format       : 2 or 3 (SRT V2 raw vs V3 varbyte). 3 recommended.
 *
 * Returns NULL on error. Use hot_compact_bg_stop() to join & free. */
HotCompactBg* hot_compact_bg_start(Forest* forest,
                                   HotOverlay* hot,
                                   const char* index_dir,
                                   int threshold_docs,
                                   int sleep_ms,
                                   int out_format);

/* Signal + join the thread. Blocks until thread exits. */
void hot_compact_bg_stop(HotCompactBg* bg);

/* Stats. Read racy but useful for logs. */
uint64_t hot_compact_bg_n_compactions(const HotCompactBg* bg);
uint64_t hot_compact_bg_n_docs_merged(const HotCompactBg* bg);

#endif
