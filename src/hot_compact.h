#ifndef HOT_COMPACT_H
#define HOT_COMPACT_H

/* Per-tree compaction : merge HOT snapshot into MAIN SRT.
 * Reads input MAIN (V2 or V3), merges with sorted HOT leaves,
 * writes new MAIN as SRT V2. Atomic rename swaps in.
 *
 * MVP contract :
 *   - Input MAIN : SRT V2 or V3 (both supported)
 *   - Output     : SRT V2 (raw uint32; delta re-encoding in a follow-up)
 *   - Per-leaf union : concat MAIN + HOT docs, sort, dedupe
 *   - Memory footprint : sparse index of one tree in RAM (~8B × n_nonempty)
 *     + per-leaf buffer (few KB) ; safe for SIFT 10M / DEEP 100M scales.
 *
 * Throttle : caller controls rate by scheduling one call at a time and
 * pacing between calls. Function itself does NOT rate-limit.
 */

#include <stdint.h>
#include "hot_store.h"

/* Compact one tree's HOT snapshot into a new SRT file at out_path.
 * hot_snap = array of HotSnapEntry (leaf_id, n_docs, docs*), sorted asc.
 * out_format = 2 (SRT V2 raw uint32) or 3 (SRT V3 varbyte delta).
 * On success, writes new SRT to out_path (caller renames onto main_path). */
int hot_compact_tree(const char* main_path,
                     const HotSnapEntry* hot_snap, int hot_n,
                     const char* out_path,
                     int out_format,
                     int* out_new_n_nonempty,
                     uint64_t* out_new_total_docs);

/* Legacy alias : V2 output. */
static inline int hot_compact_tree_v2(const char* main_path,
                                       const HotSnapEntry* hot_snap, int hot_n,
                                       const char* out_path,
                                       int* out_new_n_nonempty,
                                       uint64_t* out_new_total_docs) {
    return hot_compact_tree(main_path, hot_snap, hot_n, out_path, 2,
                            out_new_n_nonempty, out_new_total_docs);
}

#endif
