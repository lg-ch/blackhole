#ifndef HOT_WAL_H
#define HOT_WAL_H

/* Write-ahead log for HOT ingestion.
 *
 * Append-only file `<hot_dir>/hot.wal` containing a stream of 12-byte
 * records : [tree_id u32][leaf_id u32][doc_id u32].
 *
 * Durability model :
 *   - hot_append writes to WAL buffer (in-memory)
 *   - background fsync flushes buffer to disk every `fsync_interval_ms` ms
 *   - on crash : at most `fsync_interval_ms` worth of appends lost
 *
 * On process restart : `hot_wal_replay` reads WAL, replays each entry via
 * hot_append into the (freshly opened) HotOverlay. Idempotent — replaying
 * the same entry twice produces the same state as once (append accumulates,
 * so this is NOT strictly idempotent — replay is meant for CLEAN cold-open,
 * meaning after successful compaction that truncated both WAL and .hot).
 *
 * Truncate policy : after a successful bg compaction round that drained
 * HOT, WAL is truncated to 0 bytes. See hot_wal_truncate.
 */

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

typedef struct HotWal HotWal;

/* Open (or create) WAL file at path. Buffered writes with periodic fsync.
 * fsync_interval_ms : 0 = sync per record (very slow), 100+ = batched.     */
HotWal* hot_wal_open(const char* path, int fsync_interval_ms);
void    hot_wal_close(HotWal* w);

/* Append a record. Thread-safe. Non-blocking (buffered). */
int  hot_wal_append(HotWal* w, uint32_t tree_id, uint32_t leaf_id, uint32_t doc_id);

/* Force-flush any buffered records + fsync. */
int  hot_wal_flush(HotWal* w);

/* Truncate WAL to 0 bytes (called after successful global compaction). */
int  hot_wal_truncate(HotWal* w);

/* Stats. */
uint64_t hot_wal_size_bytes(const HotWal* w);
uint64_t hot_wal_n_records(const HotWal* w);

/* Replay callback signature. */
typedef int (*hot_wal_cb)(void* ctx, uint32_t tree_id, uint32_t leaf_id, uint32_t doc_id);

/* Read the WAL file at `path` and invoke `cb` for each record. Used at
 * startup to reconstruct HOT state. Returns records replayed or -1. */
int  hot_wal_replay(const char* path, hot_wal_cb cb, void* ctx);

#endif
