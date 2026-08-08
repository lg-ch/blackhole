#ifndef SRT_HASH_H
#define SRT_HASH_H

#include <stdint.h>

/* xxhash64 footer for .srt files (SRT3+).
   - On build: srt_finalize_with_hash() opens the file, hashes everything
     written so far, appends 8-byte XXH64 trailer.
   - On verify: srt_verify_hash() reads file, recomputes XXH64 of all bytes
     except the last 8, compares with the stored footer.

   Hash is NOT checked at sorted_store_open_rdonly() — that would force a
   full-file read on every startup. The data_bytes in the header already
   defines the data block range; the footer is purely additional integrity. */

#define SRT_HASH_BYTES 8

/* Compute hash, append footer. Returns 0 on success, -1 on error.
   File must be flushed/closed before calling.                              */
int srt_finalize_with_hash(const char* path);

/* Reads file, recomputes XXH64 over [0, filesize-8), compares with the
   8-byte footer. Returns 1 if OK, 0 if mismatch, -1 on IO error.           */
int srt_verify_hash(const char* path);

#endif
