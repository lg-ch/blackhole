#ifndef LIBURING_COMPAT_H
#define LIBURING_COMPAT_H

/* liburing 2.2+ exposes io_uring_sqe_set_data64 / cqe_get_data64 directly.
   Older releases (e.g. Ubuntu 22.04 ships 2.1) only have the void* variant.
   We always go through the void* form which works on both — the cookies
   we set fit comfortably in a pointer-sized integer on the targets we
   support (x86_64 / aarch64).                                              */

#include <liburing.h>
#include <stdint.h>

static inline void mg_set_data64(struct io_uring_sqe* sqe, uint64_t v) {
    io_uring_sqe_set_data(sqe, (void*)(uintptr_t)v);
}
static inline uint64_t mg_get_data64(struct io_uring_cqe* cqe) {
    return (uint64_t)(uintptr_t)io_uring_cqe_get_data(cqe);
}

#define io_uring_sqe_set_data64 mg_set_data64
#define io_uring_cqe_get_data64 mg_get_data64

#endif
