#ifndef VARBYTE_H
#define VARBYTE_H

#include <stdint.h>
#include <stddef.h>

/* LEB128-style VarByte for uint32. MSB=1 ⇒ continuation. LSB-first 7-bit chunks.
   Max 5 bytes for any uint32. Encoder writes at *pos, advances *pos.
   Decoder reads from *pos, advances *pos. No bounds checks (caller-owned). */

static inline size_t varbyte_max_bytes_u32(void) { return 5; }

static inline void varbyte_encode_u32(uint8_t* out, size_t* pos, uint32_t v) {
    while (v >= 0x80u) {
        out[(*pos)++] = (uint8_t)((v & 0x7Fu) | 0x80u);
        v >>= 7;
    }
    out[(*pos)++] = (uint8_t)v;
}

static inline uint32_t varbyte_decode_u32(const uint8_t* in, size_t* pos) {
    uint32_t v = 0;
    int shift = 0;
    uint8_t b;
    do {
        b = in[(*pos)++];
        v |= (uint32_t)(b & 0x7Fu) << shift;
        shift += 7;
    } while (b & 0x80u);
    return v;
}

#endif
