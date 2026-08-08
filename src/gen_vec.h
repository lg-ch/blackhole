#ifndef GEN_VEC_H
#define GEN_VEC_H

#include <stdint.h>

/* Two gen_vec variants. v0 = original (4 xorshift/dim, quasi-gaussian).
   v3 = packed (1 xorshift / 2 dims, uniform). Both produce a unit vector. */
void gen_vec_v0(uint64_t seed, float* out, int dim);
void gen_vec_v3(uint64_t seed, float* out, int dim);

/* Function-pointer dispatch. Defaults to v0 for backward compat with the
   existing on-disk index. Call set_gen_version(3) before any build/query. */
typedef void (*gen_vec_fn)(uint64_t, float*, int);
extern gen_vec_fn gen_vec;

void set_gen_version(int v);   /* 0 -> v0, 3 -> v3 */
int  get_gen_version(void);

void normalize(float* v, int dim);

#endif
