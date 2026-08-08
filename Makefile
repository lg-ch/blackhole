CC      := gcc
CFLAGS  := -O3 -march=native -fPIC -Wall -Wextra -Wno-unused-parameter -std=c11 -fopenmp
LDFLAGS := -lm -fopenmp -luring -lroaring -lxxhash -lpthread

# Core .o's (no main, no top-level cmd). Used by both binary and .so.
CORE := src/gen_vec.c src/traversal.c src/sorted_store.c src/srt_hash.c \
        src/build_tree.c src/query_tree.c src/tombstones.c \
        src/recall.c src/croaring_io.c src/tquant.c src/tq1.c src/ffi.c \
        src/calibration.c src/slot_store.c src/slot_query.c src/hot_store.c \
        src/hot_compact.c src/hot_compact_bg.c src/hot_wal.c
CORE_OBJ := $(CORE:.c=.o)
SRC := $(CORE) src/main.c
OBJ := $(SRC:.c=.o)

all: rpforest libmangrove.so

rpforest: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

libmangrove.so: $(CORE_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(CORE_OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) rpforest libmangrove.so

.PHONY: clean all
