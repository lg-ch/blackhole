# Multi-stage build for mangrove-search.
# Stage 1: compile rpforest + libmangrove.so
# Stage 2: slim runtime with the binary, .so, and Python orchestrator

FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc make pkg-config \
        liburing-dev libroaring-dev libxxhash-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY src/ ./src/
COPY Makefile ./
RUN make all

# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime deps (Debian bookworm via python:3.11-slim).
# libroaring's runtime so is shipped in the -dev pkg on bookworm; ~50 KB extra
# vs only the .so file. liburing + libxxhash use stable suffix-less names.
RUN apt-get update && apt-get install -y --no-install-recommends \
        liburing2 libroaring-dev libxxhash0 libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the orchestrator
RUN pip install --no-cache-dir numpy clickhouse-driver

WORKDIR /opt/mangrove
COPY --from=builder /src/rpforest        /opt/mangrove/rpforest
COPY --from=builder /src/libmangrove.so  /opt/mangrove/libmangrove.so
COPY scripts/                            /opt/mangrove/scripts/
COPY README.md ROADMAP_10D.md            /opt/mangrove/

ENV PATH="/opt/mangrove:${PATH}"
ENV PYTHONPATH="/opt/mangrove/scripts"

# Default: print usage. Override CMD to run actual workloads.
ENTRYPOINT ["/opt/mangrove/rpforest"]
CMD []
