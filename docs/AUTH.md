# Authentication & Authorization

mangrove-search supports two AuthN modes :

1. **API keys** — built-in, simple, suitable for service-to-service
2. **OIDC via reverse proxy** — recommended for human users + SSO

Both can be combined : a reverse proxy validates the OIDC token,
mints a short-lived API key, mangrove enforces per-index AuthZ.

---

## 1. API keys (built-in)

### Config format

Each line in the keys file (or comma-separated env value) :

```
label:secret_key:patterns:perms:qps
```

- **label**     — human-readable name (used in audit logs)
- **secret_key**— the token clients send as `X-API-Key`
- **patterns**  — comma-separated globs of allowed index names (e.g. `arxiv-*,docs-*`)
- **perms**     — `r` (read-only), `rw` (read+write), `admin` (everything incl. create/drop)
- **qps**       — float, max queries/sec (token bucket). `0` = unlimited

Example `mangrove-keys.txt` :

```text
# admin operator
ops:s3cr3t1::admin:0

# read-only for the search frontend, 100 qps limit
search-fe:abc123:docs-*:r:100

# ingestion worker, full RW on arxiv-* but rate-limited
ingest-arxiv:xyz789:arxiv-*:rw:50
```

### Loading

```bash
# Via env var (comma-separated)
export MG_API_KEYS="ops:s3cr3t1::admin:0,search-fe:abc123:docs-*:r:100"
python3 scripts/serve_cluster.py --root /data

# Via file
python3 scripts/serve_cluster.py --root /data --auth-keys-file /etc/mangrove/keys.txt
```

In Kubernetes, mount the keys file from a Secret (see Helm chart).

### Public endpoints (no auth needed)

- `GET /health` — liveness/readiness probes
- `GET /metrics` — Prometheus scraping

Everything else requires `X-API-Key`.

### Behavior

- Missing key → **401** with `WWW-Authenticate: API-Key`
- Wrong perm or scope → **403**
- Rate limited → **429** with `Retry-After: 1`
- Counters : `mg_auth_rejects_total`, `mg_rate_limited_total`

---

## 2. OIDC integration (via reverse proxy)

mangrove doesn't validate OIDC tokens directly — instead we recommend
running an authenticating reverse proxy that does the heavy lifting and
forwards the request to mangrove with a short-lived API key derived
from the user's identity.

### Why this pattern

- **OIDC is non-trivial** : JWKS rotation, claim mapping, refresh flow,
  PKCE — implementing it well is a project on its own.
- **Reverse proxies do it natively** : Caddy, oauth2-proxy, Pomerium,
  nginx with `auth_request`, Traefik with `ForwardAuth` plugin.
- **Audit logs stay clean** : the proxy logs who logged in, mangrove
  logs what they did.

### Recommended stack : Caddy + oauth2-proxy

```caddy
# Caddyfile
mangrove.example.com {
    # Validate OIDC via oauth2-proxy
    reverse_proxy /oauth2/* localhost:4180
    forward_auth localhost:4180 {
        uri /oauth2/auth
        copy_headers X-Auth-Request-User X-Auth-Request-Email
    }
    reverse_proxy localhost:8000 {
        # Inject the matched-to-user API key from a mapping
        header_up X-API-Key {http.request.header.X-Auth-Request-User-Key}
    }
}
```

Mapping `OIDC sub → mangrove API key` lives in your IdP claims or
oauth2-proxy's `--basic-auth-password-header` config.

### Alternative : nginx-ingress with OIDC annotations

In k8s :

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mangrove
  annotations:
    nginx.ingress.kubernetes.io/auth-url: "https://auth.example.com/oauth2/auth"
    nginx.ingress.kubernetes.io/auth-signin: "https://auth.example.com/oauth2/start"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-Auth-Request-Email"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      proxy_set_header X-API-Key $upstream_http_x_api_key;
spec:
  ...
```

### Alternative : Pomerium (modern OIDC proxy, no separate IdP needed)

Pomerium handles OIDC end-to-end and can route to mangrove based on
authorization policies. See https://www.pomerium.com/docs/ .

### TLS

mangrove serves plain HTTP. **Always front it with TLS** in production —
the reverse proxy handles cert provisioning (Let's Encrypt, cert-manager).

---

## Combining the two

A common pattern :

1. Human users → OIDC at the proxy → proxy maps to a per-team API key
2. Service accounts → direct `X-API-Key` for low-latency machine-to-machine

Both end up at mangrove with a `X-API-Key` header ; mangrove enforces
the scope/perm/rate-limit from there.
