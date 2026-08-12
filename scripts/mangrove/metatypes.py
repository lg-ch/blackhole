"""metatypes — typage des métadonnées au-dessus du meta_store natif.

Le moteur ne connaît que des clés opaques → bitmaps. Ce module fait le
pont typé dans les deux sens :

  ENCODE (ingest)  : un dict {champ: valeur} → la liste de clés à alimenter
  COMPILE (query)  : un `where` typé → les groupes AND-de-OR pour
                     MetaStore.filter (plages décomposées, regex expansées
                     sur le dictionnaire de clés)

Types supportés :
  str        → 'f=v'                    (catégoriel ; ~10k valeurs max/champ)
  bool       → 'f=true' / 'f=false'
  int ≥ 0    → clé exacte + buckets décimaux 'f.b3=', 'f.b6=', 'f.b9='
               (préfixes alignés 10^3/10^6/10^9) → les PLAGES se
               décomposent en un petit OR de buckets, zéro coût moteur
  date       → int yyyymmdd (réutilise le mécanisme int ; les buckets
               couvrant des jours inexistants sont des bitmaps vides,
               donc inoffensifs)
  float      → int via round(v * 10^decimals) — decimals OBLIGATOIRE
               (déclaré par champ dans FloatSpec)
  list/tuple → chaque élément encodé (tags multi-valués)

Chaque champ reçoit aussi la clé de présence 'f:*' (opérateur exists).

Opérateurs de `where` :
  {'f': v}                      égalité (v: str/bool/int/date/float typé)
  {'f': [v1, v2]}               IN (OR)
  {'f': ('range', a, b)}        plage inclusive sur int/date/float
  {'f': ('re', pattern)}        regex sur les VALEURS du dictionnaire
                                de clés du champ (pas sur le contenu !)
  {'f': ('exists',)}            présence du champ
Les champs sont ANDés entre eux.
"""
from __future__ import annotations

import datetime as _dt
import re as _re

BUCKETS = (3, 6, 9)          # échelles décimales 10^3, 10^6, 10^9
INT_MAX = 10 ** 12           # bornes du domaine int supporté


class FloatSpec:
    """Déclaration de précision d'un champ float : decimals chiffres après
    la virgule (quantification à l'ingest ET à la query)."""

    def __init__(self, decimals: int):
        self.decimals = decimals


def _as_int(field: str, v, float_specs: dict) -> int:
    if isinstance(v, bool):
        raise TypeError(f'{field}: bool ne passe pas par _as_int')
    if isinstance(v, int):
        iv = v
    elif isinstance(v, (_dt.date, _dt.datetime)):
        iv = v.year * 10000 + v.month * 100 + v.day
    elif isinstance(v, float):
        spec = float_specs.get(field)
        if spec is None:
            raise TypeError(
                f'{field}: float sans FloatSpec(decimals=...) déclaré')
        iv = round(v * 10 ** spec.decimals)
    else:
        raise TypeError(f'{field}: type {type(v).__name__} non supporté')
    if not (0 <= iv < INT_MAX):
        raise ValueError(f'{field}: {iv} hors domaine [0, 10^12)')
    return iv


def _int_keys(field: str, iv: int) -> list[str]:
    keys = [f'{field}={iv}']
    for b in BUCKETS:
        keys.append(f'{field}.b{b}={iv // (10 ** b)}')
    return keys


def encode_meta(metadata: dict, float_specs: dict | None = None
                ) -> list[str]:
    """dict {champ: valeur} → clés à alimenter dans le MetaStore."""
    float_specs = float_specs or {}
    keys: list[str] = []
    for field, value in metadata.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if isinstance(v, bool):
                keys.append(f'{field}={"true" if v else "false"}')
            elif isinstance(v, str):
                keys.append(f'{field}={v}')
            else:
                keys.extend(_int_keys(field, _as_int(field, v, float_specs)))
        keys.append(f'{field}:*')
    return keys


def _decompose_range(field: str, lo: int, hi: int) -> list[str]:
    """Couverture gloutonne de [lo, hi] par buckets décimaux alignés +
    valeurs exactes aux bords. O(9 × échelles) clés au pire."""
    keys: list[str] = []
    a = lo
    while a <= hi:
        step = 1
        for b in reversed(BUCKETS):
            s = 10 ** b
            if a % s == 0 and a + s - 1 <= hi:
                keys.append(f'{field}.b{b}={a // s}')
                step = s
                break
        else:
            keys.append(f'{field}={a}')
        a += step
    return keys


def compile_where(where: dict, meta_keys: list[str] | None = None,
                  float_specs: dict | None = None
                  ) -> tuple[list[str], list[int]]:
    """`where` typé → (clés à plat, tailles des groupes) pour
    MetaStore.filter. meta_keys (dictionnaire du store) n'est requis que
    pour l'opérateur regex."""
    float_specs = float_specs or {}
    flat: list[str] = []
    lens: list[int] = []
    for field, cond in where.items():
        if isinstance(cond, tuple) and cond and isinstance(cond[0], str) \
                and cond[0] in ('range', 're', 'exists'):
            op = cond[0]
            if op == 'exists':
                group = [f'{field}:*']
            elif op == 'range':
                lo = _as_int(field, cond[1], float_specs)
                hi = _as_int(field, cond[2], float_specs)
                if lo > hi:
                    lo, hi = hi, lo
                group = _decompose_range(field, lo, hi)
            else:                                    # 're'
                if meta_keys is None:
                    raise ValueError("l'opérateur 're' exige meta_keys "
                                     "(MetaStore.keys())")
                rx = _re.compile(cond[1])
                prefix = f'{field}='
                group = [k for k in meta_keys
                         if k.startswith(prefix) and rx.search(
                             k[len(prefix):])]
                if not group:
                    # clé impossible (0x01 est refusé à l'écriture par le
                    # store) → groupe vide → AND vide le résultat, comme
                    # attendu pour une regex sans aucune valeur matchée
                    group = [f'{field}=\x01aucun-match']
        else:
            values = cond if isinstance(cond, (list, tuple)) else [cond]
            group = []
            for v in values:
                if isinstance(v, bool):
                    group.append(f'{field}={"true" if v else "false"}')
                elif isinstance(v, str):
                    group.append(f'{field}={v}')
                else:
                    group.append(f'{field}={_as_int(field, v, float_specs)}')
        flat.extend(group)
        lens.append(len(group))
    return flat, lens
