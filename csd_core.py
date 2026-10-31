"""
csd_core.py

Mathematical core of Conceptual Shift Decomposition (CSD).

CSD decomposes how reasoning was revised after an external input (here, a
critique) into two orthogonal components, measured relative to the
reference axis defined by initial reasoning I and external
input C:

  - aligned component: the part of the revision movement that lies *along*
    the reference axis from I toward C
  - residual (perpendicular) component: the part of the revision movement
    that lies *perpendicular* to that reference axis

These are geometric descriptions of revision movement, not measures of
agreement with, acceptance of, or compliance with the input.

csd_decompose() is the single authoritative implementation; every script
and notebook in this repo computes CSD through it.

No embeddings or web UI here -- just the vector geometry, so the math can be
verified independently before wiring it up to real embeddings.
"""

import numpy as np


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors; nan if either has zero norm."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def csd_decompose(I, C, F):
    """
    Compute the Conceptual Shift Decomposition (CSD) of a single
    decision -> critique -> revision triple.

    Geometry
    --------
    Given three points in embedding space:
      I = initial decision
      C = external input (here, a critique)
      F = final (revised) decision

    define:
      d  = F - I            the *revision*: how the decision actually moved
                             between I and F.
      c  = C - I            the *reference vector*: the direction from the
                             initial decision toward the external input.
      c_hat = c / ||c||     the unit vector along the *reference axis*.

    We then decompose d into a component parallel to c_hat and a component
    perpendicular to it:

      d_parallel      = (d . c_hat) c_hat      <- projection onto reference axis
      d_perpendicular = d - d_parallel         <- everything else

    Returns
    -------
    dict with keys:
      aligned_signed            = d . c_hat
                                  Signed projection of the revision onto the
                                  reference axis. Positive means the
                                  same sign as C - I; negative means the
                                  opposite sign along the same direction.
      aligned_signed_normalized = (d . c_hat) / ||d||
                                  aligned_signed as a share of the revision
                                  magnitude, in [-1, 1].
      aligned_raw               = |d . c_hat|
                                  Unsigned magnitude of the aligned component.
      residual_raw              = ||d_perpendicular||
                                  Magnitude of the residual (perpendicular)
                                  component.
      revision_magnitude        = ||d|| = ||F - I||
      aligned_normalized        = aligned_raw / revision_magnitude, in [0, 1]
      residual_normalized       = residual_raw / revision_magnitude, in [0, 1]
      cos_IF, cos_IC, cos_CF    = cosine similarities between the named pair.
      status                    = "ok", "no_revision", or
                                  "reference_undefined" (see Edge cases).

    residual_raw is computed directly from the geometry, not shortcut via
    aligned_raw, but it is a geometric complement rather than an independent
    quantity: aligned_raw**2 + residual_raw**2 == revision_magnitude**2, so
    aligned_normalized**2 + residual_normalized**2 == 1 whenever status is
    "ok".

    Parameters
    ----------
    I, C, F : array-like
        Vectors (any equal dimensionality) representing the initial
        decision, the external input, and the final decision, respectively.

    Edge cases
    ----------
    - If ||d|| == 0 (F == I, the decision didn't change at all): status is
      "no_revision". There is no movement to decompose, so revision_magnitude,
      aligned_signed, aligned_raw and residual_raw are 0.0, and the
      normalized quantities (aligned_signed_normalized, aligned_normalized,
      residual_normalized) are nan rather than dividing by zero. This check
      takes precedence, so it also applies when C == I as well.
    - If ||c|| == 0 (C == I, the external input coincides with the initial
      decision) and the decision did change: status is
      "reference_undefined". There is no direction to project onto, so every
      quantity that requires the reference axis is nan;
      revision_magnitude is still reported.
    - Cosine similarities are always computed from I, C, F directly; one is
      nan only if a vector in its pair is itself the zero vector (cosine
      similarity is undefined against a zero-norm vector).
    """
    I = np.asarray(I, dtype=float)
    C = np.asarray(C, dtype=float)
    F = np.asarray(F, dtype=float)

    d = F - I
    critique_vec = C - I

    norm_d = np.linalg.norm(d)
    norm_critique = np.linalg.norm(critique_vec)

    cosines = {
        "cos_IF": _cosine_similarity(I, F),
        "cos_IC": _cosine_similarity(I, C),
        "cos_CF": _cosine_similarity(C, F),
    }
    nan = float("nan")

    # Edge case: no revision at all -- nothing to decompose.
    if norm_d == 0.0:
        return {
            "aligned_signed": 0.0,
            "aligned_signed_normalized": nan,
            "aligned_raw": 0.0,
            "residual_raw": 0.0,
            "revision_magnitude": 0.0,
            "aligned_normalized": nan,
            "residual_normalized": nan,
            **cosines,
            "status": "no_revision",
        }

    # Edge case: reference axis undefined (input == initial decision).
    if norm_critique == 0.0:
        return {
            "aligned_signed": nan,
            "aligned_signed_normalized": nan,
            "aligned_raw": nan,
            "residual_raw": nan,
            "revision_magnitude": float(norm_d),
            "aligned_normalized": nan,
            "residual_normalized": nan,
            **cosines,
            "status": "reference_undefined",
        }

    c_hat = critique_vec / norm_critique

    along = np.dot(d, c_hat)              # signed length of d along c_hat
    d_parallel = along * c_hat
    d_perpendicular = d - d_parallel

    aligned_raw = abs(along)
    residual_raw = np.linalg.norm(d_perpendicular)

    return {
        "aligned_signed": float(along),
        "aligned_signed_normalized": float(along / norm_d),
        "aligned_raw": float(aligned_raw),
        "residual_raw": float(residual_raw),
        "revision_magnitude": float(norm_d),
        "aligned_normalized": float(aligned_raw / norm_d),
        "residual_normalized": float(residual_raw / norm_d),
        **cosines,
        "status": "ok",
    }


if __name__ == "__main__":
    cases = [
        (
            "F moves exactly along reference axis "
            "(expect aligned_normalized=1, residual_normalized=0)",
            (0, 0),   # I
            (1, 0),   # C  -> reference axis is +x
            (2, 0),   # F  -> moved straight along +x, past C
        ),
        (
            "F moves exactly perpendicular to reference axis "
            "(expect aligned_normalized=0, residual_normalized=1)",
            (0, 0),   # I
            (1, 0),   # C  -> reference axis is +x
            (0, 3),   # F  -> moved straight along +y
        ),
        (
            "F moves at 45 degrees to reference axis "
            "(expect aligned_normalized=residual_normalized~0.707)",
            (0, 0),   # I
            (1, 0),   # C  -> reference axis is +x
            (1, 1),   # F  -> moved equally along +x and +y
        ),
        (
            "F moves along reference axis with negative sign "
            "(expect aligned_signed_normalized=-1, aligned_normalized=1, "
            "residual_normalized=0)",
            (0, 0),    # I
            (1, 0),    # C  -> reference axis is +x
            (-1, 0),   # F  -> moved straight along -x, away from C
        ),
        (
            "F moves at 135 degrees: partly negative along reference, partly "
            "perpendicular (expect aligned_signed_normalized=-0.707, "
            "aligned_normalized=0.707, residual_normalized=0.707)",
            (0, 0),    # I
            (1, 0),    # C  -> reference axis is +x
            (-1, 1),   # F  -> moved -x and +y
        ),
        (
            "No revision: F == I (expect status=no_revision, raw=0, "
            "normalized=nan)",
            (1, 1),   # I
            (3, 1),   # C
            (1, 1),   # F  -> same as I
        ),
        (
            "Reference axis undefined: C == I (expect "
            "status=reference_undefined, decomposition=nan)",
            (0, 0),   # I
            (0, 0),   # C  -> same as I
            (1, 1),   # F
        ),
    ]

    for label, I, C, F in cases:
        result = csd_decompose(I, C, F)
        print(f"{label}")
        print(f"  I={I}, C={C}, F={F}")
        print(f"  status={result['status']}")
        print(
            f"  aligned_signed={result['aligned_signed']:.4f}, "
            f"aligned_raw={result['aligned_raw']:.4f}, "
            f"residual_raw={result['residual_raw']:.4f}, "
            f"revision_magnitude={result['revision_magnitude']:.4f}"
        )
        print(
            f"  aligned_signed_normalized={result['aligned_signed_normalized']:.4f}, "
            f"aligned_normalized={result['aligned_normalized']:.4f}, "
            f"residual_normalized={result['residual_normalized']:.4f}"
        )
        print(
            f"  cos_IF={result['cos_IF']:.4f}, cos_IC={result['cos_IC']:.4f}, "
            f"cos_CF={result['cos_CF']:.4f}"
        )

        # Pythagorean identity check (only meaningful when the decomposition
        # is defined): the aligned and residual components split the revision.
        if result["status"] == "ok":
            identity = (
                result["aligned_normalized"] ** 2
                + result["residual_normalized"] ** 2
            )
            assert abs(identity - 1.0) < 1e-9, (
                f"aligned_normalized^2 + residual_normalized^2 = {identity!r}, "
                f"expected ~1.0"
            )
            raw_identity = result["aligned_raw"] ** 2 + result["residual_raw"] ** 2
            assert abs(raw_identity - result["revision_magnitude"] ** 2) < 1e-9, (
                f"aligned_raw^2 + residual_raw^2 = {raw_identity!r}, expected "
                f"revision_magnitude^2 = {result['revision_magnitude'] ** 2!r}"
            )
            print(
                f"  aligned_normalized^2 + residual_normalized^2 = "
                f"{identity:.4f} (expect 1.0)"
            )
        print()
