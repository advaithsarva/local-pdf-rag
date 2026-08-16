"""Prove the suite is not decorative: run it against the original implementation.

    python tests/verify_tests.py

A test suite that passes on the broken code tests nothing. The chunking tests
are the ones the original can be subjected to at all (the rest cover modules it
never had); every one of them must fail.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import original_impl  # noqa: E402
import test_ragpdf as T  # noqa: E402

# The tests written directly against defects found in the original, on the one
# stage the original implemented at all. The rest of the suite covers modules
# the notebook never had (retrieval guards, index consistency, degradation,
# abstention), so running them against it would prove nothing.
#
# test_chunk_carries_its_page is deliberately NOT here: the original satisfies
# it. It is a sanity check, not a discriminator, and counting it would inflate
# this number.
CHUNKING_TESTS = [
    T.test_chunks_are_verbatim_slices,
    T.test_no_fused_sentence_boundaries,
    T.test_chunk_text_is_findable_in_the_page,
]


def main():
    print("=" * 72)
    print("Running the chunking assertions against the ORIGINAL notebook code.")
    print("Expected: all of them FAIL. If any pass, the suite is decorative.")
    print("=" * 72)

    failed = 0
    for t in CHUNKING_TESTS:
        try:
            t(chunker=original_impl.chunk_page)
        except Exception as e:
            failed += 1
            print(f"  FAIL (expected) {t.__name__}\n        {str(e).splitlines()[0][:150]}")
        else:
            print(f"  PASS (PROBLEM)  {t.__name__} -- the original satisfies this assertion")

    n = len(CHUNKING_TESTS)
    print(f"\n{failed}/{n} chunking assertions fail against the original.")

    print("\n" + "=" * 72)
    print("Same assertions against the rebuild:")
    print("=" * 72)
    passed, rebuild_failed = T.run(label="rebuild (full suite)")

    ok = failed == n and not rebuild_failed
    print("\nRESULT:", "suite is trustworthy" if ok else "SUITE IS NOT TRUSTWORTHY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
