"""Static regression tests for release-job safety guards.

These tests never invoke a workflow or contact PyPI. They verify the checked-in
GitHub Actions expressions and the intended fork/upstream decision matrix.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
UPSTREAM_REPO = "google-ai-edge/litert-torch"
FORK_REPO = "secheverrianogal-beep/litert-torch"
APPROVED_RELEASE_REF = "refs/heads/main"


def job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    if not match:
        raise AssertionError(f"job {job_name!r} was not found")
    return match.group(1)


class ReleaseWorkflowGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nightly = (WORKFLOWS / "nightly_release.yml").read_text(encoding="utf-8")
        cls.stable = (WORKFLOWS / "stable_release.yml").read_text(encoding="utf-8")

    def test_nightly_publisher_is_restricted_to_upstream(self):
        job = job_block(self.nightly, "build-and-release-nightly")
        self.assertIn(
            f"if: github.repository == '{UPSTREAM_REPO}' && "
            f"github.ref == '{APPROVED_RELEASE_REF}'",
            job,
        )
        self.assertRegex(self.nightly, r"(?m)^permissions:\n  contents: read$")
        self.assertIn("persist-credentials: false", job)

    def test_nightly_uses_masked_environment_for_twine(self):
        job = job_block(self.nightly, "build-and-release-nightly")
        self.assertIn("TWINE_USERNAME: __token__", job)
        self.assertIn("TWINE_PASSWORD: ${{ secrets.PYPI_UPLOAD_TOKEN }}", job)
        self.assertIn("run: twine upload dist/* --non-interactive", job)
        self.assertNotRegex(job, r"(?m)^\s*run:.*twine upload.* -p ")

    def test_stable_publisher_requires_upstream_and_explicit_publish(self):
        job = job_block(self.stable, "publish-stable")
        self.assertIn(
            f"if: success() && github.repository == '{UPSTREAM_REPO}' && "
            f"github.ref == '{APPROVED_RELEASE_REF}' && inputs.dry_run == false",
            job,
        )
        self.assertRegex(job, r"(?m)^    permissions: \{\}$")
        self.assertIn("TWINE_PASSWORD: ${{ secrets.PYPI_UPLOAD_TOKEN }}", job)
        self.assertRegex(self.stable, r"(?m)^permissions:\n  contents: read$")
        self.assertIn("persist-credentials: false", self.stable)

    def test_nightly_decision_matrix(self):
        cases = [
            (FORK_REPO, APPROVED_RELEASE_REF, "workflow_dispatch", False),
            (UPSTREAM_REPO, "refs/heads/feature", "workflow_dispatch", False),
            (UPSTREAM_REPO, APPROVED_RELEASE_REF, "workflow_dispatch", True),
            (UPSTREAM_REPO, APPROVED_RELEASE_REF, "schedule", True),
            (UPSTREAM_REPO, "refs/heads/feature", "schedule", False),
        ]
        for repository, ref, event_name, expected in cases:
            with self.subTest(repository=repository, ref=ref, event=event_name):
                self.assertEqual(
                    repository == UPSTREAM_REPO and ref == APPROVED_RELEASE_REF,
                    expected,
                )

    def test_stable_decision_matrix(self):
        cases = [
            (FORK_REPO, APPROVED_RELEASE_REF, True, False),
            (FORK_REPO, APPROVED_RELEASE_REF, False, False),
            (UPSTREAM_REPO, "refs/heads/feature", False, False),
            (UPSTREAM_REPO, APPROVED_RELEASE_REF, True, False),
            (UPSTREAM_REPO, APPROVED_RELEASE_REF, False, True),
        ]
        for repository, ref, dry_run, expected in cases:
            with self.subTest(repository=repository, ref=ref, dry_run=dry_run):
                self.assertEqual(
                    repository == UPSTREAM_REPO
                    and ref == APPROVED_RELEASE_REF
                    and not dry_run,
                    expected,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
