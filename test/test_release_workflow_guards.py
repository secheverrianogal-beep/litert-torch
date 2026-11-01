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
        self.assertIn(f"if: github.repository == '{UPSTREAM_REPO}'", job)
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
            f"if: success() && github.repository == '{UPSTREAM_REPO}' && inputs.dry_run == false",
            job,
        )
        self.assertRegex(job, r"(?m)^    permissions: \{\}$")
        self.assertIn("TWINE_PASSWORD: ${{ secrets.PYPI_UPLOAD_TOKEN }}", job)
        self.assertRegex(self.stable, r"(?m)^permissions:\n  contents: read$")
        self.assertIn("persist-credentials: false", self.stable)

    def test_nightly_decision_matrix(self):
        # The job-level expression intentionally ignores event type and requires
        # repository identity, so manual dispatches in forks remain blocked.
        cases = [
            (FORK_REPO, "workflow_dispatch", False),
            (FORK_REPO, "schedule", False),
            (UPSTREAM_REPO, "workflow_dispatch", True),
            (UPSTREAM_REPO, "schedule", True),
        ]
        for repository, event_name, expected in cases:
            with self.subTest(repository=repository, event=event_name):
                self.assertEqual(repository == UPSTREAM_REPO, expected)

    def test_stable_decision_matrix(self):
        cases = [
            (FORK_REPO, True, False),
            (FORK_REPO, False, False),
            (UPSTREAM_REPO, True, False),
            (UPSTREAM_REPO, False, True),
        ]
        for repository, dry_run, expected in cases:
            with self.subTest(repository=repository, dry_run=dry_run):
                self.assertEqual(
                    repository == UPSTREAM_REPO and not dry_run, expected
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
