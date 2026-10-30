"""Execution gate for CAPA data workflows; never grants or verifies legal permission."""
import os


class ConsentRequiredError(PermissionError):
    pass


def require_consent():
    """Require a local, nonempty written consent record before reading CAPA data."""
    path = os.environ.get("CAPA_CONSENT_FILE", "").strip()
    if not path:
        raise ConsentRequiredError(
            "CAPA data use is blocked. Obtain written consent from ULG first, "
            "then set CAPA_CONSENT_FILE to a local file containing that permission. "
            "A download or a yes/no prompt is not consent."
        )
    path = os.path.realpath(os.path.expanduser(path))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        inside_repo = os.path.commonpath((repo, path)) == repo
    except ValueError:  # different drives on Windows: the file is outside the repository
        inside_repo = False
    if inside_repo:
        raise ConsentRequiredError("Store the written CAPA permission outside the project repository.")
    try:
        valid = os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        valid = False
    if not valid:
        raise ConsentRequiredError(
            "CAPA_CONSENT_FILE must point to a nonempty, readable local written consent record. "
            "Do not store or commit the record in the project repository."
        )
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise ConsentRequiredError("CAPA_CONSENT_FILE cannot be read.") from exc


if __name__ == "__main__":
    try:
        require_consent()
    except ConsentRequiredError as exc:
        raise SystemExit(str(exc)) from None
    print("Local CAPA consent record is readable. Check its authenticity and scope yourself.")
