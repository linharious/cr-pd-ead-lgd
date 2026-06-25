"""Result-file storage: local filesystem (dev) or S3 (prod).

STORAGE_DIR selects the backend:
  - a local path (default "out/store"), or
  - "s3://bucket/prefix" on Lambda.

Scored batch results are written under a "results/" prefix. Downloads are served
from disk locally, or via an S3 presigned URL in production (so big files never
flow back through Lambda / API Gateway).
"""
import os

RESULTS_PREFIX = "results"


def _base():
    return os.environ.get("STORAGE_DIR", os.path.join("out", "store"))


def _is_s3(path):
    return path.startswith("s3://")


def _split_s3(uri):
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _s3():
    import boto3
    return boto3.client("s3")


def _key(prefix, *parts):
    return "/".join([p for p in (prefix, RESULTS_PREFIX, *parts) if p])


def save_result(filename, data):
    """Write result bytes under results/<filename>. Returns the filename."""
    base = _base()
    if _is_s3(base):
        bucket, prefix = _split_s3(base)
        _s3().put_object(Bucket=bucket, Key=_key(prefix, filename),
                         Body=data, ContentType="text/csv")
    else:
        path = os.path.join(base, RESULTS_PREFIX, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return filename


def list_results():
    """Result filenames, newest first."""
    base = _base()
    if _is_s3(base):
        bucket, prefix = _split_s3(base)
        resp = _s3().list_objects_v2(Bucket=bucket, Prefix=_key(prefix) + "/")
        names = [os.path.basename(o["Key"]) for o in resp.get("Contents", [])
                 if not o["Key"].endswith("/")]
    else:
        d = os.path.join(base, RESULTS_PREFIX)
        names = os.listdir(d) if os.path.isdir(d) else []
    return sorted(names, reverse=True)


def save_named(relkey, data):
    """Write bytes at an arbitrary relative key (e.g. 'monitoring/psi.json')."""
    base = _base()
    if _is_s3(base):
        bucket, prefix = _split_s3(base)
        key = "/".join([p for p in (prefix, relkey) if p])
        _s3().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        path = os.path.join(base, *relkey.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return relkey


def read_named(relkey):
    """Read bytes at a relative key, or None if it does not exist."""
    base = _base()
    if _is_s3(base):
        bucket, prefix = _split_s3(base)
        key = "/".join([p for p in (prefix, relkey) if p])
        try:
            return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:  # noqa: BLE001 — missing object
            return None
    path = os.path.join(base, *relkey.split("/"))
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def fetch(filename):
    """Return ('file', path) locally or ('url', presigned_url) on S3; None if absent."""
    filename = os.path.basename(filename)
    if filename not in list_results():
        return None
    base = _base()
    if _is_s3(base):
        bucket, prefix = _split_s3(base)
        url = _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": _key(prefix, filename)},
            ExpiresIn=3600,
        )
        return ("url", url)
    return ("file", os.path.join(base, RESULTS_PREFIX, filename))
