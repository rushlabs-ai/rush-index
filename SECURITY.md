# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on this
repository, or by email to admin@rushlabs.ai. We aim to reply within a few days.

Rush Index runs on your machine, and nothing leaves it after the one-time model download. Every
model file and the ONNX Runtime archive is pinned by URL and SHA-256 and checked after download.
PDFs are read by Poppler's `pdftotext` in a separate process with a time limit, so keep Poppler
updated through your system's packages. Dependencies are exact versions, built with `--locked`
and checked by `cargo deny` (RustSec advisories, licences and sources).
