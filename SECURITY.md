# Security Policy

hk is my project, and I take the security of its parsers, file formats and server seriously: model files, tokenizers, chat templates and HTTP input are all treated as hostile.

---

## Supported Versions

Only the latest stable release line is actively supported with security patches and critical bug fixes.

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x     | :x: (upgrade to 2.0) |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability in the HK binary format, Zig runtime, C API bindings, or Python deserialization logic (e.g. out-of-bounds memory read, buffer overflow, arbitrary code execution, or container denial-of-service), **please do not open a public issue.**

Instead, please report it through one of the following channels:
1. **GitHub Private Vulnerability Reporting**: Go to the **Security** tab of the HK repository and click **"Report a vulnerability"**.
2. **Email**: Send details to me, Harshit Khandelwal, at `harshitkhandelwal208@gmail.com`.

### What to Include in Your Report
- A description of the vulnerability and its potential impact.
- Clear step-by-step reproduction steps or a minimal proof-of-concept (`.hk` or `.gguf` file).
- The operating system, architecture, and version of HK being run.

### What to expect
This is a personal project, so I can't promise a support-desk turnaround, but I read every report. I'll acknowledge it, reproduce it, and fix it in the next release (or sooner for something serious). If you want credit in the release notes, tell me how you'd like to be named; if you prefer to stay anonymous, say so.
