# Security Policy

## Supported Versions

Until `ri` starts publishing versioned releases, security fixes target the
`main` branch.

## Reporting A Vulnerability

Please do not open public issues for suspected security vulnerabilities.

Report security issues by email:

- nowazhu@gmail.com

Include as much detail as practical:

- affected crate, module, or provider
- steps to reproduce
- expected and actual behavior
- whether API keys, OAuth tokens, local files, proxy behavior, or session data
  are involved
- any suggested fix or mitigation

## Security Scope

Relevant issues include, but are not limited to:

- API key, OAuth token, or credential exposure
- unsafe handling of provider auth headers
- request smuggling, SSRF, or proxy bypass behavior
- incorrect `NO_PROXY` or proxy matching behavior
- local file/session storage leaks
- unsafe command execution behavior in harness utilities
- dependency vulnerabilities that affect this workspace

For non-sensitive bugs, use normal GitHub issues.

## Disclosure

Security reports are handled on a best-effort basis. Please allow time to
confirm the issue, prepare a fix, and publish an update before public
disclosure.
