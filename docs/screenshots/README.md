# Dashboard Screenshot Guide

Placeholder directory for NYAYANTRA console captures referenced by the
commented image tags in the main `README.md` (Quick Demo → Screenshots).

## Capture checklist

- Launch: `streamlit run dashboard/app.py`
- Viewport: 1440×900, browser zoom 100%, dark theme
- Format: PNG, keep files under ~500 KB each

## Views to capture

| Filename | View | Suggested content |
|---|---|---|
| `live-triage.png` | Live Triage | A REVIEW-verdict dispute with at least two triggered gates visible |
| `manual-intake.png` | Manual Case Intake | Completed form + sanitizer audit panel (original vs sanitized) |
| `benchmark.png` | Benchmark & Economics | Metric cards + strategy comparison table |
| `audit-ledger.png` | Audit Chain | Entry table with hash prefixes + integrity banner |
| `sanitizer-firewall.png` | Input Firewall | Injection sample with detected threat categories |

After saving the PNGs here, uncomment the matching image lines in the main
`README.md` (Quick Demo → Screenshots) and commit.

Reminder for captions: all data shown is synthetic; provenance IDs are simulated.
