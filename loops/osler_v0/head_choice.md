# O9 — head choice at 4B (dev only)

Rule (ADVISORY O9): start with L; switch to P or N only if its dev macro accuracy exceeds L's by at least 1.0 points with a paired bootstrap lower bound above 0 on dev (1000 resamples, items resampled within templates, seed 0). If both qualify, take the larger gain. Macro accuracy is the mean over templates of per-template accuracy (G1's and D24's statistic).

Dev items shared by all three arms: 10843.

| arm | macro accuracy | accuracy | mean Brier |
|---|---|---|---|
| L | 0.8636 | 0.8932 | 0.1609 |
| P | 0.8687 | 0.8939 | 0.1603 |
| N | 0.8663 | 0.8940 | 0.1593 |

| challenger minus L | point | 95 % lower | 95 % upper | qualifies |
|---|---|---|---|---|
| P | +0.0051 | -0.0029 | +0.0137 | no |
| N | +0.0027 | -0.0041 | +0.0098 | no |

**Chosen head: L.** no challenger exceeds L by at least 0.010 with a paired lower bound above 0 on dev; keep L.
