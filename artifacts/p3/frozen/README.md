# Frozen P3

Frozen after the user reported **all passed** in platform Debugging and requested a freeze. The exact test count was not specified; no graded outcome is claimed.

- Passing source: [task1_agent.py](task1_agent.py), 96,112 bytes.
- SHA256: `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`.
- [User-reported platform result](platform-validation.json) is authoritative for the new pass. Pending-platform wording in historical candidate reports predates this confirmation.
- [Freeze record](freeze.json), [local qualification](qualification-local.json), [differential evidence](differential.json), [preservation receipt](preservation.json).
- Local validation: 247 tests, 326/326 episodes, 257,467 public/compact comparisons, and all prior paired rewards preserved.
- [Complete frozen bundle](../p3-frozen-d4877284.tar.gz), with checksum recorded alongside it and in [freeze receipt](../freeze-receipt.json).

Frozen files are read-only. `agents/p3.py` remains byte-identical. Develop future changes under a new version; preserve this accepted source and all prior evidence. Default Agent uses 3/3.5-second deadlines; explicit Task3Agent uses 11/12 seconds under a caller-enforced 15-second watchdog. V3 integration remains a separate task. No platform submission was performed by the agent.
