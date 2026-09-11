"""The wheel bench: build a mechanic in isolation, prove it, then merge it.

Deliberately empty of imports. The live runner imports `bench.config` at
module scope, so anything expensive or fragile pulled in here would
be pulled in by the live bot at 17:00 ET. Import submodules directly.
"""
