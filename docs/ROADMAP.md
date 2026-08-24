# Roadmap

Scoped from the contract-based held-out IPI evaluation idea.

## Done (scaffold)

- [x] Project layout
- [x] Episode contract JSON Schema
- [x] Pydantic models + loader
- [x] Rule-based referee stubs
- [x] Ten seed episodes across train / heldout, including ambiguous-recipient,
      cross-email, fact-poisoning, hidden-CC/BCC, travel reroute, payroll routing,
      board minutes, and expense poisoning cases
- [x] Email sandbox tool surface placeholder
- [x] Structured attack/defense rule negotiation
- [x] Immutable constitution and judge-side rule locking
- [x] Multi-candidate attack qualification against a true no-defense baseline
- [x] Suppress defense optimization after clean safety + utility wins
- [x] Historical attack archive replay after defense updates
- [x] Mechanical compliance and utility/safety/friction payoffs
- [x] Lock attack success conditions to episode contracts during generation
- [x] Inject required benign facts into adaptive attack requests

## Next

1. Run three post-fix negotiated-game replications and compare fallback / held-out rates
2. Add held-out prompt families and randomized secret environment variants
3. Add calibrated stealth/relevance judges with pairwise consistency checks
4. Persist the hall of fame across runs and sample diverse historical opponents
5. Extend the constitutional protocol to web, documents, memory, and agent messages
