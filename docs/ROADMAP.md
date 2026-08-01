# Roadmap

Scoped from the contract-based held-out IPI evaluation idea.

## Done (scaffold)

- [x] Project layout
- [x] Episode contract JSON Schema
- [x] Pydantic models + loader
- [x] Rule-based referee stubs
- [x] Two seed episodes (train / heldout)
- [x] Email sandbox tool surface placeholder

## Next

1. Expand to ~10 hand-written email episodes
2. Strengthen referee (replay / trivial-defense / impossibility checks)
3. Wire LLM agent loop against `EmailSandbox`
4. Implement baseline defenses
5. Report train vs held-out ASR, utility, and transfer gap
