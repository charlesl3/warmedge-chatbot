
# WarmGPT Engineering Logs

## Engineering Log — 2026-05-24

- Added blade sharpening tracker infrastructure with skating-session logging, session notes, automatic blade-hour accumulation, and sharpening cycle management through Supabase-backed APIs.
- Built a global topic-memory distillation pipeline to periodically clean and canonicalize user skating interests using LLM-based summarization, synonym merging, vague-topic removal, and full-memory overwrite architecture.
- Added internal admin maintenance endpoint for global topic distillation across all users without requiring user authentication or frontend interaction.
- Set up autonomous weekly backend maintenance using GitHub Actions scheduled workflows to silently trigger periodic topic-memory distillation in production.


## Engineering Log — 2026-05-25

- Added sharpening-aware response injection using tracker state with dual-gate relevance control.
- Built smart natural-language blade tracker querying (hours since sharpening, last sharpening, etc.) without RAG.
- Added automatic topic-memory distillation with admin-wide overwrite support in Supabase.
- Configured weekly backend automation using GitHub Actions for silent maintenance tasks.
- Added long-term skater profile update detection for stable user progression signals.
- Added backend/frontend latency instrumentation across major pipeline stages.
- Removed duplicated semantic profile-update calls and added hard prefilters before expensive LLM agents.
- Reduced backend request latency from roughly ~10s to ~4s.
- Cleaned backend observability/logging into a more production-style request trace format.
- Identified orchestration/side-agent overhead as the new primary bottleneck rather than retrieval or generation.