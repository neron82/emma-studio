# Avatar memory architecture

## Implementation plan and result

1. Keep SQLite as the durable source for full session transcripts and summaries.
   Add a public six-character alphanumeric `memory_id` to each session while
   preserving the existing internal ID and every reference to it.
2. Backfill existing IDs and build FTS5 indexes over messages and summaries.
   Keep indexes current with insert/update/delete triggers. New ID candidates
   are checked against the database; a unique index enforces uniqueness.
3. Put session summaries in the system prompt, each prefaced by its memory ID.
   Bound this prompt pool by `STUDIO_MEMORY_MAX_WORDS`. Keep all historical
   transcripts and summaries in SQLite even when older summaries leave the prompt.
4. Supply two bounded retrieval tools and a private, iterative tool-turn loop.
   Share the parser and retrieval implementation between Telegram and Facetime.
5. Test persistence/migration, retrieval, transport integration, parser recovery,
   cancellation, usage accounting, and the five-call limit, then run the suite.

## Storage and retrieval

`data/studio.db` retains full text in `messages` and summaries in `sessions`.
`memory_messages_fts` and `memory_sessions_fts` are external-content indexes:
no transcript truncation, new embedding service, vector database, GPU load, or
new dependency is required. They use SQLite's Unicode tokenizer and BM25 ranking.
An existing database is upgraded on startup; subsequent opens do not rebuild
existing indexes. Late transcript inserts and summary replacements are indexed.

All avatar retrieval is restricted to its bound persona. Omitting `session_id`
searches that persona's database across Telegram and Facetime. Supplying the
public memory ID restricts retrieval to one session. Internal IDs also work
for server-side compatibility. Public IDs are case insensitive on lookup.

`Store.search_memory` is the adaptation point for future hybrid/semantic
retrieval. Current search extracts up to 32 Unicode terms from ordinary text,
uses OR matching, and returns the strongest excerpts. Query text cannot inject
FTS operators or SQL. Results include public ID, transport, timestamp, role,
kind, score, and a bounded excerpt. This is lexical retrieval: paraphrases and
synonyms may require a reformulated query. Results across the two indexes are
merged by score; those scores are a heuristic rather than calibrated confidence.

`memory_read_session` returns exact transcript text in character pages, with
`next_offset` for continuation and a bounded summary. Search hits identify the
session to read. Source text is never shortened in storage.

SQLite summaries drive the prompt. The existing `memory.md` file remains a
readable export and compatibility fallback, including old short internal-ID
headers resolved to public IDs when their sessions exist. No summary generation
is required to index a transcript: a failed summarization leaves it searchable.
Call completion and Telegram `/new` retain their existing finalization behavior.

## Private tool protocol

The avatar receives examples and instructions to search liberally when it lacks
context about previous user interactions. It emits only one JSON call per turn:

```json
{"tool":"memory_search","arguments":{"query":"Clara jasmine","session_id":null,"limit":5}}
```

```json
{"tool":"memory_read_session","arguments":{"session_id":"A1B2C3","offset":0,"limit":8}}
```

Search `limit` is 1–10 excerpts (default 5); read `limit` is 1–10 thousand
characters (default 8). Queries are limited to 1,000 characters. Excerpts are
limited to 2,000 characters each.

The parser accepts fenced JSON, surrounding prose, function wrappers, native
memory function calls, and JSON-string arguments. Recognizable broken calls
receive a private validation error so the avatar can retry. Ordinary answers,
including ordinary JSON, are preserved.

The sequence is:

```text
user → model call → retrieval → private result → model call → … → final answer
```

Results use `<memory_tool_result>` blocks with tool name, result and remaining
budget. Instructions explicitly distinguish them from user input and warn
against obeying instructions quoted in retrieved conversations. Delimiters in
retrieved text are escaped. Realtime uses proper function-call/output history;
Telegram normalizes calls into private assistant JSON and system result blocks.
Only the final answer is delivered or persisted as the assistant reply.

Five calls total are allowed per user turn, including reads and invalid calls.
After the fifth, a final inference is instructed to answer. A sixth attempted
call is suppressed and replaced by a conversational fallback. The count resets
on the next user turn. All provider usage is included in realtime accounting.

## Voice latency and cancellation

For a persona with memory enabled, a full inference turn is buffered before
speech delivery. This is necessary to suppress stray text before a late tool
call. Search turns are never sent to TTS or the browser. The final answer still
uses the existing speech chunking pipeline, but starts after model generation
has completed. Connections without a memory executor retain streaming behavior.

Private retrieval history belongs to a generation snapshot. It cannot leave
pending function calls in the live conversation on cancellation. Cancellation
and speculative-turn freshness are checked while collecting events and after
retrieval; follow-up streams are closed on completion or interruption.

## Validation

`tests/test_memory_retrieval.py` covers ID collision handling, migration/index
backfill and stable reopen, persona/session scope, Unicode search, exact text
pagination, late inserts and summary updates, summary IDs, malformed/wrapped
calls, private follow-up turns, caps, realtime usage, cancellation, and call
finalization including failed summarization. `tests/test_telegram.py` additionally
checks `/new` followed by a private search and a clean persisted final reply.
Run the regression suite with the project's environment:

```bash
python -m pytest tests/ -q
```
