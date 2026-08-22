# Recall

Local memory engine for LLM applications.

## Quick Start

1. Install the `recall` CLI from a release artifact or curl installer.
2. Run `recall init` in your project root to generate `recall.toml`.
3. Run `recall serve` to start the backend and local Qdrant.

If you are using the curl installer, point it at your GitHub repository and version, for example:

```bash
RECALL_REPO=jashan7305/recall RECALL_VERSION=v1.0.0 curl -fsSL https://raw.githubusercontent.com/jashan7305/recall/v1.0.0/install.sh | bash
```

## Querying Memory

The default `/query` path searches both normal memory and document chunks.

You can narrow the search with the request scope:

- `auto` searches both memory and documents.
- `memory` searches only normal memory entries.
- `documents` searches only PDF chunks.
- `document_id` can be used with document scope to focus on one PDF.