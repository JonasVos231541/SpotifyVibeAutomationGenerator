# SpotifyVibeAutomationGenerator

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:10000
```

## Stack

- **Backend:** Flask + Python 3.10, waitress WSGI server
- **Database:** SQLite with WAL mode (`vibe_splitter/db.py`)
- **Embeddings:** fastembed ONNX MiniLM-L6-v2 (384-dim, no PyTorch) + optional Spotify audio feature fusion
- **Clustering:** HDBSCAN with L2-normalized euclidean distance
- **Real-time:** Server-Sent Events (`/api/events`)
- **Deployment:** Render free tier (512MB RAM), Docker, auto-deploy from GitHub `main`

## Architecture

```
app.py                    # Flask entry, APScheduler (hourly/weekly jobs)
vibe_splitter/
  config.py               # All settings, env vars, genre rules
  hourly.py               # Hourly update logic (extracted from routes to avoid circular imports)
  router.py               # Route new tracks to target playlists via embedding similarity
  routes/                 # Flask Blueprints
    __init__.py           # register_routes(), core middleware (CSRF, CSP, error handler), SSE
    helpers.py            # Shared: _sanitize_name, _valid_id, _ref, rate_limit decorator
    auth.py               # /login, /callback, /logout, /api/wipe-token, /api/token-info
    data.py               # /api/state, /api/stats, /api/playlists, /api/cache-*, /api/overrides
    playlist.py           # /api/update-name, /api/cleanup-playlists, /api/cover-*, /api/retag, /api/override
    inbox.py              # /api/inbox/approve, /api/inbox/dismiss
    targets.py            # /api/targets — manage target playlists and their embeddings
    admin.py              # /api/test-fetch, /api/test-playlist
  spotify_client.py       # Spotify API wrapper, circuit breaker, rate budget
  lastfm.py               # Parallel Last.fm tag fetching with global rate limiter
  embeddings.py           # fastembed encoding pipeline + optional audio feature fusion
  naming.py               # Cluster naming (energy/mood scoring, AI names)
  playlists.py            # Push clusters to Spotify playlists
  state.py                # Thread-safe JSON state manager
  db.py                   # SQLite layer (tracks, config, embeddings)
  cache.py                # Legacy JSON file cache (deprecated, use db.py)
  events.py               # SSE event bus for real-time updates
  templates/index.html    # Frontend (vanilla JS, dark theme)
```

## Key Constraints

- **Memory:** Must stay under 512MB RAM (Render free tier). Use `db.get_track()` / `db.get_tracks_batch()` instead of `cache_mod.load()` which loads everything into memory.
- **Spotify API:** 150 calls/hr budget (`_RateBudget` in spotify_client.py). Circuit breaker opens after 3 consecutive failures. Got an 18.5hr ban once — be cautious with batch operations.
- **Last.fm API:** 0.26s delay between calls (`LASTFM_RATE_DELAY`). Use artist_cache dict to avoid duplicate artist tag lookups.

## Code Standards

- Use SQLite (`db.py`) for all data access, not JSON file cache
- Security: `_sanitize_name()` for user input, `esc()` in frontend for XSS
- CSRF: All POST/PUT/DELETE requests require `X-Requested-With` header (enforced in `routes/__init__.py`, auto-injected by frontend fetch wrapper)
- All POST endpoints should validate input with `_valid_id()` and `_sanitize_name()` (in `routes/helpers.py`)
- Rate limiting: Use `@rate_limit(seconds)` decorator on expensive endpoints (in `routes/helpers.py`)
- SSE events: publish progress via `sse_publish("progress", {...})` during long operations
- Thread safety: use `sm.atomic_update()` for read-modify-write on state
- New endpoints go in the appropriate blueprint file under `routes/`, not in a monolithic routes.py

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPOTIFY_CLIENT_ID` | dev fallback | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | dev fallback | Spotify app client secret |
| `SPOTIFY_REDIRECT_URI` | `http://localhost:10000/callback` | OAuth redirect |
| `FLASK_SECRET_KEY` | random | Session encryption |
| `VS_DB_FILE` | `vibe_splitter.db` | SQLite database path |
| `VS_LOG_FORMAT` | `text` | `text` or `json` |
| `VS_LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
# Import check (all modules)
python -c "from app import app; print('OK')"

# Health check (when running)
curl http://localhost:10000/api/health
```

## Skills & Automation

Skills are installed at `.claude/skills/`. Use them automatically when the task matches the scenarios below.

> **Portability:** The `.claude/skills/` folder is installed per-project and can be removed from this repo if transferred elsewhere. Reinstall any skill with:
> `npx claude-code-templates@latest --skill <category>/<name>`

### Skill Routing

| When the task involves... | Use skill | Key files |
|---------------------------|-----------|-----------|
| Flask routes, API endpoints, SQLite, Spotify client | **senior-backend** | `references/api_design_patterns.md` |
| Redesigning pages, building new UI, CSS/styling | **frontend-design** | `SKILL.md` (methodology) |
| Color palettes, font pairing, UX guidelines, style search | **ui-ux-pro-max** | `scripts/search.py` |
| Design tokens, CSS variables, visual consistency | **ui-design-system** | `scripts/design_token_generator.py` |
| System architecture, module dependencies, tech decisions | **senior-architect** | `references/architecture_patterns.md` |
| React/Next.js/TypeScript (if stack migrates from vanilla JS) | **senior-frontend** | `references/react_patterns.md` |
| PR reviews, code quality audits, anti-pattern checks | **code-reviewer** | `references/code_review_checklist.md` |
| Browser testing, screenshot verification, UI debugging | **webapp-testing** | `scripts/with_server.py` |

### Design Skills

**ui-ux-pro-max** -- Design intelligence with searchable databases (50 styles, 97 palettes, 57 font pairings).

```bash
# Generate full design system (ALWAYS start here for design work)
python .claude/skills/ui-ux-pro-max/scripts/search.py "music dark dashboard" --design-system -p "Vibe Splitter"

# Search specific domain (style|color|typography|ux|chart|landing|product)
python .claude/skills/ui-ux-pro-max/scripts/search.py "<keywords>" --domain <domain>

# Stack-specific guidelines (this project uses vanilla HTML)
python .claude/skills/ui-ux-pro-max/scripts/search.py "<keywords>" --stack html-tailwind
```

**ui-design-system** -- Generate design tokens from a brand color.

```bash
# Generate CSS tokens from Spotify green (styles: modern|classic|playful, formats: css|scss|json)
python .claude/skills/ui-design-system/scripts/design_token_generator.py "#1DB954" modern css
```

**frontend-design** -- No scripts. Provides aesthetic methodology: bold direction, distinctive typography, cohesive color, motion, spatial composition. Read `.claude/skills/frontend-design/SKILL.md` when building or restyling `index.html`.

### Development Skills

**senior-backend** -- Primary skill for this Flask/SQLite project. API scaffolding, DB migrations, load testing.

```bash
python .claude/skills/senior-backend/scripts/api_scaffolder.py <project-path>
python .claude/skills/senior-backend/scripts/database_migration_tool.py <target-path> --verbose
python .claude/skills/senior-backend/scripts/api_load_tester.py <arguments>
```

References: `.claude/skills/senior-backend/references/` -- `api_design_patterns.md`, `database_optimization_guide.md`, `backend_security_practices.md`

**senior-architect** -- Architecture diagrams, dependency analysis, tech decisions.

```bash
python .claude/skills/senior-architect/scripts/architecture_diagram_generator.py <project-path>
python .claude/skills/senior-architect/scripts/dependency_analyzer.py <arguments>
python .claude/skills/senior-architect/scripts/project_architect.py <target-path> --verbose
```

References: `.claude/skills/senior-architect/references/` -- `architecture_patterns.md`, `system_design_workflows.md`, `tech_decision_guide.md`

**senior-frontend** -- React/Next.js/TypeScript toolkit. Not the current stack (vanilla JS + DM Mono), but available if the project migrates to a framework.

```bash
python .claude/skills/senior-frontend/scripts/component_generator.py <project-path>
python .claude/skills/senior-frontend/scripts/bundle_analyzer.py <target-path> --verbose
python .claude/skills/senior-frontend/scripts/frontend_scaffolder.py <arguments>
```

### Quality Skills

**code-reviewer** -- PR analysis, code quality checks, anti-pattern detection.

```bash
python .claude/skills/code-reviewer/scripts/pr_analyzer.py <project-path>
python .claude/skills/code-reviewer/scripts/code_quality_checker.py . --verbose
python .claude/skills/code-reviewer/scripts/review_report_generator.py <arguments>
```

References: `.claude/skills/code-reviewer/references/` -- `code_review_checklist.md`, `coding_standards.md`, `common_antipatterns.md`

**webapp-testing** -- Playwright-based browser testing with server lifecycle management.

```bash
# Start app and run test script
python .claude/skills/webapp-testing/scripts/with_server.py \
  --server "python app.py" --port 10000 \
  -- python your_test_script.py
```

Pattern: reconnaissance-then-action. Always `wait_for_load_state('networkidle')` before inspecting DOM. Examples in `.claude/skills/webapp-testing/examples/`.

### Skill Composition Workflows

**Redesign the dashboard UI:**
1. `ui-ux-pro-max` -- generate design system (`--design-system "music dark dashboard"`)
2. `ui-design-system` -- generate CSS tokens from brand color (`#1DB954`)
3. `frontend-design` -- apply aesthetic methodology to implement in `index.html`
4. `webapp-testing` -- verify the result with Playwright screenshots

**Add a new API endpoint:**
1. `senior-backend` -- scaffold endpoint, check DB patterns and security references
2. `code-reviewer` -- review the implementation
3. `webapp-testing` -- verify endpoint through browser if it has UI

**Architecture review:**
1. `senior-architect` -- generate dependency graph and architecture diagram
2. `code-reviewer` -- run quality check across codebase

### Project-Specific Defaults

- **Stack:** Vanilla HTML/CSS/JS (not React). Use `--stack html-tailwind` with ui-ux-pro-max.
- **Brand color:** `#1DB954` (Spotify green). CSS vars defined in `index.html :root`.
- **Fonts:** Bebas Neue (display) + DM Mono (body). Loaded from Google Fonts.
- **Memory limit:** 512MB RAM on Render. Scope analysis skills to `vibe_splitter/`.
- **Rate limits:** Never exceed 150 Spotify API calls/hr when load-testing.
