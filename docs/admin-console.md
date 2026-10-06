# Admin console, presets, and avatars

The console is a NiceGUI application mounted at `/admin/` on the existing private FastAPI service. `/` and the previous server URLs redirect into it. Sign in with Discord using a server administrator account. Bot and web still run separately against the same SQLite file.

## Server setup and characters

Use **Server setup** to create worlds and hubs, link worlds, bind channels, choose casts, and enable ambient participation. Rebinding resets a channel's cast and ambient setting.

**Characters** edits card fields, including main and post-history instructions, moves home worlds with confirmation, and archives or restores characters. Card instructions are added to the selected guild prompt structure. Post-history instructions are placed after conversation history when the provider permits it.

## Lore workspace

Choose an owner on each side of **Lore**: character, channel, world/hub, named book, or an existing thread-local lore owner. Search filters content and keywords; lists show 50 entries per page.

Dragging an entry into another owner moves it. **Edit / transfer** also offers Move and Copy. Copies are independent; moves retain their activation identity, rule settings, and source-message provenance. Transfers only work within the selected server. Moving channel lore to a character makes it follow that character; moving character lore into a channel shares it with eligible characters there.

The editor includes enabled, constant, pinned, keyword, and priority fields plus advanced matching, probability, grouping, recursion, timing, character filters, and placement settings. Keyword arrays use JSON so a comma inside a regex is preserved. Higher priority values win when the World Info budget is constrained; selected entries are assembled in ascending order. Priority zero is valid.

Unsupported imported features remain visible and inactive. Use the preserved import-fields editor to remove unavailable dependencies, and choose a supported placement. Export retains effective settings and unknown source fields.

Owner exports retain imported source IDs and include a `llmcord_pinned` field for restoring pin state in llmcord. Reimporting an exported manual entry into the same owner presents a collision decision rather than silently duplicating or overwriting it.

## Imports

**Imports** previews V2/V3 JSON/PNG character cards and standalone lorebook JSON. Select a home world or destination book before uploading. Review changes and resolve every conflict before applying. Previews expire after 15 minutes; a changed owner revision requires another preview.

Reimports preserve manual additions. An imported entry that was edited, moved, or deleted requires a decision if its source changes. Keeping the local decision does not recreate it in its previous owner. Character text and manually uploaded default avatars also have conflict choices. Moving a character's home world can prune ineligible casts.

For older databases, known embedded-card entries are tracked conservatively. Unmatched legacy lore remains independent, and uncertain card-text changes require a choice.

## Guild prompt presets

Each server has its own preset library and active revision. The built-in default is read-only: edit it and choose **Save draft** to create a copy. Other presets can be renamed by changing their name and saving, duplicated with **Save as new preset**, or deleted once inactive.

Each bundle contains separate configurations for dialogue, director selection, memory extraction, summaries, and image descriptions. Purposes missing from an imported native bundle inherit the built-in defaults.

Blocks support ordering, enabling, custom text, context markers, roles, and dialogue-history injection depth/order. Trimming priority is separate from display order. Required input markers, structured-output schemas, and the emotion-header contract remain present. Eligibility, consent, world-lore isolation, and branch visibility are enforced before the renderer receives data.

**Save draft** creates an immutable revision. **Activate saved revision** applies that revision to the next scene turn. A turn captures its preset once: every speaker, image description, summary, and memory call in that turn keeps that revision. Changing providers later can require a new compatibility mapping before the preset will render successfully.

Use **Assembled request preview** with a sample character, input, and history. It shows roles, expanded text, token estimates, omissions, and provider adaptations without calling a model. It does not read members' stored personal memories.

### SillyTavern compatibility

Import Chat Completion presets containing `prompts` and `prompt_order`. Choose an order profile if several are present. SillyTavern character IDs are profile labels, not Discord identities. Imports populate dialogue only; other purposes keep defaults.

Supported substitutions are `{{char}}`, `{{user}}`, `{{description}}`, `{{personality}}`, `{{scenario}}`, `{{mesExamples}}`, `{{mesExamplesRaw}}`, `{{summary}}`, and `{{group}}`. Expansion runs once. Standard character, scenario, examples, history, and World Info markers are mapped to runtime sources. Scenario/personality formatting and the World Info `{0}` wrapper are supported.

The import preview adds editable llmcord context markers, including card instructions, local memories, summaries, recent messages, and depth-injected lore. Review these alongside imported ordering. Unknown macros, markers, extension dependencies, generation triggers, and prefills need explicit disabling or remapping before activation. Edit a block's preserved fields to clear unsupported `injection_trigger` values or set `extension` to false. A separate checkbox disables an imported assistant prefill.

Anthropic only accepts system instructions at the top level. For imported late or depth-injected system blocks, choose **Move to top-level system instructions**, **Convert late system block to user instructions**, or disable the block. The built-in default uses the top-level adaptation and reports it in preview and traces.

**Export full native bundle** preserves all purposes, mappings, revisions' content, and import metadata. **Export SillyTavern dialogue preset** exports only dialogue and requires a JSON array of explicitly omitted nonportable block IDs, or an equivalent supported marker mapping. Different before/after World Info wrappers cannot be represented by SillyTavern's one global wrapper.

Imported generation settings, models, endpoints, and credentials are retained as source data for export but never applied. Runtime model and sampling configuration remains in `config.yaml`. Text Completion/Instruct templates, STscript, and the full SillyTavern macro language are unsupported.

## Emotion avatars

In **Server setup**, select a private text channel for avatar assets. Explicitly deny **View Channel** to `@everyone`, then allow the bot to view it, read history, send messages, and attach files. A private channel controls channel access; published images are Discord CDN assets.

Each character starts with neutral, happy, sad, angry, surprised, and embarrassed slots. Labels and descriptions are editable; additional slots use stable lowercase keys. Neutral is required and uses the character's default avatar. Other slots may be removed.

Upload static PNG, JPEG, or WebP images up to 8 MiB and 16 megapixels. Images are normalized to PNG thumbnails no larger than 256 pixels. Save a slot, then choose **Publish / repair image**. Only published usable slots are offered to the model, alongside neutral. Replacing an image retains older published assets for historical messages.

The dialogue model selects one emotion before visible text streams. The bot chooses the avatar before creating its placeholder, and continuation chunks use the same avatar. Headers are removed from saved dialogue, summaries, and extraction inputs. Unknown/malformed headers or missing assets fall back to neutral/default. Assets are checked on first use after a bot restart; repair deleted assets from the console.

## Deployment and security

Use one web worker. Keep Tailscale Serve forwarding the existing localhost port, including its live Socket.IO/WebSocket traffic. NiceGUI assets are served locally; the dashboard content-security policy uses per-response script nonces and allows the framework's local runtime evaluation. Existing non-dashboard routes retain their policy.

Live events bind to the authenticated session and recheck administrator permission. Upload endpoints additionally validate session binding, CSRF, origin, and body size. Logs and response traces record preset identities, included blocks, omissions, and adaptations rather than complete prompts.

Schema version 3 upgrades preserve existing entry IDs, legacy activation keys, avatars, conversations, and traces. Before upgrading an older database, the migration service creates a dated `*.pre-v3-*.sqlite3` backup. Keep a manual backup too. Rolling back requires the matching older application and pre-upgrade database; older code must not open the upgraded schema.

Live Discord delivery, real OAuth/Tailscale forwarding, and the ARM64 container still require validation on your private server and target Pi. See [verification](verification.md).
