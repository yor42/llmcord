# Admin console, presets, and avatars

The console is a NiceGUI application mounted at `/admin/` on the existing private FastAPI service. `/` and the previous server URLs redirect into it. Sign in with Discord using a server administrator account. Bot and web still run separately against the same SQLite file.

## Server setup and characters

Use **Server setup** to create worlds and hubs, link worlds, bind channels, choose casts, and enable ambient participation. Rebinding a channel keeps its ambient setting and removes cast members who are not available in the new space; unlinking a world from a hub removes that world's characters from the hub's channel casts.

Expand a world or hub to edit its **World guidelines** or **Hub guidelines**. Each bound channel has **Channel guidelines**. Use these for the setting, participants' shared fictional roles, tone, and conventions such as occasional fourth-wall jokes. Channel guidance takes precedence over the current world/hub guidance when they conflict. Threads inherit their parent channel's guidance. In hubs, the hub's guidelines describe the current scene; a guest's home-world guidelines are not added. Each field accepts up to 6,000 UTF-8 bytes.

Guidelines are managed instructions included after preset/card instructions, independent of lore matching and lore budgets. They remain present when prompt trimming occurs and work with existing custom presets. All speakers in a turn share one guideline snapshot. Prompt previews include the selected character's home-world guidance, or the current world/channel guidance when **Sample channel (optional)** is selected.

**Delete world** or **Delete hub** shows a confirmation. First move/delete home characters, including archived characters, and rebind any channels elsewhere. Deletion removes owned lore, guidelines, encounters, and hub/book links; shared lorebooks, entries moved elsewhere, and past messages remain. **Delete book** in Imports removes a named lorebook, including empty books, its remaining entries, and its world/hub links after confirmation. Deleted world/book IDs are not reused.

**Characters** edits card fields, including main and post-history instructions, moves home worlds with confirmation, and archives or restores characters. Editing a character shows a bar at the bottom of the page (**Reset** and **Save changes**) while any field differs from what is saved; other changes are refused until you save or reset. Card instructions are added to the server's selected prompt structure. Post-history instructions are placed after conversation history when the provider permits it.

Choose **Create character**, enter a name and home world, and choose **Create** to start with empty card fields and the default emotion slots. Create a world in Server setup first if none exists. Names must be unique within the server.

**Delete character** opens a confirmation. **Delete permanently** removes the character, its owned lore, personal memories, encounters, saved avatar data, and channel/thread cast assignments. Past chat history and Discord messages remain; lore previously moved to another owner also remains. Archive is available when you want to restore a character later. Deletion rejects changes made since the page was loaded, and character IDs are never reused for a new character.

## Lore workspace

Choose an owner on each side of **Lore**: character, channel, world/hub, lorebook, or an existing thread lore owner. Search filters content and keywords; lists show 50 entries per page.

**Server: Server-wide lore** is also available and applies to eligible characters throughout that server. Use **Import JSON entries** under either owner list to preview a SillyTavern or RisuAI file and add entries directly to that owner. The **Direct lore entry import** section in Imports provides the same flow. These imports create independent entries without creating a lorebook, replacing existing entries, or removing entries absent from the file. Identical content/rules/pin state are skipped, including duplicates within one upload. Unsupported imported features stay preserved and inactive. A changed destination requires a fresh preview. Named lorebooks retain their separate synchronization workflow.

Drag an entry by its handle onto the other drop area, including its padding or an empty list. The workspace shows **Saving lore changes…** and waits for that move to finish before accepting another drag. Lists update after the saved change, and the destination opens the page containing the moved entries. Drops within the same owner do not change insertion priority.

Each entry has **Move left/right** and **Delete** shortcuts beside **Edit / transfer**. Check individual entries or use **Select page**, then choose **Move selected left/right** or **Delete selected**. Selection can span pages and owners; **Clear selection** resets it. Deletes show a confirmation with the selected count and a preview. Bulk moves and deletes either complete for the entire selection or leave it unchanged if an entry was edited, moved, or removed in another session. After a conflict, both lists reload and selection clears so you can review current data.

**Edit / transfer** also offers Move and Copy. Copies are independent; moves retain their activation identity, rule settings, and source-message provenance. Transfers only work within the selected server. Moving channel lore to a character makes it follow that character; moving character lore into a channel shares it with eligible characters there.

The editor includes enabled, constant, pinned, keyword, and priority fields plus advanced matching, probability, grouping, recursion, timing, character filters, and placement settings. Keyword arrays use JSON so a comma inside a regex is preserved. Within each prompt position, entries are assembled in ascending priority: higher values appear later in the context. Higher priority values also win when the World Info budget is constrained. Priority zero is valid.

Unsupported imported features remain visible and inactive. Use the preserved import-fields editor to remove unavailable dependencies, and choose a supported placement. Export retains effective settings and unknown source fields.

Owner exports retain imported source IDs and include a `llmcord_pinned` field for restoring pin state in llmcord. Reimporting an exported manual entry into the same owner presents a collision decision rather than silently duplicating or overwriting it.

## Imports

The **Import lorebook** button in **Lore** opens **Imports**. Under **Named lorebooks**, enter a lorebook name, choose **Server lorebook** or **Channel lorebook**, and click **Create lorebook**. Expand the lorebook, choose **Upload JSON lorebook for preview**, resolve any conflicts, and click **Apply lorebook sync**. Enable server lorebooks in the desired worlds or hubs; channel lorebooks apply to their bound channel and threads. **Edit entries in Lore** opens the lorebook's entries in the **Lore** workspace.

**Imports** previews V2/V3 JSON/PNG character cards and standalone lorebook JSON. Lorebooks accept entry arrays, SillyTavern `entries` objects/arrays, and RisuAI version-1 exports (`type: risu`, `ver: 1`, `data: [...]`). The preview identifies the format. Select a home world or destination book before uploading. Review changes and resolve every conflict before applying. Previews expire after 15 minutes; a changed owner revision requires another preview.

RisuAI primary/secondary keywords, selective matching, insertion order (including zero), always-active flags, and regex toggles are mapped to runtime rules. **Keyword matching** in the lore editor can explicitly select literal keys, regex patterns, or automatic detection of `/pattern/flags` syntax. Original entry fields, folder references, and unknown metadata are retained. Folder records are inactive metadata; the editor does not recreate RisuAI's folder tree. Other RisuAI modes, `@@` decorators, and content macros beyond `{{char}}`/`{{user}}` are preserved and flagged as inactive until rewritten or remapped. Exports use the existing llmcord/SillyTavern entry structure and retain source fields; this adds import support, not a RisuAI-format exporter.

RisuAI entry IDs are used when present. Exports without IDs use array indices for reimport matching, so keep their entry order stable when syncing into an existing book. The original export wrapper is retained with the book.

Reimports preserve manual additions. An imported entry that was edited, moved, or deleted requires a decision if its source changes. Keeping the current entry does not recreate it in its previous owner. Character text and manually uploaded default avatars also have conflict choices. Moving a character's home world can prune ineligible casts.

For older databases, known embedded-card entries are tracked conservatively. Unmatched legacy lore remains independent, and uncertain card-text changes require a choice.

## Server prompt presets

Each server has its own preset library and active revision. The built-in default is read-only: edit it and choose **Save draft** to create a copy. Other presets can be renamed by changing their name and saving, duplicated with **Save as new preset**, or deleted once inactive.

Each bundle contains separate configurations for dialogue, director selection, memory extraction, summaries, and image descriptions. Purposes missing from an imported native bundle inherit the built-in defaults.

The built-in dialogue prompt asks for a character's next fictional Discord reply, with knowledge, motives, and reactions grounded in their card. Chat questions are things the character may react to rather than tasks they must always solve. A short reminder after history reinforces the character's voice, followed by any card-specific post-history instructions. This follows SillyTavern's [main prompt and post-history approach](https://docs.sillytavern.app/usage/prompts/); the wording is tailored to casual Discord conversations. Saved custom presets keep their own instructions.

For a stronger individual voice, add contrasting dialogue examples to the character card: an ordinary interaction, an unexpected request, and something the character actually knows about. Examples should demonstrate their phrasing and temperament, not just describe them. Start a fresh conversation when comparing prompt changes; earlier replies can encourage the model to repeat their style.

Blocks support ordering, enabling, custom text, context markers, roles, and dialogue-history injection depth/order. Trimming priority is separate from display order. Required input markers, structured-output schemas, and the emotion-header contract remain present. Eligibility, consent, world-lore isolation, and branch visibility are enforced before the renderer receives data.

**Save draft** creates an immutable revision. **Activate saved revision** applies that revision to the next scene turn. A turn captures its preset once: every speaker, image description, summary, and memory call in that turn keeps that revision. Changing providers later can require a new compatibility mapping before the preset will render successfully.

Use **Assembled request preview** with a sample character, input, and history. It shows roles, expanded text, token estimates, omissions, and provider adaptations without calling a model. It does not read members' stored personal memories.

### SillyTavern compatibility

Import Chat Completion presets containing `prompts` and `prompt_order`. Choose an order profile if several are present. SillyTavern character IDs are profile labels, not Discord identities. Imports populate dialogue only; other purposes keep defaults.

Supported substitutions are `{{char}}`, `{{user}}`, `{{description}}`, `{{personality}}`, `{{scenario}}`, `{{mesExamples}}`, `{{mesExamplesRaw}}`, `{{summary}}`, and `{{group}}`. Expansion runs once.

Time macros (dialogue blocks only) use the timezone of the member whose message started the turn: their own timezone setting for this server (`/time set`), else the server default (Server setup → **Timezone**), else UTC. The time is when that message was sent (not when the reply is written), so every character speaking in that turn sees the same time. The prompt preview uses the current time in the server's timezone. `{{time}}` gives `5:39 PM`, `{{date}}` `October 8, 2026`, `{{weekday}}` `Thursday`, `{{isotime}}` `17:39`, `{{isodate}}` `2026-10-08`, and `{{local_time}}` `Thursday, October 8, 2026, 5:39 PM (Asia/Seoul)`. The built-in preset has a **Time** block after **Location** (`Local time for {{user}}: {{local_time}}`); presets saved before this change keep their blocks, so add one there if you want it. Standard character, scenario, examples, history, and World Info markers are mapped to runtime sources. Scenario/personality formatting and the World Info `{0}` wrapper are supported.

The import preview adds editable llmcord context markers, including card instructions, local memories, summaries, recent messages, and depth-injected lore. Review these alongside imported ordering. Unknown macros, markers, extension dependencies, generation triggers, and prefills need explicit disabling or remapping before activation. Edit a block's preserved fields to clear unsupported `injection_trigger` values or set `extension` to false. A separate checkbox disables an imported assistant prefill.

Anthropic and Google's Gemini endpoint use system instructions at the top level. For imported late or depth-injected system blocks, choose **Move to top-level system instructions**, **Convert late system block to user instructions**, or disable the block. The built-in default uses the top-level adaptation and reports it in preview and traces. On Google's OpenAI-compatible endpoint, system blocks are combined into one message in their original system order after budget trimming, so earlier character details and response contracts are preserved. The assembled preview shows this same request.

**Export full native bundle** preserves all purposes, mappings, revisions' content, and import metadata. **Export SillyTavern dialogue preset** exports only dialogue and requires a JSON array of explicitly omitted nonportable block IDs, or an equivalent supported marker mapping. Different before/after World Info wrappers cannot be represented by SillyTavern's one global wrapper.

Imported generation settings, models, endpoints, and credentials are retained as source data for export but never applied. Runtime model and sampling configuration remains in `config.yaml`. Text Completion/Instruct templates, STscript, and the full SillyTavern macro language are unsupported.

## Emotion avatars

In **Server setup**, select a private text channel for avatar assets. Explicitly deny **View Channel** to `@everyone`, then allow the bot to view it, read history, send messages, and attach files. A private channel controls channel access; published images are Discord CDN assets.

Each character starts with neutral, happy, sad, angry, surprised, and embarrassed slots. Labels and descriptions are editable; additional slots use stable lowercase keys. Neutral is required but its image is optional. Other slots may be removed. Emotion images are separate from the static fallback.

In **Characters → Fallback static avatar**, upload an image; it is saved as soon as the upload finishes. Imported card portraits automatically serve as static fallbacks. This avatar is used when the selected emotion has no usable image, including characters with no emotion images. No asset channel or separate publication is needed for the fallback. **Remove fallback avatar** clears it; **Remove emotion image** clears an image without deleting its emotion slot. Both ask for confirmation, as do **Remove emotion**, **Delete preset**, and every other delete.

Upload static PNG, JPEG, or WebP images up to 8 MiB and 16 megapixels. Images are normalized to PNG thumbnails no larger than 256 pixels. Uploading an emotion image saves it (with the emotion's current label and description); **Save emotion** saves label and description edits. Then choose **Publish / repair image**. Only published usable slots are offered to the model, alongside neutral. Replacing an image retains older published assets for historical messages.

The dialogue model selects one emotion before visible text streams. The bot chooses the avatar before creating its placeholder, and continuation chunks use the same avatar. Headers are removed from saved dialogue, summaries, and extraction inputs. Unknown/malformed headers select neutral; a missing or unavailable emotion image uses the static fallback. With neither image configured, Discord shows the webhook's default icon. Assets are checked on first use after a bot restart; repair deleted assets from the console. Response traces identify whether the avatar came from an emotion image, the static fallback, or Discord's default.

## Model usage and reply status

Generation status shows the dialogue model and its tracked input-plus-output tokens over a rolling 24 hours for the current server. The total includes director, memory, summary, and image calls using that same profile/model. Server setup shows model roles and 24-hour usage/cost totals. Tracking starts with this feature and persists across restarts; it cannot reconstruct earlier provider usage.

By default each completed Discord message has a footer with model, reply input/output tokens, and estimated USD cost. Continuation messages repeat the same reply-level figures; they are not separate model calls. Footers are excluded from saved conversation text and memory extraction. Provider-reported counts include thinking tokens when reported as output. Missing counts and unknown prices display as unavailable rather than being inferred from visible text. Turn the footer off per server with the "Show model and cost footer on replies" switch in the Server setup tab; usage is still recorded and the Usage figures are unaffected.

For Google's `gemini-3.8-flash` endpoint, estimates use [standard paid list rates](https://ai.google.dev/gemini-api/docs/pricing): $0.75/M input, $3.75/M output, and $0.075/M cached input through December 31, 2026, doubling from January 1, 2027 (UTC). These estimates are not billing statements and exclude credits, taxes, and non-token charges. Set a model profile's `billing_tier: free` for a free-tier estimate, or configure `input_cost_per_million`, `output_cost_per_million`, and optional `cached_input_cost_per_million` in `config.yaml`. Other models require configured prices. Set `stream_usage: false` only for compatible backends that reject usage reporting.

## Deployment and security

Use one web worker. Keep Tailscale Serve forwarding the existing localhost port, including its live Socket.IO/WebSocket traffic. NiceGUI assets are served locally; the dashboard content-security policy uses per-response script nonces and allows the framework's local runtime evaluation. Existing non-dashboard routes retain their policy.

Live events bind to the authenticated session and recheck administrator permission. Upload endpoints additionally validate session binding, CSRF, origin, and body size. Logs and response traces record preset identities, included blocks, omissions, and adaptations rather than complete prompts.

Schema version 3 and 4 upgrades preserve existing entry IDs, legacy activation keys, avatars, conversations, and traces. Before upgrading an older database, the migration service creates a dated `*.pre-v3-*.sqlite3` (from v1/v2) or `*.pre-v4-*.sqlite3` (from v3) backup. Keep a manual backup too. Rolling back requires the matching older application and pre-upgrade database; older code must not open the upgraded schema.

Live Discord delivery, real OAuth/Tailscale forwarding, and the ARM64 container still require validation on your private server and target Pi. See [verification](verification.md).
