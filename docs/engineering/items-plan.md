# Items and the shop: plan (designed 2026-10-11, FEAT-22, D30)

Status: **built 2026-10-11 (parts A–E, schema v18).** Designed with the user in a grilling session the same day. Words are in [the glossary](../glossary.md): Item, Inventory, Shop, Use, Recipe.

## Principle

An item's behavior is **data, never code**. A recipe names a `kind`; the kind is a handler registered in code and validates its own recipe fields. Recipes carry no executable content, so a stranger who finds this project on GitHub and gives an admin a recipe file cannot run anything on the host. As in the minigames, randomness comes from a seed that is stored with the result, so every use can be replayed and tests can pin it.

## v1 scope (decided)

| Topic | Decision |
| --- | --- |
| Item effects | Flavor only: a use produces a message. No coin or scene effect in v1, but the recipe format and the use log leave room for both. |
| Kinds | One registry of kinds. v1 ships `random_message` only. Each kind declares an `audience`: `none` (self, v1), later `member` (a human only), `any` (human or NPC) and `npc` (NPC only). The registry rejects a recipe whose audience does not match its kind. |
| Targets | No `target` option on `/use` in v1; it is added with the first kind that needs one. |
| Consumable / reusable | Per recipe. A consumable loses one per use; a reusable item is held at most once and has a per-member cooldown. |
| Cooldown | Per member per item, a catalog default that each server can override. |
| Hold cap | Reusable: 1. Consumable: a server setting (default 10). No stock limit and no daily purchase cap. |
| Whose inventory | One member on one server. Kept when the member leaves. No trading, gifting or selling back. |
| Curation | A built-in catalog every server can switch on or off, with a per-server price. Admins can **add entries to**, or **replace**, a built-in item's flavor lists. Creating new items from the dashboard is a follow-up (FEAT-29). |
| Two switches | **For sale** (new purchases) and **Allow use** (a kill switch for a bad list entry). Turning off sale never takes items away, and members keep what they own. A price change affects new purchases only. |
| Reply | An embed (title with the item's emoji and name, the result as the description, a footer saying it came from the toy shop, and the remaining count for consumables), public in the channel. No model call. |
| Language | English only in v1, but recipe text is **stored locale-keyed** (`{"en": [...]}`), lookup falls back to `en`, and admin-added entries carry a locale key from day one. No `locale` column until the localization item (FEAT-30). |

## Built-in catalog

Prices are server settings; an admin sets the final value, since the daily amount defaults to 0. Suggested starting prices assume a daily of about 100.

| Item | Kind | Use | Suggested price | Notes |
| --- | --- | --- | --- | --- |
| Magic 8-ball | `random_message` | reusable | 150 | Fixed list of answers; cooldown default 30 s |
| Fortune cookie | `random_message` | consumable | 25 | List of fortunes, plus a lucky number line |
| Dice | `random_message` | reusable | 50 | Numeric range table `1..sides`, `sides` is a recipe parameter (default 6, editable per server) |
| d20 | `random_message` | reusable | 50 | Same, `sides` default 20 |
| Tarot | `random_message` | consumable | 60 | One card from the 22 major arcana, drawn independently, 50% reversed; rows are records (name, upright meaning, reversed meaning) |

Rows in a table are plain strings or small records; a template refers to a record's fields. Admin-added entries are always plain strings (so the tarot deck stays fixed, and an admin adds only flavor lines).

## Data model (schema v18, sketch)

Backup before upgrade as usual. All tables are keyed by `guild_id`.

- `shop_items(guild_id, item_key, for_sale, allow_use, price, cooldown_seconds, extra_entries JSON, replace_entries JSON, params JSON, revision)`. A row appears when an admin first changes an item; with no row the built-in defaults apply (price taken from the built-in's suggested price, off for sale until the server enables it).
- `inventory(guild_id, user_id, item_key, qty CHECK(qty>=0), last_used_at)`.
- `shop_purchases(ledger_id PK, guild_id, user_id, item_key)`. This links a ledger entry to the item it bought, so a reversal can find it.
- `item_uses(id, guild_id, user_id, channel_id, item_key, seed, result JSON, used_at)`. Append-only. It is the replay record and the future hook for effects.
- `guild_settings`: `item_hold_cap` (default 10).

Recipes (built-ins) are files in the repo under `llmcord_core/items/catalog/`, in one format shared with future custom recipes.

## Flows

- **Buy** (one `write_admin` transaction): the item is for sale, the cap is not reached, then a ledger entry (`source='shop'`, reason `Bought <item>`) through `_append_ledger`, a `shop_purchases` row, and the inventory upsert. An insufficient balance fails before anything is written.
- **Use** (one transaction, under a per-member lock like `TableRuntime.lock`): the item is held and `allow_use` is on, the cooldown has passed, a seed is drawn, the kind's handler rolls, then the consumable count drops by one (or `last_used_at` is set) and an `item_uses` row is written. The embed is sent afterwards, best-effort; if it fails the result can still be viewed from the use log.
- **Reverse a purchase**: today a ledger reversal returns coins only. For `source='shop'` entries it also removes one held item if the member still has one. If they already used it, only the coins return and the reversal's note says so. This changes the reversal path and needs an Opus review.

## Surfaces

- `/shop`: an ephemeral listing with price and the member's balance, and a Buy control.
- `/inventory`: the member's items, with a Use control.
- `/use item:<autocomplete>`: direct use.
- Dashboard **Shop** tab (own tab, not in the Currency panel): the catalog table (for sale, allow use, price, cooldown, `sides`, custom entries), the hold-cap setting and a use log. Admin writes use `write_admin` with revision checks (`ConflictError`) and audit entries.

## Invariant check

| Invariant | How it holds |
| --- | --- |
| Guild isolation | Every table and query is keyed by `guild_id`; the dashboard goes through `AuthService.guard` / `AdminService.run`. |
| Consent | No personal facts are stored. The use log holds only the member id, item and result. |
| Admin writes | Shop settings use `write_admin` and revisions. |
| Schema | v18 with backup before upgrade; no existing migration is edited. |
| Currency | Spending goes through the ledger; a mistake is a reversal, never an edit. |
| Model snapshot / presets | Not touched; no model call. |

## Work breakdown (FEAT-22, one commit each)

| Part | Content | Model |
| --- | --- | --- |
| A | `llmcord_core/items/`: kind registry, `random_message` handler, recipe validation, built-in catalog files, seeded rolls, tests | Sonnet |
| B | Schema v18 and store API: buy, use, cooldown, hold cap, reversal hook; tests including concurrency (two simultaneous buys or uses) | Sonnet implementer, Opus review (ledger and concurrency) |
| C | Discord: `/shop`, `/inventory`, `/use`, embeds | Sonnet |
| D | Dashboard **Shop** tab (follows `ui-polish` and `ui-guide.md`) and browser tests | Sonnet |
| E | User docs, glossary check | main session |

## Follow-ups (not in FEAT-22)

- **FEAT-29** Custom items: an admin creates or imports a recipe (validated, size-limited, no code). Includes the extra tables or kinds the admin may use.
- **FEAT-30** Localization: a per-server `locale` setting; recipes already store text locale-keyed.
- **FEAT-31** More kinds: interaction items with an audience (`member`, `any`, `npc`) and a `/use` target; wallet kinds (lottery, scratch card; coins are simply minted, no shared pool); NPC kinds that may call the model; scene-effect groundwork reading from `item_uses`.

## Open details to settle during the build

- Final default prices, the default cooldowns for dice and d20, and the hold-cap default (10 proposed).
- Embed colors and emoji per item.
- Whether the use log shows member names in the Shop tab or ids only (the Currency tab already caps name lookups).
