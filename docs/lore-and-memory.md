# Lore, imports, and memory

## What a character can know

Each response starts with the character card and the history of the selected conversation branch. Lore is then selected from these scopes:

| Location | Eligible shared lore |
| --- | --- |
| World channel | Character book, home-world lore, server lorebooks enabled for that world, and current channel lore or channel lorebook. |
| Hub channel | Character book, **that character's** home-world lore and enabled books, hub lore and enabled lorebooks, and current channel lore or channel lorebook. |
| Thread | The parent channel's scopes plus thread lore and its own branch history. |

Another guest's world lore is not included merely because both guests are in the hub. Character encounters are keyed to character and space. Opted-in personal facts are keyed to person and character, so a bond with that character can follow it between its home world and a linked hub. Ordinary hub events stay in hub-scoped encounter or local scene memory.

Admins can use `/admin lore promote` or the dashboard's **Promote** control to copy a shared fact into another channel, world, or hub. Automatic fact promotion stays local to the channel or thread where it was observed. The bot requires evidence from two distinct scene roots before turning a repeated shared fact into durable local lore. Each durable fact keeps source message IDs when available.

## Importing SillyTavern lorebooks

The dashboard accepts JSON files up to 8 MiB and 2,000 entries in either of these shapes:

```json
[
  {"uid": 1, "key": ["station"], "content": "The station closes at dusk."}
]
```

```json
{
  "entries": {
    "1": {"key": ["station"], "content": "The station closes at dusk."}
  }
}
```

It also accepts an object whose `entries` value is an array. Entry IDs are stable sync keys. A **server lorebook** must be explicitly enabled for chosen worlds and/or hubs. A **channel lorebook** belongs to one bound channel and its threads and needs no world or hub links.

Use **Preview sync** before applying an import. The preview labels each entry as added, updated, removed, unchanged, or conflicted. A conflict means an imported entry changed in the file after its local copy was edited. Choose **Keep local edit** or **Use imported entry** for each conflict; applying without a choice is rejected. Handwritten world, hub, channel, and thread lore is stored separately and is never deleted by a lorebook sync. An import preview expires after 15 minutes, and a changed book revision requires another preview.

V2/V3 character cards can also carry `character_book.entries`. They use the same World Info evaluator as standalone books. A PNG card supplies a thumbnail avatar; a JSON card has no image by itself. Edit embedded entries in the Lore workspace or reimport with conflict review.

## World Info evaluation

The evaluator supports constants; primary and secondary keywords; case and whole-word options; JavaScript-style `/pattern/flags` regex; character name/tag filters; ordering; optional probability; groups; bounded recursion; sticky, cooldown, and delay rules; scan depth; and prompt positions with a Discord conversation equivalent. Evaluation uses a token budget and a maximum of 12 selected entries by default. Probability choices use a deterministic turn/entry seed, and saved activations along the selected message branch inform timing on later turns. `/context` (administrators only) shows the lorebook name, scope, activation reason, and other saved context references for a character line.

SillyTavern-only prompt surfaces such as author-note slots and outlets, and features requiring its vector or automation extensions, are preserved in the imported rule and flagged in the dashboard. Those entries remain inactive until they can be remapped. Discord's prompt layout and conversation history are different from SillyTavern's, so inspect imported books in a private channel before relying on exact World Info behavior.

## Conversation and personal memory

The bot stores message nodes with parent links, snapshots of nearby context, summaries, and response traces in SQLite. Replying to an older character line follows its ancestors rather than later sibling turns. Branch summaries are generated when a scene grows, and brief scene facts are extracted after character replies.

Personal memory starts disabled for each person. `/memory opt_in` allows explicitly stated personal facts to be saved for that person and character. `/memory list` shows their saved facts, `/memory forget` removes one, and `/memory opt_out` disables and erases their personal facts. The dashboard manages shared lore; it does not give admins a page to browse members' personal memories.

Stored conversation nodes and response traces expire after `history_retention_days` (90 by default); the bot runs cleanup at startup and about daily thereafter. Durable lore and opted-in personal facts remain until removed through their controls. `/admin scene delete` is different: it also forgets personal facts and character encounters first learned from the deleted messages (lore already promoted from them stays). Source IDs on durable facts may refer to messages whose stored text has expired. Application error logs record error types rather than chat transcripts.
