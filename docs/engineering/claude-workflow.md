# Claude Code workflow audit

Reviewed on 2026-10-08 against official Anthropic documentation. Scope: repository instructions, agents, skills, hook, verification script, and representative auth/store code used to judge delegation risk. This is a workflow audit, not a fresh application security audit or a measured quota benchmark.

## Findings and changes

| Finding | Evidence at the start | Refinement |
| --- | --- | --- |
| Parent model propagated into every custom role | Committed agents used `model: inherit`. Local edits already selected Sonnet for implementation/tests and Opus for review. | Keep Sonnet workers; make routine review Sonnet and request Opus for high-risk reasoning. All six project agents have explicit models. |
| Exploration could still spend Opus usage | No project Explore override existed. | Add bounded, read-only Haiku `Explore`. |
| Small work had no cheaper editing route | Every application change used the same behavioral worker path. | Add Haiku `quick-editor` for exact mechanical changes with existing coverage; preserve the orchestrator's delegation boundary. |
| Full verification was repeated | Test-writer ran all offline tests; implementer ran full verification; orchestrator reran it; reviewer could run it again. | Workers run targeted checks. Haiku `verify-runner` runs the required final suite; orchestrator inspects the log and diff. Retest after executable edits. |
| Routine review always used Opus locally | Reviewer had `model: opus`. | Default to Sonnet; select Opus for auth/security, isolation, consent, migrations, concurrency, or unresolved complex reasoning. |
| Read-only reviewer had command access | Bash could execute arbitrary commands, including checks that write artifacts. | Remove Bash; supply diffs, new-file inventory, and verification evidence. |
| Workers could repeat orchestration instructions | Root delegation instructions lacked an explicit worker exception. | Each worker executes its brief directly without recursive delegation. |
| Verification guidance conflicted | Root required full checks for every change; verify skill required nothing for docs/config and suggested stashing for a base. | Validate metadata/diffs for workflow changes; preserve dirty work and use recorded or isolated baseline evidence. |
| Known-defect handoff conflicted | Implementer allowed only decorator removal; test conventions also required name/docstring cleanup. | Permit the exact documented metadata transition without weakening assertions. |

## Routing policy

| Model | Repository use | Escalation |
| --- | --- | --- |
| Haiku, low effort | Explicit lookups/inventories; sanitized summaries; exact mechanical edits; verification execution and failure excerpts | Ambiguous matches, unclear failures, or behavior/design interpretation → Sonnet |
| Sonnet, medium effort | Main coordination, behavioral tests, ordinary implementation, routine review | High-risk invariants or unresolved cross-module reasoning → Opus |
| Opus | Difficult architecture/debugging and review of security, isolation, consent, migrations, concurrency | Record the reason; reuse gathered evidence; return routine work to the default roles |

These boundaries and turn limits are repository policy based on the code's risk areas. Anthropic's [cost guidance](https://code.claude.com/docs/en/costs#choose-the-right-model) recommends Sonnet for most coding, Opus for complex reasoning, and Haiku for simple subagent tasks. Delegation still consumes usage. Use ordinary tools for repetitive transformations and Haiku for exception summaries rather than starting an agent per file or query.

## Configuration and operational checks

Project settings select Sonnet, medium effort, and a Haiku fallback for unspecified subagents. Current [subagent documentation](https://code.claude.com/docs/en/sub-agents#choose-a-model) orders selection as invocation override → agent definition → subagent environment default → parent. Force mode (`CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`) overrides that routing; leave it off. Older versions had different precedence. Built-in Plan needs an explicit model when the parent is Opus. Turn caps limit individual runs, not total spending.

Anthropic [released Haiku 5.5 on 2026-10-07](https://www.anthropic.com/claude-haiku-5-5), describing narrow, repetitive work and coding subagent use. The [model configuration docs](https://code.claude.com/docs/en/model-config#model-aliases) recommend Claude Code **2.1.293 or newer** for it; the local installation was 2.1.293 at audit time. On the Anthropic API, `haiku` currently maps to 5.5; provider mappings differ. If needed, set `ANTHROPIC_DEFAULT_HAIKU_MODEL` locally to the provider's supported ID rather than committing a provider-specific override.

Before evaluating savings:

1. Check `claude --version`. Start a fresh session to apply project settings; restart after upgrading. `claude update` is the documented upgrade command; this audit did not upgrade your installation.
2. Check `/model` and `/status`. CLI flags, local/managed settings, environment overrides, and allowlists can alter effective routing. Inspect only the relevant model variables, without exposing credentials.
3. During delegation, check `/tasks` for the actual model. Verify Haiku for Explore/quick-editor/verify-runner and Sonnet for behavioral workers; resolve substitutions before high-volume work.
4. Use `/usage` before and after comparable tasks. The [usage documentation](https://code.claude.com/docs/en/costs#track-your-costs) distinguishes usage tracking from model configuration. API price ratios do not establish subscription quota savings. Record escalations, retries, and duplicated work.

No model API calls were made to benchmark this workflow. Static validation confirms configuration structure, not provider access or runtime model selection. The full offline/browser/performance requirements remain for code changes.

## Validation

```bash
git diff --check
claude plugin validate .claude/agents --strict
claude plugin validate .claude/skills --strict
```

Also parse settings JSON and agent/skill YAML; check unique names, explicit models, turn limits, and new files. The [orchestration skill](../../.claude/skills/orchestrate-change/SKILL.md) owns the sequence; the [verify skill](../../.claude/skills/verify/SKILL.md) owns check selection. Anthropic's [best practices](https://code.claude.com/docs/en/best-practices) inform scoped prompts and concrete acceptance checks; they do not prescribe this repository's risk matrix.

Audit result: strict agent/skill validation, JSON/YAML parsing, routing/tool checks, local documentation links, and whitespace checks passed. Existing permission rules and the lint hook were preserved. Application suites were not run for these documentation/configuration changes.
