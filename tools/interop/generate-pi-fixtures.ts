// Interop fixture generator — pi side.
//
// Writes session/auth artifacts using pi's OWN storage code so ri's readers
// are tested against bytes pi actually produces (not hand-written samples).
//
//   bun generate-pi-fixtures.ts <pi-repo> <out-dir>
//
// Entry ids and timestamps are deterministic so the committed fixtures are
// stable across regenerations.

const [, , piRepo, outDir] = process.argv;
if (!piRepo || !outDir) {
  console.error("usage: bun generate-pi-fixtures.ts <pi-repo> <out-dir>");
  process.exit(1);
}

const { JsonlSessionStorage } = await import(
  `${piRepo}/packages/agent/src/harness/session/jsonl-storage.ts`
);
const { NodeExecutionEnv } = await import(`${piRepo}/packages/agent/src/node.ts`);

const fs = new NodeExecutionEnv({ cwd: outDir });
const sessionPath = `${outDir}/pi-session-v3.jsonl`;
await Bun.write(sessionPath, "");
await Bun.$`rm -f ${sessionPath}`.quiet();

const TS = (n: number) => new Date(Date.UTC(2026, 6, 26, 0, 0, n)).toISOString();
const usage = {
  input: 120,
  output: 45,
  cacheRead: 10,
  cacheWrite: 5,
  totalTokens: 180,
  cost: { input: 0.1, output: 0.2, cacheRead: 0.01, cacheWrite: 0.02, total: 0.33 },
};

const storage = await JsonlSessionStorage.create(fs, sessionPath, {
  cwd: "/workspace/project",
  sessionId: "01998f1e-0000-7000-8000-00000000abcd",
  metadata: { fixture: "interop", nested: { keep: true } },
});

// A realistic tree: a compacted prefix, a branch that was summarized, custom
// app state, labels, and a leaf marker — every entry type ri claims to read.
const entries: any[] = [
  {
    type: "message",
    id: "e0000001",
    parentId: null,
    timestamp: TS(1),
    message: { role: "user", content: "Explain the parser", timestamp: 1_784_000_001_000 },
  },
  {
    type: "model_change",
    id: "e0000002",
    parentId: "e0000001",
    timestamp: TS(2),
    provider: "anthropic",
    modelId: "claude-sonnet-4-5",
  },
  {
    type: "thinking_level_change",
    id: "e0000003",
    parentId: "e0000002",
    timestamp: TS(3),
    thinkingLevel: "high",
  },
  {
    type: "active_tools_change",
    id: "e0000004",
    parentId: "e0000003",
    timestamp: TS(4),
    activeToolNames: ["read", "bash"],
  },
  {
    type: "message",
    id: "e0000005",
    parentId: "e0000004",
    timestamp: TS(5),
    message: {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Checking the file", thinkingSignature: "sig-abc" },
        { type: "text", text: "Reading it now." },
        { type: "toolCall", id: "call_1", name: "read", arguments: { path: "parser.rs" } },
      ],
      api: "anthropic-messages",
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      usage,
      stopReason: "toolUse",
      timestamp: 1_784_000_005_000,
    },
  },
  {
    type: "message",
    id: "e0000006",
    parentId: "e0000005",
    timestamp: TS(6),
    message: {
      role: "toolResult",
      toolCallId: "call_1",
      toolName: "read",
      content: [{ type: "text", text: "fn parse() {}" }],
      isError: false,
      timestamp: 1_784_000_006_000,
    },
  },
  {
    type: "custom_message",
    id: "e0000007",
    parentId: "e0000006",
    timestamp: TS(7),
    customType: "bashExecution",
    content: [{ type: "text", text: "$ cargo test\nok" }],
    display: true,
    details: { exitCode: 0 },
  },
  {
    type: "custom",
    id: "e0000008",
    parentId: "e0000007",
    timestamp: TS(8),
    customType: "appState",
    data: { panel: "diff", scroll: 42 },
  },
  {
    // A branch that was explored and summarized.
    type: "branch_summary",
    id: "e0000009",
    parentId: "e0000006",
    timestamp: TS(9),
    fromId: "e0000008",
    summary: "The user explored formatting options.",
    usage,
    fromHook: false,
  },
  {
    type: "compaction",
    id: "e0000010",
    parentId: "e0000009",
    timestamp: TS(10),
    summary: "Earlier: the user asked about the parser and we read parser.rs.",
    firstKeptEntryId: "e0000006",
    tokensBefore: 12_345,
    usage,
    // pi persists the retained tail on the entry; ri has no counterpart yet.
    retainedTail: [
      { role: "user", content: "Keep going", timestamp: 1_784_000_010_000 },
    ],
  },
  {
    type: "label",
    id: "e0000011",
    parentId: "e0000010",
    timestamp: TS(11),
    targetId: "e0000005",
    label: "first read",
  },
  { type: "session_info", id: "e0000012", parentId: "e0000011", timestamp: TS(12), name: "Parser work" },
  {
    type: "message",
    id: "e0000013",
    parentId: "e0000012",
    timestamp: TS(13),
    message: { role: "user", content: "Now optimize it", timestamp: 1_784_000_013_000 },
  },
];

for (const entry of entries) {
  await storage.appendEntry(entry);
}
await storage.setLeafId("e0000013");

// A second session exercising pi's optional-field shapes that ri's stricter
// types might reject: a compaction with no firstKeptEntryId, a label clearing
// a name, a leaf reset to null, and string (not block) custom content.
const optionalPath = `${outDir}/pi-session-v3-optional-fields.jsonl`;
await Bun.$`rm -f ${optionalPath}`.quiet();
const optional = await JsonlSessionStorage.create(fs, optionalPath, {
  cwd: "/workspace/project",
  sessionId: "01998f1e-0000-7000-8000-0000000f0f0f",
});
for (const entry of [
  {
    type: "message",
    id: "f0000001",
    parentId: null,
    timestamp: TS(21),
    message: { role: "user", content: "hi", timestamp: 1_784_000_021_000 },
  },
  {
    type: "compaction",
    id: "f0000002",
    parentId: "f0000001",
    timestamp: TS(22),
    summary: "Compacted without a kept entry.",
    tokensBefore: 999,
  },
  {
    type: "custom_message",
    id: "f0000003",
    parentId: "f0000002",
    timestamp: TS(23),
    customType: "note",
    content: "plain string content",
    display: false,
  },
  { type: "label", id: "f0000004", parentId: "f0000003", timestamp: TS(24), targetId: "f0000001", label: undefined },
  { type: "session_info", id: "f0000005", parentId: "f0000004", timestamp: TS(25) },
]) {
  await optional.appendEntry(entry as any);
}
await optional.setLeafId(null);

// auth.json in pi's on-disk shape (`Record<providerId, Credential>`, written
// by pi's CLI with JSON.stringify(auth, null, 2)).
const auth = {
  anthropic: {
    type: "oauth",
    refresh: "refresh-token",
    access: "access-token",
    expires: 1_784_000_000_000,
  },
  "openai-codex": {
    type: "oauth",
    refresh: "codex-refresh",
    access: "codex-access",
    expires: 1_784_000_100_000,
    accountId: "acct_123",
  },
  "github-copilot": {
    type: "oauth",
    refresh: "ghu_refresh",
    access: "tid=x;proxy-ep=proxy.individual.githubcopilot.com;",
    expires: 1_784_000_200_000,
    enterpriseUrl: "ghe.example.com",
    availableModelIds: ["gpt-5.2-codex", "claude-sonnet-4.6"],
  },
  openai: { type: "api_key", key: "sk-test" },
  "amazon-bedrock": { type: "api_key", env: { AWS_PROFILE: "prod" } },
};
await Bun.write(`${outDir}/pi-auth.json`, `${JSON.stringify(auth, null, 2)}\n`);

console.log(`wrote fixtures to ${outDir}`);
