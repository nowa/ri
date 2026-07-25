// Interop ground truth — pi side reader.
//
// Loads a session file with pi's own storage/Session code and dumps the
// observable state as JSON, so ri's readers (and ri-written files) can be
// compared against what pi actually sees.
//
//   bun dump-pi-session.ts <pi-repo> <session.jsonl>

const [, , piRepo, sessionPath] = process.argv;
if (!piRepo || !sessionPath) {
  console.error("usage: bun dump-pi-session.ts <pi-repo> <session.jsonl>");
  process.exit(1);
}

const { JsonlSessionStorage } = await import(
  `${piRepo}/packages/agent/src/harness/session/jsonl-storage.ts`
);
const { Session } = await import(`${piRepo}/packages/agent/src/harness/session/session.ts`);
const { NodeExecutionEnv } = await import(`${piRepo}/packages/agent/src/node.ts`);

const fs = new NodeExecutionEnv({ cwd: process.cwd() });
const storage = await JsonlSessionStorage.open(fs, sessionPath);
const session = new Session(storage);

const entries = await storage.getEntries();
const context = await session.buildContext();
const branch = await session.getBranch();

console.log(
  JSON.stringify(
    {
      metadata: storage.getMetadata?.() ?? null,
      leafId: await storage.getLeafId(),
      entryIds: entries.map((entry: any) => entry.id),
      entryTypes: entries.map((entry: any) => entry.type),
      branchIds: branch.map((entry: any) => entry.id),
      context: {
        thinkingLevel: context.thinkingLevel,
        model: context.model,
        activeToolNames: context.activeToolNames,
        messageRoles: context.messages.map((message: any) => message.role ?? message.type ?? "custom"),
        messageTexts: context.messages.map((message: any) =>
          typeof message.content === "string"
            ? message.content
            : Array.isArray(message.content)
              ? message.content
                  .filter((block: any) => block.type === "text")
                  .map((block: any) => block.text)
                  .join("")
              : null,
        ),
      },
    },
    null,
    2,
  ),
);
