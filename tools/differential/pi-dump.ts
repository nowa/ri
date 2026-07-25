// Differential payload dumper — pi side.
// Usage: bun pi-dump.ts cases.json > pi-payloads.json
import { getModel, streamSimple } from "/Users/nowa/Projects/agents/pi/packages/ai/src/compat.ts";

const TS = 1000;

function mapContext(c: any): any {
  const ctx = c.context;
  const messages = ctx.messages.map((m: any) => {
    if (m.role === "user") {
      if (m.blocks) {
        return {
          role: "user",
          content: m.blocks.map((b: any) =>
            b.type === "text"
              ? { type: "text", text: b.text }
              : { type: "image", data: b.data, mimeType: b.mimeType },
          ),
          timestamp: TS,
        };
      }
      return { role: "user", content: m.text, timestamp: TS };
    }
    if (m.role === "assistant") {
      const from = m.from ?? { provider: c.provider, model: c.model, api: undefined };
      const base = getModel(from.provider, from.model);
      const content: any[] = [];
      if (m.thinking !== undefined) {
        content.push({
          type: "thinking",
          thinking: m.thinking,
          thinkingSignature: m.thinkingSignature,
        });
      }
      if (m.text !== undefined) {
        content.push({ type: "text", text: m.text });
      }
      for (const tc of m.toolCalls ?? []) {
        content.push({ type: "toolCall", id: tc.id, name: tc.name, arguments: tc.arguments });
      }
      return {
        role: "assistant",
        content,
        api: from.api ?? (c.apiOverride && !m.from ? c.apiOverride : base?.api),
        provider: from.provider,
        model: from.model,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: (m.toolCalls ?? []).length > 0 ? "toolUse" : "stop",
        timestamp: TS,
      };
    }
    if (m.role === "toolResult") {
      return {
        role: "toolResult",
        toolCallId: m.toolCallId,
        toolName: m.toolName,
        content: [{ type: "text", text: m.text }],
        isError: m.isError ?? false,
        timestamp: TS,
      };
    }
    throw new Error(`unknown role ${m.role}`);
  });
  const out: any = { messages };
  if (ctx.system !== undefined) out.systemPrompt = ctx.system;
  if (ctx.tools) out.tools = ctx.tools;
  return out;
}

function mapOptions(options: any): any {
  const out: any = {};
  if (!options) return out;
  if (options.reasoning !== undefined) out.reasoning = options.reasoning;
  if (options.temperature !== undefined) out.temperature = options.temperature;
  if (options.maxTokens !== undefined) out.maxTokens = options.maxTokens;
  if (options.sessionId !== undefined) out.sessionId = options.sessionId;
  return out;
}

const spec = JSON.parse(await Bun.file(process.argv[2]).text());
const results: Record<string, any> = {};

for (const c of spec.cases) {
  try {
    const found = getModel(c.provider, c.model);
    if (!found) throw new Error(`model not found: ${c.provider}/${c.model}`);
    const model: any = structuredClone(found);
    if (c.apiOverride) model.api = c.apiOverride;
    model.baseUrl = "http://127.0.0.1:9";

    let captured: unknown;
    const s = streamSimple(model, mapContext(c), {
      ...mapOptions(c.options),
      apiKey: "fake-key",
      onPayload: (payload: unknown) => {
        captured = payload;
        throw new Error("diff payload captured");
      },
    });
    await s.result();
    results[c.id] = captured === undefined ? { error: "no payload captured" } : { payload: captured };
  } catch (error) {
    results[c.id] = { error: String(error) };
  }
}

console.log(JSON.stringify(results, null, 1));
