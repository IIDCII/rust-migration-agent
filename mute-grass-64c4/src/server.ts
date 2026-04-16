import { createWorkersAI } from "workers-ai-provider";
import { Agent, callable, routeAgentRequest, type Schedule } from "agents";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
    convertToModelMessages,
    generateText,
    tool,
    pruneMessages,
    stepCountIs,
    streamText,
    type ModelMessage
} from "ai";
import { z } from 'zod';

function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
    return messages.map((msg) => {
        if (msg.role !== "user" || typeof msg.content === "string") return msg;
        return {
            ...msg,
            content: msg.content.map((part) => {
                if (part.type !== "file" || typeof part.data !== "string") return part;
                const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
                if (!match) return part;
                const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
                return { ...part, data: bytes, mediaType: match[1] };
            })
        };
    });
}

export class CppToRustWorker extends Agent<Env> {
    @callable()
    async transform(cppCode: string) {
        const ai = createWorkersAI(({ binding: this.env.AI }));
        const { text } = await generateText({
            model: ai("@cf/moonshotai/kimi-k2.5", {
                sessionAffinity: this.sessionAffinity
            }),
            prompt: "Convert this C++ to idiomatic Rust: \n${cppCode}"
        });
        return text;
    }
}

export class ChatAgent extends AIChatAgent<Env> {
    maxPersistedMessages = 100;

    onStart() {
        // Configure OAuth popup behavior for MCP servers that require authentication
        this.mcp.configureOAuthCallback({
            customHandler: (result) => {
                if (result.authSuccess) {
                    return new Response("<script>window.close();</script>", {
                        headers: { "content-type": "text/html" },
                        status: 200
                    });
                }
                return new Response(
                    `Authentication Failed: ${result.authError || "Unknown error"}`,
                    { headers: { "content-type": "text/plain" }, status: 400 }
                );
            }
        });
    }

    @callable()
    async addServer(name: string, url: string) {
        return await this.addMcpServer(name, url);
    }

    @callable()
    async removeServer(serverId: string) {
        await this.removeMcpServer(serverId);
    }

    async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
        const workersai = createWorkersAI({ binding: this.env.AI });

        const result = streamText({
            model: workersai("@cf/moonshotai/kimi-k2.5", {
                sessionAffinity: this.sessionAffinity
            }),
            messages: pruneMessages({
                messages: inlineDataUrls(await convertToModelMessages(this.messages)),
                toolCalls: "before-last-2-messages"
            }),
            tools: {
                // This is the trigger for your orchestrator logic
                convertProjectToRust: tool({
                    description: "Use this when the user provides C++ files to convert to Rust.",
                    inputSchema: z.object({ snippets: z.array(z.string()) }),
                    execute: async ({ snippets }) => {
                        // PARALLELIZATION PATTERN:
                        // Spawn a unique sub-agent for every snippet provided
                        const tasks = snippets.map(async (code, i) => {
                            const worker = await this.subAgent(CppToRustWorker, `worker-${i}`);
                            return worker.transform(code);
                        });

                        const results = await Promise.all(tasks);
                        return results.join("\n\n---\n\n");
                    }
                })
            },
            stopWhen: stepCountIs(5),
            abortSignal: options?.abortSignal
        });

        return result.toUIMessageStreamResponse();
    }

    async executeTask(description: string, _task: Schedule<string>) {
        // Do the actual work here (send email, call API, etc.)
        console.log(`Executing scheduled task: ${description}`);

        // Notify connected clients via a broadcast event.
        // We use broadcast() instead of saveMessages() to avoid injecting
        // into chat history — that would cause the AI to see the notification
        // as new context and potentially loop.
        this.broadcast(
            JSON.stringify({
                type: "scheduled-task",
                description,
                timestamp: new Date().toISOString()
            })
        );
    }
}

export default {
    async fetch(request: Request, env: Env) {
        return (
            (await routeAgentRequest(request, env)) ||
            new Response("Not found", { status: 404 })
        );
    }
} satisfies ExportedHandler<Env>;
