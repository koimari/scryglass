import { chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { matchTopic, METHODOLOGY_SECTIONS, type MethodologyTopic } from "@/lib/supportContent";

export const runtime = "nodejs";

async function get(request: Request) {
  const topicParam = clean(searchParams(request).get("topic"));
  const topic: MethodologyTopic =
    topicParam === "all" || (Object.keys(METHODOLOGY_SECTIONS) as MethodologyTopic[]).includes(topicParam as MethodologyTopic)
      ? (topicParam as MethodologyTopic)
      : (matchTopic(topicParam) ?? "all");
  return chatJson({ topic, sections: [METHODOLOGY_SECTIONS[topic]] });
}

export const GET = secureChatRoute(get);
