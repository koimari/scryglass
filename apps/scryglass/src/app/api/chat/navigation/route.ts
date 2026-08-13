import { chatJson, secureChatRoute } from "@/lib/chatApi";
import { NAVIGATION_HELP } from "@/lib/supportContent";

export const runtime = "nodejs";

async function get() {
  return chatJson({ pages: NAVIGATION_HELP });
}

export const GET = secureChatRoute(get);
