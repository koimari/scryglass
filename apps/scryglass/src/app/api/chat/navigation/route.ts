import { chatJson } from "@/lib/chatApi";
import { NAVIGATION_HELP } from "@/lib/supportContent";

export const runtime = "nodejs";

export async function GET() {
  return chatJson({ pages: NAVIGATION_HELP });
}
