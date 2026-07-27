import { permanentRedirect } from "next/navigation";

/** Permanent home is the article; keep /grubs for old links. */
export default function GrubsRedirectPage() {
  permanentRedirect("/articles/void-grubs-contest-or-leave");
}
