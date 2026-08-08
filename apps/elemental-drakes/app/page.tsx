import study from "@/data/drake-study.json";
import { DrakeStudy } from "@/components/DrakeStudy";
import type { StudyArtifact } from "@/lib/study";

export default function Page() {
  return <DrakeStudy study={study as StudyArtifact} />;
}
