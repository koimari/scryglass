"use client";

import { useState } from "react";

type Props = {
  src: string;
  title?: string;
};

export function PdfEmbed({ src, title = "PDF" }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className="blog-cta"
          style={{ marginTop: 0 }}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide PDF" : "Open main PDF"}
        </button>
        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="status-pill ghost pill-btn"
        >
          Open in new tab
        </a>
      </div>
      {open && (
        <div className="pdf-frame anim-fade-up">
          <iframe title={title} src={`${src}#view=FitH`} className="pdf-iframe" loading="lazy" />
        </div>
      )}
    </div>
  );
}
