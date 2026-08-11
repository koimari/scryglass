"use client";

import Lenis from "lenis";
import { useEffect } from "react";

const EXPONENTIAL_EASE = (value: number) =>
  Math.min(1, 1.001 - Math.pow(2, -10 * value));

export function SmoothScroll() {
  useEffect(() => {
    const root = document.documentElement;
    const finePointer = window.matchMedia("(pointer: fine)").matches;
    const lenis = new Lenis({
      anchors: { offset: -92 },
      autoRaf: true,
      duration: 0.82,
      easing: EXPONENTIAL_EASE,
      smoothWheel: finePointer,
      stopInertiaOnNavigate: true,
      syncTouch: false,
      prevent: (node) =>
        node instanceof HTMLElement && Boolean(node.closest("[data-native-scroll]")),
    });

    const updateChrome = ({ scroll }: { scroll: number }) => {
      root.dataset.scrolled = scroll > 28 ? "true" : "false";
    };

    root.dataset.scrollMotion = "ready";
    updateChrome({ scroll: window.scrollY });
    lenis.on("scroll", updateChrome);

    return () => {
      lenis.destroy();
      delete root.dataset.scrolled;
      delete root.dataset.scrollMotion;
    };
  }, []);

  return null;
}
