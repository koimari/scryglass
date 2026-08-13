/** Cactus/Needle on-device client (WASM). Loads the engine + weights from the
 * configured URLs and exposes an OpenAI-compatible chat/tool-calling entry
 * point. Returns null when the engine or weights are unavailable so the
 * caller degrades to the deterministic router.
 */

export type NeedleClientConfig = {
  engineUrl: string;
  modelUrl: string;
};

export type NeedleClient = {
  call: (prompt: string) => Promise<string>;
};

export async function createNeedleClient(
  config: NeedleClientConfig,
): Promise<NeedleClient | null> {
  // Fetch the engine and weights; a 404/network failure degrades gracefully.
  let engine: ArrayBuffer;
  let weights: ArrayBuffer;
  try {
    [engine, weights] = await Promise.all([
      fetch(config.engineUrl).then((response) => {
        if (!response.ok) throw new Error(`engine ${response.status}`);
        return response.arrayBuffer();
      }),
      fetch(config.modelUrl).then((response) => {
        if (!response.ok) throw new Error(`model ${response.status}`);
        return response.arrayBuffer();
      }),
    ]);
  } catch {
    return null;
  }

  try {
    // Cactus exposes an OpenAI-compatible chat completion API from the WASM
    // engine. The exact symbol is resolved at runtime; if the binary does not
    // expose it, degrade to the deterministic router.
    const { instance } = await WebAssembly.instantiate(engine, {});
    const exports = instance.exports as Record<string, unknown>;
    const chatFunction = (exports as Record<string, CallableFunction>)["chat"];
    if (typeof chatFunction !== "function") return null;
    const modelLoad = exports["model_load"] as CallableFunction | undefined;
    if (typeof modelLoad === "function") modelLoad(new Uint8Array(weights));
    return {
      call: async (prompt: string) => String(chatFunction(prompt) ?? ""),
    };
  } catch {
    return null;
  }
}
