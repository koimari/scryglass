/** Inline before paint: light default, respect an explicit stored choice. */
export function ThemeScript({ nonce }: { nonce?: string }) {
  const code = `(function(){try{var s=localStorage.getItem('scryglass-theme');var t=s==='dark'?'dark':'light';document.documentElement.dataset.theme=t;document.documentElement.style.colorScheme=t;}catch(e){}})();`;
  return <script nonce={nonce} dangerouslySetInnerHTML={{ __html: code }} />;
}
