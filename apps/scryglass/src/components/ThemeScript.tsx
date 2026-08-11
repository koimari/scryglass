/** Inline before paint: light default, respect an explicit stored choice. */
export function ThemeScript() {
  const code = `(function(){try{var s=localStorage.getItem('scryglass-theme');var t=s==='dark'?'dark':'light';document.documentElement.dataset.theme=t;document.documentElement.style.colorScheme=t;}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: code }} />;
}
