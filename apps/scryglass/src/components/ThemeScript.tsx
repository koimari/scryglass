/** Inline before paint: system default, respect stored light/dark. */
export function ThemeScript() {
  const code = `(function(){try{var k='scryglass-theme';var s=localStorage.getItem(k);var c=(s==='light'||s==='dark')?s:'system';var d=c==='dark'||(c==='system'&&window.matchMedia('(prefers-color-scheme: dark)').matches);var t=d?'dark':'light';document.documentElement.dataset.theme=t;document.documentElement.style.colorScheme=t;}catch(e){}})();`;
  return <script dangerouslySetInnerHTML={{ __html: code }} />;
}
