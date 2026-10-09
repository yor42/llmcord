// Mirror-behind-textarea highlighting. Textareas opt in with the ll-macro class on their q-field; each one is
// attached only once it is visible or focused, so collapsed prompt blocks cost nothing.
// One shared controller serves every MacroHighlight instance on the page (refcounted), so extra instances never double-attach.
const RX = /\{\{(.*?)\}\}/gs; // same pattern as prompts.substitute
const SELECTOR = ".ll-macro textarea";
const COPIED = ["fontFamily", "fontSize", "fontWeight", "fontStyle", "lineHeight", "letterSpacing", "paddingTop", "paddingBottom",
  "paddingLeft", "paddingRight", "whiteSpace", "wordBreak", "overflowWrap", "tabSize", "textIndent"];
const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

let shared = null;

function createController(names) {
  const attached = new Map(); // textarea -> undo
  const seen = new WeakSet();
  const prune = () => {
    for (const [ta, undo] of attached) if (!ta.isConnected) { attached.delete(ta); undo(); }
  };
  const attach = (ta) => {
    if (attached.has(ta) || !ta.isConnected) return;
    const host = ta.parentElement;
    const mirror = document.createElement("div");
    mirror.className = "mh-mirror";
    mirror.setAttribute("aria-hidden", "true");
    const hostPosition = host.style.position;
    const paint = () => {
      const style = getComputedStyle(ta);
      for (const name of COPIED) mirror.style[name] = style[name];
      mirror.style.top = ta.offsetTop + ta.clientTop + "px";
      mirror.style.left = ta.offsetLeft + ta.clientLeft + "px";
      mirror.style.width = ta.clientWidth + "px";
      mirror.style.height = ta.clientHeight + "px";
      const text = ta.value;
      let html = "", last = 0;
      for (const match of text.matchAll(RX)) {
        html += escape(text.slice(last, match.index)) + `<span class="${names.has(match[1]) ? "mh-known" : "mh-unknown"}">${escape(match[0])}</span>`;
        last = match.index + match[0].length;
      }
      mirror.innerHTML = html + escape(text.slice(last)) + (text.endsWith("\n") ? "​" : "");
      mirror.scrollTop = ta.scrollTop;
    };
    const sync = () => { mirror.scrollTop = ta.scrollTop; };
    const resize = new ResizeObserver(paint);
    const native = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
    const undo = () => {
      ta.removeEventListener("input", paint);
      ta.removeEventListener("scroll", sync);
      resize.disconnect();
      delete ta.value;
      ta.classList.remove("mh-ta");
      host.style.position = hostPosition;
      mirror.remove();
    };
    try {
      if (getComputedStyle(host).position === "static") host.style.position = "relative";
      host.insertBefore(mirror, ta);
      paint();
      ta.classList.add("mh-ta"); // text turns transparent only once a mirror has painted
      ta.addEventListener("input", paint);
      ta.addEventListener("scroll", sync);
      resize.observe(ta);
      // Vue sets el.value for server-side changes (import, revert, discard), which fires no input event.
      Object.defineProperty(ta, "value", { configurable: true, get() { return native.get.call(this); }, set(value) { native.set.call(this, value); paint(); } });
      attached.set(ta, undo);
    } catch (error) {
      undo();
      console.error(error);
    }
  };
  const visible = new IntersectionObserver((entries) => {
    for (const entry of entries) if (entry.isIntersecting) { visible.unobserve(entry.target); attach(entry.target); }
  });
  const scan = (root) => {
    const found = root.matches(SELECTOR) ? [root] : [];
    found.push(...root.querySelectorAll(SELECTOR));
    for (const ta of found) if (!seen.has(ta)) { seen.add(ta); visible.observe(ta); }
  };
  const watcher = new MutationObserver((records) => {
    for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) scan(node);
    prune();
  });
  const onFocus = (event) => { if (event.target.matches?.(SELECTOR)) attach(event.target); };
  watcher.observe(document.body, { childList: true, subtree: true });
  document.addEventListener("focusin", onFocus);
  scan(document.body);
  return {
    count: 0,
    destroy() {
      visible.disconnect();
      watcher.disconnect();
      document.removeEventListener("focusin", onFocus);
      for (const undo of attached.values()) undo();
      attached.clear();
    },
  };
}

export default {
  template: `<span hidden></span>`,
  props: { known: Array },
  mounted() {
    shared ??= createController(new Set(this.known));
    shared.count++;
  },
  beforeUnmount() {
    if (--shared.count === 0) { shared.destroy(); shared = null; }
  },
};
