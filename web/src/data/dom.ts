// Minimal DOM helpers. No framework, no global styles.

type Attrs = Record<string, string | number | boolean | null | undefined>;
type Child = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function s<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

function setAttrs(el: Element, attrs: Attrs): void {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    el.setAttribute(key, value === true ? "" : String(value));
  }
}

function append(el: Element, children: Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child);
  }
}

/** Collects listeners so dispose can remove them all. */
export class Listeners {
  private removers: (() => void)[] = [];

  on<T extends EventTarget>(
    target: T,
    type: string,
    handler: (event: never) => void,
    options?: AddEventListenerOptions | boolean,
  ): void {
    const fn = handler as unknown as EventListener;
    target.addEventListener(type, fn, options);
    this.removers.push(() => target.removeEventListener(type, fn, options));
  }

  add(remover: () => void): void {
    this.removers.push(remover);
  }

  clear(): void {
    for (const remove of this.removers.splice(0)) remove();
  }
}

let counter = 0;
/** Unique ids for label/for and aria references within one page. */
export function uid(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter}`;
}

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}
