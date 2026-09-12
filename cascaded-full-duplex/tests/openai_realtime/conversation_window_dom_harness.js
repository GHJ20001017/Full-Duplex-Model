// Minimal DOM stub that runs the conversation window page script under Node so
// tests can assert what the browser renders. Usage:
//
//   node conversation_window_dom_harness.js <page.html> <messages.json>
//
// The JSON file is a list of SSE payloads delivered in order on the page's
// EventSource. The harness prints the resulting rows as JSON.

"use strict";

const fs = require("fs");

class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.ownText = "";
    this.children = [];
    this.parentNode = null;
    this.className = "";
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.clientHeight = 0;
    this._classList = null;
  }

  get classList() {
    if (!this._classList) {
      const el = this;
      const names = () => el.className.split(" ").filter(Boolean);
      this._classList = {
        add(name) {
          if (!names().includes(name)) el.className = names().concat(name).join(" ");
        },
        remove(name) {
          el.className = names().filter((n) => n !== name).join(" ");
        },
        toggle(name, on) {
          if (on) this.add(name);
          else this.remove(name);
        },
        contains(name) {
          return names().includes(name);
        },
      };
    }
    return this._classList;
  }

  get isConnected() {
    let node = this;
    while (node.parentNode) node = node.parentNode;
    return node === root;
  }

  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this.ownText = String(value);
    this.children.forEach((child) => {
      child.parentNode = null;
    });
    this.children = [];
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    child.parentNode = null;
    return child;
  }

  remove() {
    if (this.parentNode) this.parentNode.removeChild(this);
  }

  addEventListener() {}

  descendants() {
    const found = [];
    const walk = (node) => {
      node.children.forEach((child) => {
        found.push(child);
        walk(child);
      });
    };
    walk(this);
    return found;
  }

  querySelector(selector) {
    return this._matchAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    return this._matchAll(selector);
  }

  _matchAll(selector) {
    if (!selector.startsWith(".")) throw new Error("unsupported selector: " + selector);
    const name = selector.slice(1);
    return this.descendants().filter((node) => node.classList.contains(name));
  }
}

const root = new El("html");
const body = root.appendChild(new El("body"));
const stream = body.appendChild(new El("main"));
const empty = stream.appendChild(new El("div"));
const byId = { stream, empty };
for (const id of ["status-text", "subtitle", "hint", "title", "empty-text"]) {
  byId[id] = body.appendChild(new El("span"));
}

const sources = [];

global.document = {
  getElementById: (id) => byId[id] || null,
  createElement: (tag) => new El(tag),
  createTextNode: (text) => {
    const node = new El("#text");
    node.ownText = String(text);
    return node;
  },
  body,
};

global.EventSource = class EventSource {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.listeners = {};
    sources.push(this);
  }

  addEventListener(name, handler) {
    (this.listeners[name] = this.listeners[name] || []).push(handler);
  }

  close() {
    this.readyState = 2;
  }
};

const [pagePath, messagesPath] = process.argv.slice(2);
const html = fs.readFileSync(pagePath, "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error("no <script> block found in " + pagePath);

new Function(match[1])();

if (sources.length !== 1) throw new Error("expected one EventSource, saw " + sources.length);
const source = sources[0];

for (const message of JSON.parse(fs.readFileSync(messagesPath, "utf8"))) {
  for (const handler of source.listeners.message || []) {
    handler({ data: JSON.stringify(message) });
  }
}

const rows = stream.children
  .filter((node) => node.classList.contains("row"))
  .map((row) => {
    const bubble = row.querySelector(".bubble");
    return {
      role: ["user", "assistant", "system"].find((name) => row.classList.contains(name)) || null,
      text: bubble ? bubble.textContent : null,
      pending: bubble ? bubble.classList.contains("pending") : null,
    };
  });

process.stdout.write(JSON.stringify(rows));
