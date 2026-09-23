const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let kind = "both";
let toastTimer;

async function api(path, options) {
  const response = await fetch(path, options);
  if (response.ok) return response.json();
  const error = await response.json().catch(() => ({detail: "Request failed"}));
  throw new Error(error.detail || "Request failed");
}

function message(text) {
  $("#toast").textContent = text;
  $("#toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("show"), 2800);
}

function safePoster(value) {
  try {
    const url = new URL(value, location.origin);
    return url.protocol === "https:" && url.hostname === "image.tmdb.org" ? url.href : null;
  } catch { return null; }
}

function button(label, handler) {
  const node = document.createElement("button");
  node.textContent = label;
  node.onclick = handler;
  return node;
}

function card(item, actions = []) {
  const node = document.createElement("article");
  node.className = "card";
  const source = safePoster(item.poster);
  if (source) {
    const image = document.createElement("img");
    image.className = "poster"; image.loading = "lazy"; image.src = source; image.alt = "";
    node.append(image);
  } else {
    const image = document.createElement("div"); image.className = "poster"; node.append(image);
  }
  const meta = document.createElement("div"); meta.className = "meta";
  const title = document.createElement("div"); title.className = "title"; title.textContent = item.title || "Untitled";
  const detail = document.createElement("div"); detail.className = "detail";
  detail.textContent = [item.year, item.language_name, item.kind === "tv" ? "series" : ""].filter(Boolean).join(" · ");
  meta.append(title, detail);
  if (item.score != null) {
    const score = document.createElement("div"); score.className = "score";
    score.textContent = `${item.likely_like === false ? "Less likely" : "Likely like"} · ${Number(item.score).toFixed(1)}/10`;
    meta.append(score);
  }
  node.append(meta);
  if (actions.length) { const box = document.createElement("div"); box.className = "actions"; actions.forEach(([label, fn]) => box.append(button(label, fn))); node.append(box); }
  return node;
}

async function rate(item, verdict, node) {
  try {
    await api(item.external ? "/api/add" : "/api/rate", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(item.external ? {tmdb_id: item.tmdb_id, kind: item.kind, verdict} : {item_id: item.item_id, verdict})});
    node.remove(); message(`${item.title} — ${verdict}`);
  } catch (error) { message(error.message); }
}

function ratingCard(item) {
  let node;
  const actions = [["♥", () => rate(item, "love", node)], ["+", () => rate(item, "like", node)], ["~", () => rate(item, "meh", node)], ["−", () => rate(item, "dislike", node)], ["?", () => rate(item, "unseen", node)]];
  if (item.item_id != null) actions.push(["Predict", () => showPrediction(item, node)]);
  node = card(item, actions);
  return node;
}

async function showPrediction(item, node) {
  try {
    const prediction = await api(`/api/predict/${item.item_id}`);
    node.replaceWith(card({...item, ...prediction}, []));
  } catch (error) { message(error.message); }
}

async function libraryAction(itemId, action) {
  try { await api("/api/library/actions", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({item_id: itemId, action})}); message(action === "save" ? "Saved to your library" : action === "watched" ? "Marked watched" : "Removed"); loadLibrary(); } catch (error) { message(error.message); }
}

async function generate() {
  const orb = $("#orb"); orb.classList.add("thinking"); $("#generate").disabled = true;
  $("#message").textContent = "Listening to the shape of your taste…";
  try {
    const slate = await api("/api/recommendations/slate", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({kind, k: 10})});
    const grid = $("#slategrid"); grid.replaceChildren();
    slate.items.forEach((item) => grid.append(card(item, [["Save", () => libraryAction(item.item_id, "save")], ["Watched", () => libraryAction(item.item_id, "watched")]])));
    $("#hero").classList.add("hide"); $("#slate").classList.remove("hide");
  } catch (error) { $("#message").textContent = error.message.includes("three explicit") ? "Give me three ratings first, then I can begin." : "I could not form a slate right now."; message(error.message); }
  finally { orb.classList.remove("thinking"); $("#generate").disabled = false; }
}

async function loadExplore() {
  const feed = await api("/api/feed?limit=60&years=2");
  const grid = $("#feed"); grid.replaceChildren(); feed.items.forEach((item) => grid.append(ratingCard(item)));
}

async function search(query) {
  if (!query.trim()) return $("#searchresults").replaceChildren();
  try { const data = await api(`/api/search?q=${encodeURIComponent(query)}`); const grid = $("#searchresults"); grid.replaceChildren(); [...data.catalogue, ...data.tmdb.map((item) => ({...item, external: true}))].forEach((item) => grid.append(ratingCard(item))); } catch (error) { message(error.message); }
}

async function loadLibrary() {
  try { const data = await api("/api/library"); const root = $("#library-content"); root.replaceChildren(); [["Saved", data.saved, "remove"], ["Watched", data.watched, null], ["Rated", data.rated, null]].forEach(([label, items, action]) => { const section = document.createElement("section"); section.className = "result"; const heading = document.createElement("h2"); heading.textContent = label; section.append(heading); const grid = document.createElement("div"); grid.className = "grid"; items.forEach((item) => grid.append(card(item, action ? [["Remove", () => libraryAction(item.item_id, action)], ["Watched", () => libraryAction(item.item_id, "watched")]] : []))); section.append(grid); root.append(section); }); } catch (error) { message(error.message); }
}

function setView(name) { $$(".nav button").forEach((node) => node.classList.toggle("active", node.dataset.view === name)); $$(".view").forEach((node) => node.classList.toggle("active", node.id === name)); if (name === "explore") loadExplore().catch((error) => message(error.message)); if (name === "library") loadLibrary(); }

$$(".nav button").forEach((node) => node.onclick = () => setView(node.dataset.view));
$$(".switch button").forEach((node) => node.onclick = () => { $$(".switch button").forEach((button) => button.classList.remove("active")); node.classList.add("active"); kind = node.dataset.kind; });
$("#generate").onclick = generate;
$("#again").onclick = () => { $("#hero").classList.remove("hide"); $("#slate").classList.add("hide"); };
$("#search").oninput = (event) => { clearTimeout(window.searchTimer); window.searchTimer = setTimeout(() => search(event.target.value), 280); };
$("#undo").onclick = async () => { try { const result = await api("/api/undo", {method: "POST"}); message(result.ok ? `Undone: ${result.title || result.item_id}` : "Nothing to undo"); loadExplore(); } catch (error) { message(error.message); } };
$("#check").onclick = () => $("#modal").showModal(); $("#close").onclick = () => $("#modal").close();
$("#check-submit").onclick = async () => { try { const titles = $("#check-input").value.split("\n").map((value) => value.trim()).filter(Boolean); const data = await api("/api/check-titles", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({titles})}); const root = $("#check-results"); root.replaceChildren(); data.results.forEach((result) => { const heading = document.createElement("p"); heading.textContent = result.query; root.append(heading); const grid = document.createElement("div"); grid.className = "grid"; [...result.catalogue, ...result.tmdb].forEach((item) => grid.append(ratingCard(item))); root.append(grid); }); } catch (error) { message(error.message); } };
api("/api/progress").then((progress) => { if (progress.rated < 3) $("#message").textContent = "Give me three ratings, then I can begin to understand your taste."; }).catch((error) => message(error.message));
