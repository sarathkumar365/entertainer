/**
 * One place that knows how to talk to the engine.
 *
 * The token, when there is one, arrives in the query string and the server
 * sets a cookie from it, so fetch needs no header — but same-origin requests
 * must send credentials for that cookie to be included.
 */
async function request(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* not JSON; the text is the message */
    }
    const error = new Error(detail || `${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

const post = (path, body) =>
  request(path, { method: "POST", body: JSON.stringify(body ?? {}) });

export const api = {
  mode: () => request("/api/mode"),
  progress: () => request("/api/progress"),
  languages: () => request("/api/languages"),
  feed: (params) => request(`/api/feed?${new URLSearchParams(params)}`),
  rated: (limit = 200) => request(`/api/rated?limit=${limit}`),
  taste: (axes = 6) => request(`/api/taste?axes=${axes}`),
  audit: () => request("/api/audit"),
  similar: (itemId, k = 8) => request(`/api/similar/${itemId}?k=${k}`),
  slate: (k = 10) => post(`/api/recommendations/slate?k=${k}`),
  rate: (body) => post("/api/rate", body),
  add: (body) => post("/api/add", body),
  undo: () => post("/api/undo"),
  predict: (itemId) => request(`/api/predict/${itemId}`),
  validationCases: () => request("/api/validation/cases"),
  validationSummary: () => request("/api/validation/summary"),
  seal: (itemIds) => post("/api/validation/seal", { item_ids: itemIds }),
  reveal: (caseId, verdict) => post(`/api/validation/${caseId}/reveal`, { verdict }),
};

/** The seven the engine understands, in the order they read as a scale. */
export const VERDICTS = [
  { key: "love", label: "Loved" },
  { key: "like", label: "Liked" },
  { key: "ok", label: "Fine" },
  { key: "meh", label: "Meh" },
  { key: "dislike", label: "Disliked" },
  { key: "hate", label: "Hated" },
  { key: "unseen", label: "Not seen" },
];
