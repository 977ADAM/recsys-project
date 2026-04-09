import { API_ROOT, DEMO_RESPONSE } from "./constants.js";
import { hashUserId, normalizeDate, parseApiErrorPayload } from "./utils.js";

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = parseApiErrorPayload(await response.json(), detail);
    } catch {
      // Keep generic message when the server does not return JSON.
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

export function normalizeBanner(banner) {
  return {
    ...banner,
    cpm_bid: Number(banner.cpm_bid),
    quality_score: Number(banner.quality_score),
    created_at: normalizeDate(banner.created_at),
  };
}

export function normalizeUser(user) {
  return {
    ...user,
    age: Number(user.age),
    signup_days_ago: Number(user.signup_days_ago),
    is_premium: Boolean(user.is_premium),
  };
}

export function normalizeRecommendationResponse(response, fallbackCandidateMode = "all banners") {
  return {
    ...response,
    as_of_date: normalizeDate(response.as_of_date || DEMO_RESPONSE.as_of_date),
    candidate_mode: response.candidate_mode || fallbackCandidateMode,
    retrieval_used: response.candidate_mode === "retrieval + ranking",
    top_k: response.items?.length || 0,
    items: (response.items || []).map((item) => ({
      ...item,
      cpm_bid: Number(item.cpm_bid),
      quality_score: Number(item.quality_score),
      pred_ctr: Number(item.pred_ctr),
      final_score: Number(item.final_score),
    })),
  };
}

export function buildLocalResponse(payload, uiControls) {
  const sourceItems = [...DEMO_RESPONSE.items];
  const offset = sourceItems.length === 0 ? 0 : hashUserId(payload.user_id) % sourceItems.length;
  const rotatedItems = sourceItems.map((_, index) => sourceItems[(index + offset) % sourceItems.length]);
  const sortKey = payload.score_mode === "ctr" ? "pred_ctr" : "final_score";
  const sortedItems = [...rotatedItems].sort((left, right) => Number(right[sortKey]) - Number(left[sortKey]));
  const activeItems = payload.only_active ? sortedItems.filter((item) => item.is_active) : sortedItems;
  const limit = Math.max(1, Number(payload.top_k) || DEMO_RESPONSE.top_k);
  const items = activeItems.slice(0, limit);

  return normalizeRecommendationResponse(
    {
      user_id: payload.user_id,
      as_of_date: DEMO_RESPONSE.as_of_date,
      score_mode: payload.score_mode,
      candidate_mode: uiControls.candidateMode,
      model_type: "local-demo-feed",
      items,
    },
    uiControls.candidateMode,
  );
}

export async function fetchRecommendations(payload, uiControls) {
  const response = await requestJson("/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizeRecommendationResponse(response, uiControls.candidateMode);
}

export async function fetchBanners() {
  const response = await requestJson("/banners", { method: "GET" });
  return (response.banners || []).map(normalizeBanner);
}

export async function createBanner(payload) {
  return normalizeBanner(
    await requestJson("/banners", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function updateBanner(bannerId, payload) {
  return normalizeBanner(
    await requestJson(`/banners/${encodeURIComponent(bannerId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function deleteBanner(bannerId) {
  return requestJson(`/banners/${encodeURIComponent(bannerId)}`, { method: "DELETE" });
}

export async function fetchUsers() {
  const response = await requestJson("/users", { method: "GET" });
  return (response.users || []).map(normalizeUser);
}

export async function createUser(payload) {
  return normalizeUser(
    await requestJson("/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function updateUser(userId, payload) {
  return normalizeUser(
    await requestJson(`/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function deleteUser(userId) {
  return requestJson(`/users/${encodeURIComponent(userId)}`, { method: "DELETE" });
}
