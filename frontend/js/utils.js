import { CATEGORY_TITLES, SUBCATEGORY_TITLES } from "./constants.js";

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function humanize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function formatCategory(item) {
  return CATEGORY_TITLES[item.category] || humanize(item.category);
}

export function formatSubcategory(value) {
  return SUBCATEGORY_TITLES[value] || humanize(value);
}

export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

export function formatMoney(value) {
  return Number(value || 0).toFixed(2);
}

export function formatWholeNumber(value) {
  return String(Math.round(Number(value || 0)));
}

export function buildUrl(item) {
  return item.landing_page || `https://example.com/${item.category}/${item.subcategory}/${String(item.brand || "brand").toLowerCase()}`;
}

export function storyHeadline(item, index) {
  const descriptors = [
    "выходит в лидеры персональной ленты",
    "набирает внимание прямо сейчас",
    "попадает в зону высокого интереса",
    "хорошо смотрится в нативной выдаче",
  ];
  return `${item.brand} ${descriptors[index % descriptors.length]}`;
}

export function storySummary(item) {
  return `${formatCategory(item)} / ${formatSubcategory(item.subcategory)}. Формат ${item.banner_format}, цель кампании ${item.campaign_goal}. Прогнозный CTR ${formatPercent(item.pred_ctr)} и quality score ${formatScore(item.quality_score)}.`;
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function normalizeDate(value) {
  return String(value || "").slice(0, 10);
}

export function hashUserId(userId) {
  return Array.from(String(userId || "")).reduce(
    (sum, char, index) => sum + char.charCodeAt(0) * (index + 1),
    0,
  );
}

export function paginate(items, page, pageSize) {
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize) || 1);
  const safePage = clamp(page, 1, totalPages);
  const start = (safePage - 1) * pageSize;
  return {
    items: items.slice(start, start + pageSize),
    page: safePage,
    totalPages,
    totalItems: items.length,
    start: items.length ? start + 1 : 0,
    end: Math.min(start + pageSize, items.length),
  };
}

export function parseApiErrorPayload(payload, fallback) {
  if (!payload?.detail) {
    return fallback;
  }

  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => `${item.loc?.join(".") || "request"}: ${item.msg}`)
      .join(", ");
  }

  return String(payload.detail);
}

export function sortBanners(items, sortKey) {
  const sorted = [...items];
  const sorters = {
    created_desc: (left, right) => String(right.created_at).localeCompare(String(left.created_at)),
    cpm_desc: (left, right) => Number(right.cpm_bid) - Number(left.cpm_bid),
    quality_desc: (left, right) => Number(right.quality_score) - Number(left.quality_score),
    brand_asc: (left, right) => String(left.brand).localeCompare(String(right.brand)),
  };
  return sorted.sort(sorters[sortKey] || sorters.created_desc);
}

export function sortUsers(items, sortKey) {
  const sorted = [...items];
  const sorters = {
    signup_desc: (left, right) => Number(right.signup_days_ago) - Number(left.signup_days_ago),
    age_desc: (left, right) => Number(right.age) - Number(left.age),
    country_asc: (left, right) => String(left.country).localeCompare(String(right.country)),
    premium_first: (left, right) => Number(right.is_premium) - Number(left.is_premium),
  };
  return sorted.sort(sorters[sortKey] || sorters.signup_desc);
}
