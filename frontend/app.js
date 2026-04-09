const API_ROOT = "/api/v1";

const DEMO_RESPONSE = {
  user_id: "u_00007",
  as_of_date: "2026-04-09",
  score_mode: "value",
  candidate_mode: "retrieval + ranking",
  model_type: "demo-editorial-ranker",
  artifacts_dir: "frontend-demo",
  retrieval_used: true,
  online_state_applied: false,
  top_k: 8,
  items: [
    {
      banner_id: "bnr_demo_001",
      brand: "SkyJet",
      category: "travel",
      subcategory: "car_rental",
      banner_format: "native",
      campaign_goal: "traffic",
      pred_ctr: 0.081,
      final_score: 0.912,
      cpm_bid: 182.4,
      quality_score: 0.947,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 3,
      banner_ctr_prior: 0.063,
      user_ctr_prior: 0.052,
      user_subcategory_ctr_prior: 0.071,
      user_banner_ctr_prior: 0.048,
      served_impressions_total: 18420,
      served_clicks_total: 931,
      is_active: true,
      landing_page: "https://example.com/travel/car-rental/skyjet",
    },
    {
      banner_id: "bnr_demo_002",
      brand: "PixelOne",
      category: "electronics",
      subcategory: "smartphones",
      banner_format: "banner",
      campaign_goal: "purchase",
      pred_ctr: 0.067,
      final_score: 0.861,
      cpm_bid: 205.15,
      quality_score: 0.918,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 2,
      banner_ctr_prior: 0.059,
      user_ctr_prior: 0.052,
      user_subcategory_ctr_prior: 0.064,
      user_banner_ctr_prior: 0.045,
      served_impressions_total: 15280,
      served_clicks_total: 774,
      is_active: true,
      landing_page: "https://example.com/electronics/smartphones/pixelone",
    },
    {
      banner_id: "bnr_demo_003",
      brand: "CodeSpring",
      category: "education",
      subcategory: "coding",
      banner_format: "native",
      campaign_goal: "lead_gen",
      pred_ctr: 0.059,
      final_score: 0.833,
      cpm_bid: 168.9,
      quality_score: 0.902,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 4,
      banner_ctr_prior: 0.055,
      user_ctr_prior: 0.052,
      user_subcategory_ctr_prior: 0.073,
      user_banner_ctr_prior: 0.038,
      served_impressions_total: 12840,
      served_clicks_total: 609,
      is_active: true,
      landing_page: "https://example.com/education/coding/codespring",
    },
    {
      banner_id: "bnr_demo_004",
      brand: "CityBite",
      category: "food",
      subcategory: "snacks",
      banner_format: "static",
      campaign_goal: "awareness",
      pred_ctr: 0.043,
      final_score: 0.741,
      cpm_bid: 121.6,
      quality_score: 0.876,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.041,
      user_ctr_prior: 0.052,
      user_subcategory_ctr_prior: 0.039,
      user_banner_ctr_prior: 0.031,
      served_impressions_total: 21400,
      served_clicks_total: 801,
      is_active: true,
      landing_page: "https://example.com/food/snacks/citybite",
    },
  ],
};

const DEMO_BANNERS = [
  {
    banner_id: "bnr_demo_001",
    brand: "SkyJet",
    category: "travel",
    subcategory: "car_rental",
    banner_format: "native",
    campaign_goal: "traffic",
    target_gender: "U",
    target_age_min: 22,
    target_age_max: 45,
    cpm_bid: 182.4,
    quality_score: 0.947,
    created_at: "2026-03-11",
    is_active: true,
    landing_page: "https://example.com/travel/car-rental/skyjet",
  },
  {
    banner_id: "bnr_demo_002",
    brand: "PixelOne",
    category: "electronics",
    subcategory: "smartphones",
    banner_format: "animated",
    campaign_goal: "purchase",
    target_gender: "U",
    target_age_min: 18,
    target_age_max: 39,
    cpm_bid: 205.15,
    quality_score: 0.918,
    created_at: "2026-02-25",
    is_active: true,
    landing_page: "https://example.com/electronics/smartphones/pixelone",
  },
  {
    banner_id: "bnr_demo_003",
    brand: "CodeSpring",
    category: "education",
    subcategory: "coding",
    banner_format: "native",
    campaign_goal: "lead_gen",
    target_gender: "U",
    target_age_min: 19,
    target_age_max: 41,
    cpm_bid: 168.9,
    quality_score: 0.902,
    created_at: "2026-01-19",
    is_active: true,
    landing_page: "https://example.com/education/coding/codespring",
  },
  {
    banner_id: "bnr_demo_004",
    brand: "CityBite",
    category: "food",
    subcategory: "snacks",
    banner_format: "static",
    campaign_goal: "awareness",
    target_gender: "F",
    target_age_min: 20,
    target_age_max: 38,
    cpm_bid: 121.6,
    quality_score: 0.876,
    created_at: "2026-02-08",
    is_active: false,
    landing_page: "https://example.com/food/snacks/citybite",
  },
];

const DEMO_USERS = [
  {
    user_id: "u_00007",
    age: 29,
    gender: "U",
    city_tier: "tier_1",
    device_os: "ios",
    platform: "mobile_app",
    income_band: "high",
    activity_segment: "power",
    interest_1: "travel",
    interest_2: "gadgets",
    interest_3: "food",
    country: "RU",
    signup_days_ago: 420,
    is_premium: true,
  },
  {
    user_id: "u_00021",
    age: 34,
    gender: "F",
    city_tier: "tier_1",
    device_os: "android",
    platform: "mobile_app",
    income_band: "mid",
    activity_segment: "steady",
    interest_1: "education",
    interest_2: "design",
    interest_3: "cinema",
    country: "RU",
    signup_days_ago: 180,
    is_premium: false,
  },
  {
    user_id: "u_00032",
    age: 24,
    gender: "M",
    city_tier: "tier_2",
    device_os: "ios",
    platform: "web",
    income_band: "mid",
    activity_segment: "active",
    interest_1: "sports",
    interest_2: "fashion",
    interest_3: "electronics",
    country: "KZ",
    signup_days_ago: 90,
    is_premium: true,
  },
  {
    user_id: "u_00057",
    age: 41,
    gender: "U",
    city_tier: "tier_3",
    device_os: "android",
    platform: "mobile_web",
    income_band: "low",
    activity_segment: "casual",
    interest_1: "finance",
    interest_2: "home",
    interest_3: "travel",
    country: "BY",
    signup_days_ago: 760,
    is_premium: false,
  },
];

const CATEGORY_TITLES = {
  education: "Образование",
  travel: "Путешествия",
  electronics: "Технологии",
  entertainment: "Культура",
  food: "Город",
  sports: "Спорт",
  fashion: "Стиль",
  finance: "Финансы",
  home: "Дом",
};

const SUBCATEGORY_TITLES = {
  coding: "курсы и практикумы",
  design: "дизайн и визуальные школы",
  car_rental: "каршеринг и аренда",
  smartphones: "смартфоны и экосистемы",
  cinema: "кино и премьеры",
  events: "городские события",
  snacks: "быстрые покупки",
  running: "бег и экипировка",
};

const placementBlueprint = [
  {
    size: "970x250",
    title: "Leaderboard",
    copy: "Большой верхний баннер под шапкой для takeover-дня, запуска акции или флагманской кампании.",
  },
  {
    size: "300x600",
    title: "Right Rail Sticky",
    copy: "Правый rail для охватных историй и длинных сценариев ретаргетинга.",
  },
  {
    size: "728x90",
    title: "In-Feed Midroll",
    copy: "Встраиваемый слот между карточками, который не ломает ритм ленты.",
  },
  {
    size: "Native Card",
    title: "Sponsored Story",
    copy: "Редакционно выглядящая карточка с брендом, CTA и нативной подачей.",
  },
];

const state = {
  view: "showcase",
  response: DEMO_RESPONSE,
  source: "demo",
  banners: [...DEMO_BANNERS],
  bannersSource: "demo",
  selectedBannerId: null,
  bannerFormMode: "create",
  bannerSearch: "",
  bannerStatusFilter: "all",
  users: [...DEMO_USERS],
  usersSource: "demo",
  selectedUserId: null,
  userFormMode: "create",
  userSearch: "",
  userPremiumFilter: "all",
};

const elements = {
  apiStatus: document.querySelector("#api-status"),
  refreshButton: document.querySelector("#refresh-button"),
  demoButton: document.querySelector("#demo-button"),
  showcaseTab: document.querySelector("#showcase-tab"),
  adminTab: document.querySelector("#admin-tab"),
  usersTab: document.querySelector("#users-tab"),
  showcaseView: document.querySelector("#showcase-view"),
  adminView: document.querySelector("#admin-view"),
  usersView: document.querySelector("#users-view"),
  controlsForm: document.querySelector("#controls-form"),
  userId: document.querySelector("#user-id"),
  topK: document.querySelector("#top-k"),
  candidateMode: document.querySelector("#candidate-mode"),
  scoreMode: document.querySelector("#score-mode"),
  retrievalTopN: document.querySelector("#retrieval-top-n"),
  onlyActive: document.querySelector("#only-active"),
  excludeSeen: document.querySelector("#exclude-seen"),
  heroStory: document.querySelector("#hero-story"),
  newsBriefing: document.querySelector("#news-briefing"),
  feedGrid: document.querySelector("#feed-grid"),
  nativeBand: document.querySelector("#native-band"),
  sponsoredRail: document.querySelector("#sponsored-rail"),
  profileCard: document.querySelector("#profile-card"),
  metaGrid: document.querySelector("#meta-grid"),
  placementGrid: document.querySelector("#placement-grid"),
  midrollSlot: document.querySelector("#midroll-slot"),
  recommendationsCount: document.querySelector("#recommendations-count"),
  recommendationsSource: document.querySelector("#recommendations-source"),
  topBrand: document.querySelector("#top-brand"),
  topBrandCopy: document.querySelector("#top-brand-copy"),
  averageCtr: document.querySelector("#average-ctr"),
  inventoryTotal: document.querySelector("#inventory-total"),
  inventoryCopy: document.querySelector("#inventory-copy"),
  leaderboardCode: document.querySelector("#leaderboard-code"),
  railCode: document.querySelector("#rail-code"),
  bannerSearch: document.querySelector("#banner-search"),
  bannerStatusFilter: document.querySelector("#banner-status-filter"),
  reloadBannersButton: document.querySelector("#reload-banners-button"),
  newBannerButton: document.querySelector("#new-banner-button"),
  bannersTableBody: document.querySelector("#banners-table-body"),
  bannersTableEmpty: document.querySelector("#banners-table-empty"),
  bannersListCaption: document.querySelector("#banners-list-caption"),
  adminTotalBanners: document.querySelector("#admin-total-banners"),
  adminActiveBanners: document.querySelector("#admin-active-banners"),
  adminAverageCpm: document.querySelector("#admin-average-cpm"),
  adminSelectedBanner: document.querySelector("#admin-selected-banner"),
  bannerForm: document.querySelector("#banner-form"),
  bannerFormTitle: document.querySelector("#banner-form-title"),
  bannerFormCaption: document.querySelector("#banner-form-caption"),
  bannerId: document.querySelector("#banner-id"),
  bannerBrand: document.querySelector("#banner-brand"),
  bannerCategory: document.querySelector("#banner-category"),
  bannerSubcategory: document.querySelector("#banner-subcategory"),
  bannerFormat: document.querySelector("#banner-format"),
  bannerGoal: document.querySelector("#banner-goal"),
  bannerGender: document.querySelector("#banner-gender"),
  bannerAgeMin: document.querySelector("#banner-age-min"),
  bannerAgeMax: document.querySelector("#banner-age-max"),
  bannerCpm: document.querySelector("#banner-cpm"),
  bannerQuality: document.querySelector("#banner-quality"),
  bannerCreatedAt: document.querySelector("#banner-created-at"),
  bannerLandingPage: document.querySelector("#banner-landing-page"),
  bannerIsActive: document.querySelector("#banner-is-active"),
  saveBannerButton: document.querySelector("#save-banner-button"),
  resetBannerButton: document.querySelector("#reset-banner-button"),
  deleteBannerButton: document.querySelector("#delete-banner-button"),
  userSearch: document.querySelector("#user-search"),
  userPremiumFilter: document.querySelector("#user-premium-filter"),
  reloadUsersButton: document.querySelector("#reload-users-button"),
  newUserButton: document.querySelector("#new-user-button"),
  usersTableBody: document.querySelector("#users-table-body"),
  usersTableEmpty: document.querySelector("#users-table-empty"),
  usersListCaption: document.querySelector("#users-list-caption"),
  adminTotalUsers: document.querySelector("#admin-total-users"),
  adminPremiumUsers: document.querySelector("#admin-premium-users"),
  adminAverageAge: document.querySelector("#admin-average-age"),
  adminSelectedUser: document.querySelector("#admin-selected-user"),
  userForm: document.querySelector("#user-form"),
  userFormTitle: document.querySelector("#user-form-title"),
  userFormCaption: document.querySelector("#user-form-caption"),
  manageUserId: document.querySelector("#manage-user-id"),
  manageUserAge: document.querySelector("#manage-user-age"),
  manageUserGender: document.querySelector("#manage-user-gender"),
  manageUserCityTier: document.querySelector("#manage-user-city-tier"),
  manageUserDeviceOs: document.querySelector("#manage-user-device-os"),
  manageUserPlatform: document.querySelector("#manage-user-platform"),
  manageUserIncomeBand: document.querySelector("#manage-user-income-band"),
  manageUserActivitySegment: document.querySelector("#manage-user-activity-segment"),
  manageUserInterest1: document.querySelector("#manage-user-interest-1"),
  manageUserInterest2: document.querySelector("#manage-user-interest-2"),
  manageUserInterest3: document.querySelector("#manage-user-interest-3"),
  manageUserCountry: document.querySelector("#manage-user-country"),
  manageUserSignupDaysAgo: document.querySelector("#manage-user-signup-days-ago"),
  manageUserIsPremium: document.querySelector("#manage-user-is-premium"),
  saveUserButton: document.querySelector("#save-user-button"),
  resetUserButton: document.querySelector("#reset-user-button"),
  deleteUserButton: document.querySelector("#delete-user-button"),
  toast: document.querySelector("#toast"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function humanize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function formatCategory(item) {
  return CATEGORY_TITLES[item.category] || humanize(item.category);
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function formatMoney(value) {
  return Number(value || 0).toFixed(2);
}

function formatWholeNumber(value) {
  return String(Math.round(Number(value || 0)));
}

function storyHeadline(item, index) {
  const descriptors = [
    "выходит в лидеры персональной ленты",
    "набирает внимание прямо сейчас",
    "попадает в зону высокого интереса",
    "хорошо смотрится в нативной выдаче",
  ];
  return `${item.brand} ${descriptors[index % descriptors.length]}`;
}

function storySummary(item) {
  const subcategory = SUBCATEGORY_TITLES[item.subcategory] || humanize(item.subcategory);
  return `${formatCategory(item)} / ${subcategory}. Формат ${item.banner_format}, цель кампании ${item.campaign_goal}. Прогнозный CTR ${formatPercent(item.pred_ctr)} и quality score ${formatScore(item.quality_score)}.`;
}

function buildUrl(item) {
  return item.landing_page || `https://example.com/${item.category}/${item.subcategory}/${item.brand.toLowerCase()}`;
}

function setStatus(label, className) {
  elements.apiStatus.textContent = label;
  elements.apiStatus.className = `status-chip ${className}`.trim();
}

let toastTimer = null;
function showToast(message, tone = "info") {
  elements.toast.textContent = message;
  elements.toast.className = `toast is-visible tone-${tone}`;
  elements.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
    elements.toast.className = "toast";
  }, 2600);
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join(", ") : payload.detail;
      }
    } catch {
      // Ignore JSON parsing problems and keep generic detail.
    }
    throw new Error(detail);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function buildRequestPayload() {
  return {
    user_id: elements.userId.value.trim(),
    top_k: Number(elements.topK.value),
    candidate_mode: elements.candidateMode.value,
    score_mode: elements.scoreMode.value,
    retrieval_top_n: Number(elements.retrievalTopN.value),
    only_active: elements.onlyActive.checked,
    exclude_seen: elements.excludeSeen.checked,
  };
}

function hashUserId(userId) {
  return Array.from(String(userId || ""))
    .reduce((sum, char, index) => sum + char.charCodeAt(0) * (index + 1), 0);
}

function buildLocalResponse(payload) {
  const sourceItems = [...DEMO_RESPONSE.items];
  const offset = sourceItems.length === 0 ? 0 : hashUserId(payload.user_id) % sourceItems.length;
  const rotatedItems = sourceItems.map((_, index) => sourceItems[(index + offset) % sourceItems.length]);
  const sortKey = payload.score_mode === "ctr" ? "pred_ctr" : "final_score";
  const sortedItems = [...rotatedItems].sort((left, right) => Number(right[sortKey]) - Number(left[sortKey]));
  const activeItems = payload.only_active ? sortedItems.filter((item) => item.is_active) : sortedItems;
  const limit = Math.max(1, Number(payload.top_k) || DEMO_RESPONSE.top_k);
  const items = activeItems.slice(0, limit);

  return {
    user_id: payload.user_id,
    as_of_date: DEMO_RESPONSE.as_of_date,
    score_mode: payload.score_mode,
    candidate_mode: payload.candidate_mode,
    model_type: "local-demo-feed",
    artifacts_dir: "frontend-local",
    retrieval_used: payload.candidate_mode === "retrieval + ranking",
    online_state_applied: false,
    top_k: items.length,
    items,
  };
}

async function fetchRecommendations(payload) {
  const response = await requestJson("/recommendations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return {
    ...response,
    retrieval_used: response.candidate_mode === "retrieval + ranking",
    online_state_applied: false,
    top_k: response.items.length,
    artifacts_dir: response.artifacts_dir || "backend-generated",
  };
}

async function fetchBanners() {
  const response = await requestJson("/banners", {
    method: "GET",
  });
  return response.banners || [];
}

async function createBanner(payload) {
  return requestJson("/banners", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function updateBanner(bannerId, payload) {
  return requestJson(`/banners/${encodeURIComponent(bannerId)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function deleteBanner(bannerId) {
  return requestJson(`/banners/${encodeURIComponent(bannerId)}`, {
    method: "DELETE",
  });
}

async function fetchUsers() {
  const response = await requestJson("/users", {
    method: "GET",
  });
  return response.users || [];
}

async function createUser(payload) {
  return requestJson("/users", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function updateUser(userId, payload) {
  return requestJson(`/users/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function deleteUser(userId) {
  return requestJson(`/users/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
}

function renderHero(item) {
  if (!item) {
    elements.heroStory.innerHTML = `
      <div class="empty-state empty-state-hero">
        <p class="slot-label">Главная история</p>
        <h2 class="hero-headline">Рекомендации пока не пришли</h2>
        <p class="hero-summary">Попробуйте другой user id или ослабьте фильтры запроса.</p>
      </div>
    `;
    return;
  }

  elements.heroStory.innerHTML = `
    <div class="story-meta">
      <span class="story-tag">Главная история</span>
      <span>${escapeHtml(formatCategory(item))}</span>
      <span>${escapeHtml(item.banner_id)}</span>
    </div>
    <h2 class="hero-headline">${escapeHtml(storyHeadline(item, 0))}</h2>
    <p class="hero-summary">${escapeHtml(storySummary(item))}</p>
    <div class="hero-actions">
      <span class="metric-chip">CTR ${escapeHtml(formatPercent(item.pred_ctr))}</span>
      <span class="metric-chip">Score ${escapeHtml(formatScore(item.final_score))}</span>
      <span class="metric-chip">Brand ${escapeHtml(item.brand)}</span>
    </div>
    <div class="hero-actions">
      <a class="hero-link" href="${escapeHtml(buildUrl(item))}" target="_blank" rel="noreferrer">
        Открыть лендинг
      </a>
    </div>
  `;
}

function renderBriefing(items) {
  const content = items
    .slice(1, 4)
    .map(
      (item, index) => `
        <article class="briefing-item">
          <div class="story-meta">
            <span>${escapeHtml(formatCategory(item))}</span>
            <span>${escapeHtml(item.banner_format)}</span>
          </div>
          <h3>${escapeHtml(storyHeadline(item, index + 1))}</h3>
          <p>${escapeHtml(storySummary(item))}</p>
        </article>
      `,
    )
    .join("");

  elements.newsBriefing.innerHTML = `
    <div class="section-title compact">
      <div>
        <p class="kicker">Briefing</p>
        <h2>Что еще в выдаче</h2>
      </div>
    </div>
    <div class="briefing-list">
      ${content || '<div class="empty-state"><p>Дополнительных рекомендаций пока нет.</p></div>'}
    </div>
  `;
}

function renderFeed(items) {
  const content = items
    .slice(1)
    .map(
      (item, index) => `
        <article class="story-card reveal">
          <div class="story-meta">
            <span class="story-tag">${escapeHtml(formatCategory(item))}</span>
            <span>${escapeHtml(item.banner_format)}</span>
            <span>${escapeHtml(item.campaign_goal)}</span>
          </div>
          <h3 class="story-headline">${escapeHtml(storyHeadline(item, index + 1))}</h3>
          <p class="story-summary">${escapeHtml(storySummary(item))}</p>
          <div class="inline-meta">
            <span>CTR ${escapeHtml(formatPercent(item.pred_ctr))}</span>
            <span>Score ${escapeHtml(formatScore(item.final_score))}</span>
            <span>CPM ${escapeHtml(formatMoney(item.cpm_bid))}</span>
          </div>
          <div class="story-footer">
            <span class="inline-meta">
              <span>brand ${escapeHtml(item.brand)}</span>
              <span>seen ${escapeHtml(String(item.served_impressions_total || 0))}</span>
            </span>
            <a class="story-link" href="${escapeHtml(buildUrl(item))}" target="_blank" rel="noreferrer">
              Перейти
            </a>
          </div>
        </article>
      `,
    )
    .join("");

  elements.feedGrid.innerHTML = content || `
    <article class="story-card story-card-empty">
      <p class="slot-label">Feed Empty</p>
      <h3 class="story-headline">Лента пока пустая</h3>
      <p class="story-summary">Сервер не вернул карточки для текущего профиля. Попробуйте другой user id или измените фильтры.</p>
    </article>
  `;
}

function renderNativeBand(items) {
  const content = items
    .slice(0, 3)
    .map(
      (item) => `
        <article class="native-card">
          <p class="slot-label">Native Sponsored Story</p>
          <h3>${escapeHtml(item.brand)} в блоке ${escapeHtml(formatCategory(item))}</h3>
          <p>${escapeHtml(storySummary(item))}</p>
        </article>
      `,
    )
    .join("");

  elements.nativeBand.innerHTML =
    content || '<article class="native-card"><p>Нативные интеграции появятся здесь, когда в ответе будут рекомендации.</p></article>';
}

function renderSponsoredRail(items) {
  const content = items
    .slice(0, 3)
    .map(
      (item) => `
        <article class="sponsored-card">
          <p class="slot-label">${escapeHtml(item.banner_id)}</p>
          <h3>${escapeHtml(item.brand)}</h3>
          <p class="sponsored-copy">${escapeHtml(storySummary(item))}</p>
        </article>
      `,
    )
    .join("");

  elements.sponsoredRail.innerHTML =
    content || '<article class="sponsored-card"><p class="sponsored-copy">Партнерские блоки пока пусты.</p></article>';
}

function renderProfile(response, requestPayload) {
  const items = response.items || [];
  const topCategory = items[0] ? formatCategory(items[0]) : "Нет данных";
  const averageCtr = items.length
    ? items.reduce((sum, item) => sum + Number(item.pred_ctr || 0), 0) / items.length
    : 0;
  const averageQuality = items.length
    ? items.reduce((sum, item) => sum + Number(item.quality_score || 0), 0) / items.length
    : 0;

  elements.profileCard.innerHTML = `
    <div class="profile-stack">
      <div class="profile-item">
        <span class="profile-label">User</span>
        <strong>${escapeHtml(requestPayload.user_id)}</strong>
      </div>
      <div class="profile-item">
        <span class="profile-label">Доминирующая рубрика</span>
        <strong>${escapeHtml(topCategory)}</strong>
      </div>
      <div class="profile-item">
        <span class="profile-label">Средний CTR</span>
        <strong>${escapeHtml(formatPercent(averageCtr))}</strong>
      </div>
      <div class="profile-item">
        <span class="profile-label">Средний quality</span>
        <strong>${escapeHtml(formatScore(averageQuality))}</strong>
      </div>
      <div class="profile-item">
        <span class="profile-label">Выдача</span>
        <strong>${escapeHtml(response.retrieval_used ? "retrieval + ranking" : "all banners")}</strong>
      </div>
    </div>
  `;
}

function renderMeta(response) {
  const items = [
    {
      label: "Источник",
      value: state.source === "api" ? "backend api" : state.source === "local" ? "local fallback" : "demo mode",
    },
    {
      label: "Модель",
      value: response.model_type,
    },
    {
      label: "Artifacts",
      value: response.artifacts_dir || "n/a",
    },
    {
      label: "Top K",
      value: String(response.top_k),
    },
    {
      label: "Retrieval",
      value: response.retrieval_used ? "enabled" : "disabled",
    },
    {
      label: "Cards rendered",
      value: String(response.items.length),
    },
    {
      label: "As of date",
      value: response.as_of_date || "n/a",
    },
    {
      label: "Режим",
      value:
        state.view === "admin"
          ? "banner admin"
          : state.view === "users"
            ? "user admin"
            : "showcase",
    },
  ];

  elements.metaGrid.innerHTML = items
    .map(
      (item) => `
        <article class="meta-card">
          <p class="slot-label">${escapeHtml(item.label)}</p>
          <h3>${escapeHtml(item.value)}</h3>
          <p class="meta-value">Служебный блок для контроля состояния страницы и текущего источника данных.</p>
        </article>
      `,
    )
    .join("");
}

function renderPlacements() {
  elements.placementGrid.innerHTML = placementBlueprint
    .map(
      (item) => `
        <article class="placement-card">
          <p class="slot-label">${escapeHtml(item.size)}</p>
          <h3>${escapeHtml(item.title)}</h3>
          <p class="placement-copy">${escapeHtml(item.copy)}</p>
        </article>
      `,
    )
    .join("");
}

function renderMidroll(items) {
  const midrollItem = items[3] || items[0];
  if (!midrollItem) {
    elements.midrollSlot.innerHTML = `
      <div>
        <p class="slot-label">Ad Slot 728x90</p>
        <h3>Встраиваемый midroll после карточек</h3>
        <p>Слот свободен. Здесь можно отдавать фиксированный баннер или нативную вставку.</p>
      </div>
      <div class="slot-code">awaiting inventory</div>
    `;
    return;
  }

  elements.midrollSlot.innerHTML = `
    <div>
      <p class="slot-label">Ad Slot 728x90</p>
      <h3>Встраиваемый midroll после карточек</h3>
      <p>Хорошая точка для бренда ${escapeHtml(midrollItem.brand)} или для теста статического баннера.</p>
    </div>
    <div class="slot-code">best candidate: ${escapeHtml(midrollItem.banner_id)}</div>
  `;
}

function renderShowcaseOverview(response) {
  const items = response.items || [];
  const averageCtr = items.length
    ? items.reduce((sum, item) => sum + Number(item.pred_ctr || 0), 0) / items.length
    : 0;
  const activeBanners = state.banners.filter((banner) => banner.is_active).length;
  const topItem = items[0];

  elements.recommendationsCount.textContent = `${items.length} карточек`;
  elements.recommendationsSource.textContent =
    state.source === "api"
      ? "Данные пришли из backend recommendations API"
      : state.source === "local"
        ? "API недоступен, поэтому показан локальный fallback"
        : "Используется встроенный demo-набор";
  elements.topBrand.textContent = topItem ? topItem.brand : "-";
  elements.topBrandCopy.textContent = topItem
    ? `${formatCategory(topItem)} / ${humanize(topItem.banner_format)}`
    : "Нет данных для первой позиции";
  elements.averageCtr.textContent = formatPercent(averageCtr);
  elements.inventoryTotal.textContent = String(state.banners.length);
  elements.inventoryCopy.textContent = `${activeBanners} активных из ${state.banners.length}`;
  elements.leaderboardCode.textContent = topItem ? `feature: ${topItem.banner_id}` : "inventory syncing...";
  elements.railCode.textContent =
    state.banners[0] ? `${state.banners[0].banner_format} / ${state.banners[0].brand}` : "awaiting inventory";
}

function renderShowcase(response, requestPayload) {
  const items = response.items || [];
  renderHero(items[0]);
  renderBriefing(items);
  renderFeed(items);
  renderNativeBand(items);
  renderSponsoredRail(items);
  renderProfile(response, requestPayload);
  renderMeta(response);
  renderPlacements();
  renderMidroll(items);
  renderShowcaseOverview(response);
}

function getFilteredBanners() {
  const query = state.bannerSearch.trim().toLowerCase();
  return state.banners.filter((banner) => {
    const matchesStatus =
      state.bannerStatusFilter === "all"
        ? true
        : state.bannerStatusFilter === "active"
          ? banner.is_active
          : !banner.is_active;
    if (!matchesStatus) {
      return false;
    }
    if (!query) {
      return true;
    }

    const haystack = [
      banner.banner_id,
      banner.brand,
      banner.category,
      banner.subcategory,
      banner.banner_format,
      banner.campaign_goal,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });
}

function renderAdminStats() {
  const activeCount = state.banners.filter((banner) => banner.is_active).length;
  const averageCpm = state.banners.length
    ? state.banners.reduce((sum, banner) => sum + Number(banner.cpm_bid || 0), 0) / state.banners.length
    : 0;
  const selectedBanner = state.banners.find((banner) => banner.banner_id === state.selectedBannerId);

  elements.adminTotalBanners.textContent = String(state.banners.length);
  elements.adminActiveBanners.textContent = String(activeCount);
  elements.adminAverageCpm.textContent = formatMoney(averageCpm);
  elements.adminSelectedBanner.textContent = selectedBanner ? selectedBanner.banner_id : "Новый";
}

function renderBannersTable() {
  const banners = getFilteredBanners();
  elements.bannersListCaption.textContent =
    state.bannersSource === "api"
      ? `Показано ${banners.length} из ${state.banners.length} баннеров из backend`
      : `Показано ${banners.length} demo-баннеров во встроенном режиме`;

  elements.bannersTableEmpty.hidden = banners.length > 0;
  elements.bannersTableBody.innerHTML = banners
    .map((banner) => {
      const isSelected = banner.banner_id === state.selectedBannerId;
      return `
        <tr class="${isSelected ? "is-selected" : ""}" data-banner-id="${escapeHtml(banner.banner_id)}">
          <td><button class="table-link" type="button" data-select-banner="${escapeHtml(banner.banner_id)}">${escapeHtml(banner.banner_id)}</button></td>
          <td>${escapeHtml(banner.brand)}</td>
          <td>${escapeHtml(banner.category)} / ${escapeHtml(banner.subcategory)}</td>
          <td>${escapeHtml(banner.banner_format)}</td>
          <td>${escapeHtml(banner.campaign_goal)}</td>
          <td>${escapeHtml(formatMoney(banner.cpm_bid))}</td>
          <td><span class="table-pill ${banner.is_active ? "is-active" : "is-inactive"}">${banner.is_active ? "active" : "inactive"}</span></td>
        </tr>
      `;
    })
    .join("");
}

function getFilteredUsers() {
  const query = state.userSearch.trim().toLowerCase();
  return state.users.filter((user) => {
    const matchesPremium =
      state.userPremiumFilter === "all"
        ? true
        : state.userPremiumFilter === "premium"
          ? user.is_premium
          : !user.is_premium;
    if (!matchesPremium) {
      return false;
    }
    if (!query) {
      return true;
    }

    const haystack = [
      user.user_id,
      user.country,
      user.platform,
      user.device_os,
      user.interest_1,
      user.interest_2,
      user.interest_3,
      user.income_band,
      user.activity_segment,
    ]
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });
}

function fillBannerForm(banner) {
  elements.bannerId.value = banner.banner_id;
  elements.bannerBrand.value = banner.brand;
  elements.bannerCategory.value = banner.category;
  elements.bannerSubcategory.value = banner.subcategory;
  elements.bannerFormat.value = banner.banner_format;
  elements.bannerGoal.value = banner.campaign_goal;
  elements.bannerGender.value = banner.target_gender;
  elements.bannerAgeMin.value = String(banner.target_age_min);
  elements.bannerAgeMax.value = String(banner.target_age_max);
  elements.bannerCpm.value = formatMoney(banner.cpm_bid);
  elements.bannerQuality.value = formatScore(banner.quality_score);
  elements.bannerCreatedAt.value = banner.created_at;
  elements.bannerLandingPage.value = banner.landing_page;
  elements.bannerIsActive.checked = Boolean(banner.is_active);
}

function renderUsersStats() {
  const premiumCount = state.users.filter((user) => user.is_premium).length;
  const averageAge = state.users.length
    ? state.users.reduce((sum, user) => sum + Number(user.age || 0), 0) / state.users.length
    : 0;
  const selectedUser = state.users.find((user) => user.user_id === state.selectedUserId);

  elements.adminTotalUsers.textContent = String(state.users.length);
  elements.adminPremiumUsers.textContent = String(premiumCount);
  elements.adminAverageAge.textContent = formatWholeNumber(averageAge);
  elements.adminSelectedUser.textContent = selectedUser ? selectedUser.user_id : "Новый";
}

function renderUsersTable() {
  const users = getFilteredUsers();
  elements.usersListCaption.textContent =
    state.usersSource === "api"
      ? `Показано ${users.length} из ${state.users.length} пользователей из backend`
      : `Показано ${users.length} demo-пользователей во встроенном режиме`;

  elements.usersTableEmpty.hidden = users.length > 0;
  elements.usersTableBody.innerHTML = users
    .map((user) => {
      const isSelected = user.user_id === state.selectedUserId;
      return `
        <tr class="${isSelected ? "is-selected" : ""}" data-user-id="${escapeHtml(user.user_id)}">
          <td><button class="table-link" type="button" data-select-user="${escapeHtml(user.user_id)}">${escapeHtml(user.user_id)}</button></td>
          <td>${escapeHtml(String(user.age))}</td>
          <td>${escapeHtml(user.gender)}</td>
          <td>${escapeHtml(user.platform)}</td>
          <td>${escapeHtml(user.country)}</td>
          <td>${escapeHtml(user.income_band)}</td>
          <td><span class="table-pill ${user.is_premium ? "is-premium" : "is-standard"}">${user.is_premium ? "premium" : "standard"}</span></td>
        </tr>
      `;
    })
    .join("");
}

function fillUserForm(user) {
  elements.manageUserId.value = user.user_id;
  elements.manageUserAge.value = String(user.age);
  elements.manageUserGender.value = user.gender;
  elements.manageUserCityTier.value = user.city_tier;
  elements.manageUserDeviceOs.value = user.device_os;
  elements.manageUserPlatform.value = user.platform;
  elements.manageUserIncomeBand.value = user.income_band;
  elements.manageUserActivitySegment.value = user.activity_segment;
  elements.manageUserInterest1.value = user.interest_1;
  elements.manageUserInterest2.value = user.interest_2;
  elements.manageUserInterest3.value = user.interest_3;
  elements.manageUserCountry.value = user.country;
  elements.manageUserSignupDaysAgo.value = String(user.signup_days_ago);
  elements.manageUserIsPremium.checked = Boolean(user.is_premium);
}

function resetBannerForm() {
  state.selectedBannerId = null;
  state.bannerFormMode = "create";
  elements.bannerForm.reset();
  elements.bannerId.value = "";
  elements.bannerCreatedAt.value = new Date().toISOString().slice(0, 10);
  elements.bannerGender.value = "U";
  elements.bannerFormat.value = "static";
  elements.bannerGoal.value = "awareness";
  elements.bannerIsActive.checked = true;
  elements.bannerId.readOnly = false;
  elements.bannerFormTitle.textContent = "Создать новый баннер";
  elements.bannerFormCaption.textContent = "Заполните поля и сохраните запись в backend.";
  elements.deleteBannerButton.disabled = true;
  renderAdminStats();
  renderBannersTable();
}

function resetUserForm() {
  state.selectedUserId = null;
  state.userFormMode = "create";
  elements.userForm.reset();
  elements.manageUserId.value = "";
  elements.manageUserAge.value = "25";
  elements.manageUserGender.value = "U";
  elements.manageUserCityTier.value = "tier_1";
  elements.manageUserDeviceOs.value = "ios";
  elements.manageUserPlatform.value = "mobile_app";
  elements.manageUserIncomeBand.value = "mid";
  elements.manageUserActivitySegment.value = "steady";
  elements.manageUserInterest1.value = "travel";
  elements.manageUserInterest2.value = "electronics";
  elements.manageUserInterest3.value = "food";
  elements.manageUserCountry.value = "RU";
  elements.manageUserSignupDaysAgo.value = "30";
  elements.manageUserIsPremium.checked = false;
  elements.manageUserId.readOnly = false;
  elements.userFormTitle.textContent = "Создать нового пользователя";
  elements.userFormCaption.textContent = "Заполните поля и сохраните запись в backend.";
  elements.deleteUserButton.disabled = true;
  renderUsersStats();
  renderUsersTable();
}

function selectBanner(bannerId) {
  const banner = state.banners.find((item) => item.banner_id === bannerId);
  if (!banner) {
    resetBannerForm();
    return;
  }

  state.selectedBannerId = banner.banner_id;
  state.bannerFormMode = "edit";
  fillBannerForm(banner);
  elements.bannerId.readOnly = true;
  elements.bannerFormTitle.textContent = `Редактирование ${banner.banner_id}`;
  elements.bannerFormCaption.textContent = "Изменения сохраняются через PATCH /api/v1/banners/{banner_id}.";
  elements.deleteBannerButton.disabled = false;
  renderAdminStats();
  renderBannersTable();
}

function selectUser(userId) {
  const user = state.users.find((item) => item.user_id === userId);
  if (!user) {
    resetUserForm();
    return;
  }

  state.selectedUserId = user.user_id;
  state.userFormMode = "edit";
  fillUserForm(user);
  elements.manageUserId.readOnly = true;
  elements.userFormTitle.textContent = `Редактирование ${user.user_id}`;
  elements.userFormCaption.textContent = "Изменения сохраняются через PATCH /api/v1/users/{user_id}.";
  elements.deleteUserButton.disabled = false;
  renderUsersStats();
  renderUsersTable();
}

function buildBannerPayloadFromForm() {
  const ageMin = Number(elements.bannerAgeMin.value);
  const ageMax = Number(elements.bannerAgeMax.value);

  if (ageMax < ageMin) {
    throw new Error("target_age_max должен быть больше или равен target_age_min");
  }

  return {
    banner_id: elements.bannerId.value.trim(),
    brand: elements.bannerBrand.value.trim(),
    category: elements.bannerCategory.value.trim(),
    subcategory: elements.bannerSubcategory.value.trim(),
    banner_format: elements.bannerFormat.value,
    campaign_goal: elements.bannerGoal.value,
    target_gender: elements.bannerGender.value,
    target_age_min: ageMin,
    target_age_max: ageMax,
    cpm_bid: Number(elements.bannerCpm.value),
    quality_score: Number(elements.bannerQuality.value),
    created_at: elements.bannerCreatedAt.value,
    is_active: elements.bannerIsActive.checked,
    landing_page: elements.bannerLandingPage.value.trim(),
  };
}

function buildBannerPatchPayload(payload) {
  return {
    brand: payload.brand,
    category: payload.category,
    subcategory: payload.subcategory,
    banner_format: payload.banner_format,
    campaign_goal: payload.campaign_goal,
    target_gender: payload.target_gender,
    target_age_min: payload.target_age_min,
    target_age_max: payload.target_age_max,
    cpm_bid: payload.cpm_bid,
    quality_score: payload.quality_score,
    created_at: payload.created_at,
    is_active: payload.is_active,
    landing_page: payload.landing_page,
  };
}

function buildUserPayloadFromForm() {
  return {
    user_id: elements.manageUserId.value.trim(),
    age: Number(elements.manageUserAge.value),
    gender: elements.manageUserGender.value,
    city_tier: elements.manageUserCityTier.value.trim(),
    device_os: elements.manageUserDeviceOs.value.trim(),
    platform: elements.manageUserPlatform.value.trim(),
    income_band: elements.manageUserIncomeBand.value.trim(),
    activity_segment: elements.manageUserActivitySegment.value.trim(),
    interest_1: elements.manageUserInterest1.value.trim(),
    interest_2: elements.manageUserInterest2.value.trim(),
    interest_3: elements.manageUserInterest3.value.trim(),
    country: elements.manageUserCountry.value.trim().toUpperCase(),
    signup_days_ago: Number(elements.manageUserSignupDaysAgo.value),
    is_premium: elements.manageUserIsPremium.checked,
  };
}

function buildUserPatchPayload(payload) {
  return {
    age: payload.age,
    gender: payload.gender,
    city_tier: payload.city_tier,
    device_os: payload.device_os,
    platform: payload.platform,
    income_band: payload.income_band,
    activity_segment: payload.activity_segment,
    interest_1: payload.interest_1,
    interest_2: payload.interest_2,
    interest_3: payload.interest_3,
    country: payload.country,
    signup_days_ago: payload.signup_days_ago,
    is_premium: payload.is_premium,
  };
}

function syncBannerIntoState(savedBanner) {
  const normalized = {
    ...savedBanner,
    cpm_bid: Number(savedBanner.cpm_bid),
    quality_score: Number(savedBanner.quality_score),
  };
  const index = state.banners.findIndex((item) => item.banner_id === normalized.banner_id);
  if (index === -1) {
    state.banners = [normalized, ...state.banners];
  } else {
    state.banners[index] = normalized;
  }
}

function syncUserIntoState(savedUser) {
  const index = state.users.findIndex((item) => item.user_id === savedUser.user_id);
  if (index === -1) {
    state.users = [savedUser, ...state.users];
  } else {
    state.users[index] = savedUser;
  }
}

async function loadRecommendations() {
  const payload = buildRequestPayload();
  elements.refreshButton.disabled = true;
  elements.demoButton.disabled = true;
  setStatus("Loading recommendations", "");

  try {
    state.response = await fetchRecommendations(payload);
    state.source = "api";
    setStatus("API ready", "ok");
  } catch (error) {
    state.response = buildLocalResponse(payload);
    state.source = "local";
    setStatus("Local fallback", "demo");
    console.warn("Failed to fetch recommendations, using local fallback:", error);
    showToast(`Recommendations fallback: ${error.message}`, "warning");
  } finally {
    renderShowcase(state.response, payload);
    elements.refreshButton.disabled = false;
    elements.demoButton.disabled = false;
  }
}

async function loadBanners({ preserveSelection = true } = {}) {
  elements.reloadBannersButton.disabled = true;
  elements.newBannerButton.disabled = true;
  setStatus("Syncing banners", "");

  const previousSelection = preserveSelection ? state.selectedBannerId : null;

  try {
    const banners = await fetchBanners();
    state.banners = banners.map((banner) => ({
      ...banner,
      cpm_bid: Number(banner.cpm_bid),
      quality_score: Number(banner.quality_score),
    }));
    state.bannersSource = "api";
    setStatus("Inventory synced", "ok");
  } catch (error) {
    state.banners = [...DEMO_BANNERS];
    state.bannersSource = "demo";
    setStatus("Demo inventory", "demo");
    console.warn("Failed to fetch banners, using demo inventory:", error);
    showToast(`Banners fallback: ${error.message}`, "warning");
  } finally {
    renderAdminStats();
    renderBannersTable();
    renderShowcaseOverview(state.response);
    if (previousSelection && state.banners.some((banner) => banner.banner_id === previousSelection)) {
      selectBanner(previousSelection);
    } else {
      resetBannerForm();
    }
    elements.reloadBannersButton.disabled = false;
    elements.newBannerButton.disabled = false;
  }
}

async function loadUsers({ preserveSelection = true } = {}) {
  elements.reloadUsersButton.disabled = true;
  elements.newUserButton.disabled = true;
  setStatus("Syncing users", "");

  const previousSelection = preserveSelection ? state.selectedUserId : null;

  try {
    state.users = await fetchUsers();
    state.usersSource = "api";
    setStatus("Users synced", "ok");
  } catch (error) {
    state.users = [...DEMO_USERS];
    state.usersSource = "demo";
    setStatus("Demo users", "demo");
    console.warn("Failed to fetch users, using demo users:", error);
    showToast(`Users fallback: ${error.message}`, "warning");
  } finally {
    renderUsersStats();
    renderUsersTable();
    if (previousSelection && state.users.some((user) => user.user_id === previousSelection)) {
      selectUser(previousSelection);
    } else {
      resetUserForm();
    }
    elements.reloadUsersButton.disabled = false;
    elements.newUserButton.disabled = false;
  }
}

function switchView(view) {
  state.view = view;
  const isShowcase = view === "showcase";
  const isBannerAdmin = view === "admin";
  const isUsersAdmin = view === "users";
  elements.showcaseView.hidden = !isShowcase;
  elements.adminView.hidden = !isBannerAdmin;
  elements.usersView.hidden = !isUsersAdmin;
  elements.showcaseView.classList.toggle("is-active", isShowcase);
  elements.adminView.classList.toggle("is-active", isBannerAdmin);
  elements.usersView.classList.toggle("is-active", isUsersAdmin);
  elements.showcaseTab.classList.toggle("is-active", isShowcase);
  elements.adminTab.classList.toggle("is-active", isBannerAdmin);
  elements.usersTab.classList.toggle("is-active", isUsersAdmin);
  elements.showcaseTab.setAttribute("aria-selected", String(isShowcase));
  elements.adminTab.setAttribute("aria-selected", String(isBannerAdmin));
  elements.usersTab.setAttribute("aria-selected", String(isUsersAdmin));
  renderMeta(state.response);
}

async function handleBannerSubmit(event) {
  event.preventDefault();
  const payload = buildBannerPayloadFromForm();
  elements.saveBannerButton.disabled = true;

  try {
    if (state.bannerFormMode === "create") {
      const saved = await createBanner(payload);
      syncBannerIntoState(saved);
      state.bannersSource = "api";
      selectBanner(saved.banner_id);
      showToast(`Баннер ${saved.banner_id} создан`, "success");
    } else {
      const saved = await updateBanner(payload.banner_id, buildBannerPatchPayload(payload));
      syncBannerIntoState(saved);
      state.bannersSource = "api";
      selectBanner(saved.banner_id);
      showToast(`Баннер ${saved.banner_id} обновлен`, "success");
    }
  } catch (error) {
    showToast(`Не удалось сохранить: ${error.message}`, "error");
  } finally {
    renderAdminStats();
    renderBannersTable();
    renderShowcaseOverview(state.response);
    elements.saveBannerButton.disabled = false;
  }
}

async function handleDeleteBanner() {
  if (!state.selectedBannerId) {
    return;
  }

  const bannerId = state.selectedBannerId;
  const confirmed = window.confirm(`Удалить баннер ${bannerId}?`);
  if (!confirmed) {
    return;
  }

  elements.deleteBannerButton.disabled = true;
  try {
    await deleteBanner(bannerId);
    state.banners = state.banners.filter((banner) => banner.banner_id !== bannerId);
    resetBannerForm();
    showToast(`Баннер ${bannerId} удален`, "success");
  } catch (error) {
    showToast(`Не удалось удалить: ${error.message}`, "error");
  } finally {
    renderAdminStats();
    renderBannersTable();
    renderShowcaseOverview(state.response);
    elements.deleteBannerButton.disabled = state.bannerFormMode !== "edit";
  }
}

async function handleUserSubmit(event) {
  event.preventDefault();
  const payload = buildUserPayloadFromForm();
  elements.saveUserButton.disabled = true;

  try {
    if (state.userFormMode === "create") {
      const saved = await createUser(payload);
      syncUserIntoState(saved);
      state.usersSource = "api";
      selectUser(saved.user_id);
      showToast(`Пользователь ${saved.user_id} создан`, "success");
    } else {
      const saved = await updateUser(payload.user_id, buildUserPatchPayload(payload));
      syncUserIntoState(saved);
      state.usersSource = "api";
      selectUser(saved.user_id);
      showToast(`Пользователь ${saved.user_id} обновлен`, "success");
    }
  } catch (error) {
    showToast(`Не удалось сохранить пользователя: ${error.message}`, "error");
  } finally {
    renderUsersStats();
    renderUsersTable();
    elements.saveUserButton.disabled = false;
  }
}

async function handleDeleteUser() {
  if (!state.selectedUserId) {
    return;
  }

  const userId = state.selectedUserId;
  const confirmed = window.confirm(`Удалить пользователя ${userId}?`);
  if (!confirmed) {
    return;
  }

  elements.deleteUserButton.disabled = true;
  try {
    await deleteUser(userId);
    state.users = state.users.filter((user) => user.user_id !== userId);
    resetUserForm();
    showToast(`Пользователь ${userId} удален`, "success");
  } catch (error) {
    showToast(`Не удалось удалить пользователя: ${error.message}`, "error");
  } finally {
    renderUsersStats();
    renderUsersTable();
    elements.deleteUserButton.disabled = state.userFormMode !== "edit";
  }
}

function activateQuickUsers() {
  document.querySelectorAll("[data-user-id]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.userId.value = button.dataset.userId || "";
      loadRecommendations();
    });
  });
}

function bindEvents() {
  elements.controlsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadRecommendations();
  });

  elements.refreshButton.addEventListener("click", () => {
    loadRecommendations();
  });

  elements.demoButton.addEventListener("click", () => {
    state.response = buildLocalResponse(buildRequestPayload());
    state.source = "demo";
    setStatus("Demo mode", "demo");
    renderShowcase(state.response, buildRequestPayload());
    showToast("Витрина переключена в demo режим", "info");
  });

  elements.showcaseTab.addEventListener("click", () => {
    switchView("showcase");
  });

  elements.adminTab.addEventListener("click", () => {
    switchView("admin");
  });

  elements.usersTab.addEventListener("click", () => {
    switchView("users");
  });

  elements.bannerSearch.addEventListener("input", (event) => {
    state.bannerSearch = event.target.value;
    renderBannersTable();
  });

  elements.bannerStatusFilter.addEventListener("change", (event) => {
    state.bannerStatusFilter = event.target.value;
    renderBannersTable();
  });

  elements.reloadBannersButton.addEventListener("click", () => {
    loadBanners();
  });

  elements.newBannerButton.addEventListener("click", () => {
    resetBannerForm();
    showToast("Форма готова для создания нового баннера", "info");
  });

  elements.bannerForm.addEventListener("submit", handleBannerSubmit);

  elements.resetBannerButton.addEventListener("click", () => {
    resetBannerForm();
  });

  elements.deleteBannerButton.addEventListener("click", handleDeleteBanner);

  elements.bannersTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-banner]");
    if (!button) {
      return;
    }
    selectBanner(button.dataset.selectBanner);
  });

  elements.userSearch.addEventListener("input", (event) => {
    state.userSearch = event.target.value;
    renderUsersTable();
  });

  elements.userPremiumFilter.addEventListener("change", (event) => {
    state.userPremiumFilter = event.target.value;
    renderUsersTable();
  });

  elements.reloadUsersButton.addEventListener("click", () => {
    loadUsers();
  });

  elements.newUserButton.addEventListener("click", () => {
    resetUserForm();
    showToast("Форма готова для создания нового пользователя", "info");
  });

  elements.userForm.addEventListener("submit", handleUserSubmit);

  elements.resetUserButton.addEventListener("click", () => {
    resetUserForm();
  });

  elements.deleteUserButton.addEventListener("click", handleDeleteUser);

  elements.usersTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-user]");
    if (!button) {
      return;
    }
    selectUser(button.dataset.selectUser);
  });

  activateQuickUsers();
}

function initialize() {
  bindEvents();
  renderPlacements();
  resetBannerForm();
  resetUserForm();
  renderShowcase(state.response, buildRequestPayload());
  renderAdminStats();
  renderBannersTable();
  renderUsersStats();
  renderUsersTable();
  loadRecommendations();
  loadBanners({ preserveSelection: false });
  loadUsers({ preserveSelection: false });
}

initialize();
