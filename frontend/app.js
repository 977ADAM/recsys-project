const DEMO_RESPONSE = {
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
      is_active: 1,
      landing_page: "https://example.com/travel/car-rental/skyjet",
    },
    {
      banner_id: "bnr_demo_002",
      brand: "PixelOne",
      category: "electronics",
      subcategory: "smartphones",
      banner_format: "banner",
      campaign_goal: "conversion",
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
      is_active: 1,
      landing_page: "https://example.com/electronics/smartphones/pixelone",
    },
    {
      banner_id: "bnr_demo_003",
      brand: "CodeSpring",
      category: "education",
      subcategory: "coding",
      banner_format: "native",
      campaign_goal: "traffic",
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
      is_active: 1,
      landing_page: "https://example.com/education/coding/codespring",
    },
    {
      banner_id: "bnr_demo_004",
      brand: "CityBite",
      category: "food",
      subcategory: "snacks",
      banner_format: "banner",
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
      is_active: 1,
      landing_page: "https://example.com/food/snacks/citybite",
    },
  ],
};

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
    copy: "Большой верхний баннер под шапкой. Хорошо подходит для запуска промо-дня или брендового takeover.",
  },
  {
    size: "300x600",
    title: "Right Rail Sticky",
    copy: "Фиксированный баннер в правой колонке. Работает для длительных кампаний и повторных касаний.",
  },
  {
    size: "728x90",
    title: "In-Feed Midroll",
    copy: "Встраивается между карточками новостей после 3-4 материалов и не ломает ритм ленты.",
  },
  {
    size: "Native Card",
    title: "Sponsored Story",
    copy: "Нативная интеграция в стиле редакционного материала с брендом, CTA и качественным score.",
  },
];

const state = {
  response: DEMO_RESPONSE,
  source: "demo",
  requestInFlight: false,
};

const elements = {
  apiStatus: document.querySelector("#api-status"),
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
  refreshButton: document.querySelector("#refresh-button"),
  demoButton: document.querySelector("#demo-button"),
};

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function formatCategory(item) {
  return CATEGORY_TITLES[item.category] || humanize(item.category);
}

function humanize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function storyHeadline(item, index) {
  const descriptors = [
    "переосмысляет спрос в своей категории",
    "выходит в лидеры персональной ленты",
    "забирает внимание аудитории прямо сейчас",
    "попадает в зону высокого интереса",
  ];
  const descriptor = descriptors[index % descriptors.length];
  return `${item.brand} ${descriptor}`;
}

function storySummary(item) {
  const subcategory = SUBCATEGORY_TITLES[item.subcategory] || humanize(item.subcategory);
  return `${formatCategory(item)} / ${subcategory}. Формат ${item.banner_format}, цель кампании ${item.campaign_goal}. Прогнозный CTR ${formatPercent(item.pred_ctr)} и quality score ${item.quality_score.toFixed(3)}.`;
}

function buildUrl(item) {
  return item.landing_page || `https://example.com/${item.category}/${item.subcategory}/${item.brand.toLowerCase()}`;
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
  const activeItems = payload.only_active ? sortedItems.filter((item) => item.is_active === 1) : sortedItems;
  const limit = Math.max(1, Number(payload.top_k) || DEMO_RESPONSE.top_k);
  const items = activeItems.slice(0, limit);

  return {
    model_type: "local-demo-feed",
    artifacts_dir: "frontend-local",
    retrieval_used: payload.candidate_mode === "retrieval + ranking",
    online_state_applied: false,
    top_k: items.length,
    items,
  };
}

async function fetchRecommendations(payload) {
  const response = await fetch("/api/v1/recommendations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorPayload = await response.json();
      if (errorPayload?.detail) {
        detail = errorPayload.detail;
      }
    } catch {
      // Leave the generic message when the server doesn't return JSON.
    }
    throw new Error(detail);
  }

  return response.json();
}

function setStatus(label, className) {
  elements.apiStatus.textContent = label;
  elements.apiStatus.className = `status-chip ${className}`.trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderHero(item) {
  if (!item) {
    elements.heroStory.innerHTML = `
      <div class="empty-state empty-state-hero">
        <p class="slot-label">Главная история</p>
        <h2 class="hero-headline">Рекомендации пока не пришли</h2>
        <p class="hero-summary">Измените параметры запроса или проверьте, что backend с данными доступен.</p>
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
      <span class="metric-chip">Value ${escapeHtml(formatScore(item.final_score))}</span>
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
            <span>CPM ${escapeHtml(item.cpm_bid.toFixed(2))}</span>
          </div>
          <div class="story-footer">
            <span class="inline-meta">
              <span>brand ${escapeHtml(item.brand)}</span>
              <span>seen ${escapeHtml(String(item.served_impressions_total))}</span>
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
      <p class="story-summary">Сервер не вернул карточки для текущего профиля. Попробуйте другой user id или ослабьте фильтры.</p>
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
    content || '<article class="sponsored-card"><p class="sponsored-copy">Партнерские блоки пока пусты. Можно загрузить данные из API или использовать локальный demo-режим.</p></article>';
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
        <strong>${escapeHtml(averageQuality.toFixed(3))}</strong>
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
      value:
        state.source === "api"
          ? "backend api"
          : state.source === "local"
            ? "local fallback"
            : "demo showcase",
    },
    {
      label: "Модель",
      value: response.model_type,
    },
    {
      label: "Artifacts",
      value: response.artifacts_dir,
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
      label: "Online state",
      value: response.online_state_applied ? "applied" : "not applied",
    },
    {
      label: "Cards rendered",
      value: String(response.items.length),
    },
    {
      label: "Режим",
      value: "без inference_service",
    },
  ];

  elements.metaGrid.innerHTML = items
    .map(
      (item) => `
        <article class="meta-card">
          <p class="slot-label">${escapeHtml(item.label)}</p>
          <h3>${escapeHtml(item.value)}</h3>
          <p class="meta-value">Служебный блок для контроля запроса, состояния модели и отладки связки frontend/backend.</p>
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
        <p>Слот свободен. Здесь можно отдавать фиксированный баннер, HTML-креатив или нативную вставку.</p>
      </div>
      <div class="slot-code">awaiting inventory</div>
    `;
    return;
  }

  elements.midrollSlot.innerHTML = `
    <div>
      <p class="slot-label">Ad Slot 728x90</p>
      <h3>Встраиваемый midroll после карточек</h3>
      <p>Можно отдавать как классический баннер или как промо-инсерцию бренда ${escapeHtml(midrollItem.brand)}.</p>
    </div>
    <div class="slot-code">best candidate: ${escapeHtml(midrollItem.banner_id)}</div>
  `;
}

function render(response, requestPayload) {
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
}

async function loadRecommendations() {
  const payload = buildRequestPayload();

  state.requestInFlight = true;
  elements.refreshButton.disabled = true;
  elements.demoButton.disabled = true;
  setStatus("Loading...", "");

  try {
    const response = await fetchRecommendations(payload);
    state.response = {
      ...response,
      retrieval_used: response.candidate_mode === "retrieval + ranking",
      online_state_applied: false,
      top_k: response.items.length,
      artifacts_dir: response.artifacts_dir || "backend-generated",
    };
    state.source = "api";
    setStatus("API mode", "ok");
  } catch (error) {
    state.response = buildLocalResponse(payload);
    state.source = "local";
    setStatus("Local fallback", "demo");
    console.warn("Failed to fetch recommendations, using local fallback:", error);
  } finally {
    state.requestInFlight = false;
    render(state.response, payload);
    elements.refreshButton.disabled = false;
    elements.demoButton.disabled = false;
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
    const payload = buildRequestPayload();
    state.response = buildLocalResponse(payload);
    state.source = "demo";
    setStatus("Demo mode", "demo");
    render(state.response, payload);
  });

  activateQuickUsers();
}

bindEvents();
renderPlacements();
render(buildLocalResponse(buildRequestPayload()), buildRequestPayload());
loadRecommendations();
