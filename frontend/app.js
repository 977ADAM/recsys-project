const DEMO_RESPONSE = {
  model_type: "demo-editorial-ranker",
  artifacts_dir: "frontend-demo",
  retrieval_used: true,
  online_state_applied: false,
  top_k: 8,
  items: [
    {
      banner_id: "b_0032",
      brand: "Luma",
      category: "education",
      subcategory: "coding",
      banner_format: "animated",
      campaign_goal: "lead_gen",
      pred_ctr: 0.184,
      final_score: 1.122,
      cpm_bid: 6.91,
      quality_score: 0.884,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.132,
      user_ctr_prior: 0.089,
      user_subcategory_ctr_prior: 0.144,
      user_banner_ctr_prior: 0.071,
      served_impressions_total: 2,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0022",
      brand: "Pulse",
      category: "travel",
      subcategory: "car_rental",
      banner_format: "video",
      campaign_goal: "purchase",
      pred_ctr: 0.163,
      final_score: 1.089,
      cpm_bid: 8.15,
      quality_score: 0.846,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.118,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.126,
      user_banner_ctr_prior: 0.05,
      served_impressions_total: 0,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0039",
      brand: "Zenit",
      category: "education",
      subcategory: "design",
      banner_format: "static",
      campaign_goal: "awareness",
      pred_ctr: 0.151,
      final_score: 0.996,
      cpm_bid: 7.75,
      quality_score: 0.677,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.11,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.12,
      user_banner_ctr_prior: 0.04,
      served_impressions_total: 1,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0029",
      brand: "Mira",
      category: "electronics",
      subcategory: "smartphones",
      banner_format: "animated",
      campaign_goal: "purchase",
      pred_ctr: 0.148,
      final_score: 0.972,
      cpm_bid: 2.97,
      quality_score: 0.7,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 0,
      interest_match_count: 0,
      banner_ctr_prior: 0.095,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.094,
      user_banner_ctr_prior: 0.02,
      served_impressions_total: 0,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0036",
      brand: "Pixel",
      category: "entertainment",
      subcategory: "cinema",
      banner_format: "static",
      campaign_goal: "lead_gen",
      pred_ctr: 0.141,
      final_score: 0.948,
      cpm_bid: 2.35,
      quality_score: 0.746,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 0,
      interest_match_count: 0,
      banner_ctr_prior: 0.097,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.09,
      user_banner_ctr_prior: 0.018,
      served_impressions_total: 0,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0033",
      brand: "Orbit",
      category: "entertainment",
      subcategory: "events",
      banner_format: "static",
      campaign_goal: "awareness",
      pred_ctr: 0.134,
      final_score: 0.901,
      cpm_bid: 8.14,
      quality_score: 0.699,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.09,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.105,
      user_banner_ctr_prior: 0.014,
      served_impressions_total: 4,
      served_clicks_total: 1,
      is_active: 1,
    },
    {
      banner_id: "b_0025",
      brand: "Pulse",
      category: "food",
      subcategory: "snacks",
      banner_format: "static",
      campaign_goal: "awareness",
      pred_ctr: 0.122,
      final_score: 0.856,
      cpm_bid: 8.19,
      quality_score: 0.731,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 0,
      interest_match_count: 0,
      banner_ctr_prior: 0.08,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.078,
      user_banner_ctr_prior: 0.01,
      served_impressions_total: 0,
      served_clicks_total: 0,
      is_active: 1,
    },
    {
      banner_id: "b_0019",
      brand: "Pixel",
      category: "sports",
      subcategory: "running",
      banner_format: "static",
      campaign_goal: "purchase",
      pred_ctr: 0.118,
      final_score: 0.834,
      cpm_bid: 9.75,
      quality_score: 0.69,
      age_match: 1,
      gender_match: 1,
      interest_match_any: 1,
      interest_match_count: 1,
      banner_ctr_prior: 0.081,
      user_ctr_prior: 0.083,
      user_subcategory_ctr_prior: 0.091,
      user_banner_ctr_prior: 0.013,
      served_impressions_total: 0,
      served_clicks_total: 0,
      is_active: 1,
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
  lastError: "",
};

const elements = {
  apiStatus: document.querySelector("#api-status"),
  controlsForm: document.querySelector("#controls-form"),
  userId: document.querySelector("#user-id"),
  topK: document.querySelector("#top-k"),
  candidateMode: document.querySelector("#candidate-mode"),
  scoreMode: document.querySelector("#score-mode"),
  retrievalTopN: document.querySelector("#retrieval-top-n"),
  endpoint: document.querySelector("#recommendation-endpoint"),
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
  return `https://example.com/${item.category}/${item.subcategory}/${item.brand.toLowerCase()}`;
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
    elements.heroStory.innerHTML = "";
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
    <div class="briefing-list">${content}</div>
  `;
}

function renderFeed(items) {
  elements.feedGrid.innerHTML = items
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
}

function renderNativeBand(items) {
  elements.nativeBand.innerHTML = items
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
}

function renderSponsoredRail(items) {
  elements.sponsoredRail.innerHTML = items
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
      value: state.source === "api" ? "live API" : "demo fallback",
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
      label: "Последняя ошибка",
      value: state.lastError || "none",
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
  const endpoint = elements.endpoint.value.trim() || "/api/v1/recommendations";

  setStatus("API loading", "");
  elements.refreshButton.disabled = true;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `HTTP ${response.status}`);
    }

    const data = await response.json();
    state.response = data;
    state.source = "api";
    state.lastError = "";
    setStatus("API live", "ok");
    render(data, payload);
  } catch (error) {
    state.response = DEMO_RESPONSE;
    state.source = "demo";
    state.lastError = String(error.message || error);
    setStatus("Demo fallback", "demo");
    render(DEMO_RESPONSE, payload);
  } finally {
    elements.refreshButton.disabled = false;
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
    state.response = DEMO_RESPONSE;
    state.source = "demo";
    state.lastError = "";
    setStatus("Demo mode", "demo");
    render(DEMO_RESPONSE, buildRequestPayload());
  });

  activateQuickUsers();
}

bindEvents();
renderPlacements();
render(DEMO_RESPONSE, buildRequestPayload());
loadRecommendations();
