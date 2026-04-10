import { placementBlueprint } from "./constants.js";
import {
  buildUrl,
  escapeHtml,
  formatCategory,
  formatMoney,
  formatPercent,
  formatScore,
  formatSubcategory,
  formatWholeNumber,
  humanize,
  paginate,
  sortBanners,
  sortUsers,
  storyHeadline,
  storySummary,
} from "./utils.js";

let toastTimer = null;

export function setStatus(elements, label, className = "") {
  elements.apiStatus.textContent = label;
  elements.apiStatus.className = `status-chip ${className}`.trim();
}

export function showToast(elements, message, tone = "info") {
  elements.toast.textContent = message;
  elements.toast.className = `toast is-visible tone-${tone}`;
  elements.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
    elements.toast.className = "toast";
  }, 2600);
}

export function setInlineFeedback(element, message = "", tone = "info") {
  if (!element) {
    return;
  }

  if (!message) {
    element.hidden = true;
    element.className = "inline-feedback";
    element.textContent = "";
    return;
  }

  element.hidden = false;
  element.className = `inline-feedback tone-${tone}`;
  element.textContent = message;
}

function renderHero(elements, item, isLoading) {
  if (isLoading) {
    elements.heroStory.innerHTML = `
      <div class="skeleton-card skeleton-card-hero">
        <div class="skeleton-line short"></div>
        <div class="skeleton-line large"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
      </div>
    `;
    return;
  }

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
      ${item.retrieval_rank ? `<span class="metric-chip">Retrieval #${escapeHtml(String(item.retrieval_rank))}</span>` : ""}
    </div>
    <div class="hero-actions">
      <a class="hero-link" href="${escapeHtml(buildUrl(item))}" target="_blank" rel="noreferrer">
        Открыть лендинг
      </a>
    </div>
  `;
}

function renderBriefing(elements, items, isLoading) {
  if (isLoading) {
    elements.newsBriefing.innerHTML = `
      <div class="section-title compact">
        <div>
          <p class="kicker">Briefing</p>
          <h2>Что еще в выдаче</h2>
        </div>
      </div>
      <div class="briefing-list">
        <div class="skeleton-card"><div class="skeleton-line large"></div><div class="skeleton-line"></div></div>
        <div class="skeleton-card"><div class="skeleton-line large"></div><div class="skeleton-line"></div></div>
      </div>
    `;
    return;
  }

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

function renderFeed(elements, items, isLoading) {
  if (isLoading) {
    elements.feedGrid.innerHTML = Array.from({ length: 4 })
      .map(
        () => `
          <article class="story-card">
            <div class="skeleton-line short"></div>
            <div class="skeleton-line large"></div>
            <div class="skeleton-line"></div>
            <div class="skeleton-line"></div>
          </article>
        `,
      )
      .join("");
    return;
  }

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

  elements.feedGrid.innerHTML =
    content ||
    `<article class="story-card story-card-empty">
      <p class="slot-label">Feed Empty</p>
      <h3 class="story-headline">Лента пока пустая</h3>
      <p class="story-summary">Сервер не вернул карточки для текущего профиля. Попробуйте другой user id или измените фильтры.</p>
    </article>`;
}

function renderNativeBand(elements, items) {
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

function renderSponsoredRail(elements, items) {
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

function renderProfile(elements, response, requestPayload) {
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
        <strong>${escapeHtml(response.candidate_mode || "all banners")}</strong>
      </div>
    </div>
  `;
}

function renderMeta(elements, state) {
  const response = state.response;
  const items = [
    {
      label: "Источник",
      value:
        state.source === "api"
          ? "backend api"
          : state.source === "local"
            ? "local fallback"
            : "demo mode",
    },
    { label: "Модель", value: response.model_type },
    { label: "Candidate mode", value: response.candidate_mode || "n/a" },
    { label: "Top K", value: String(response.top_k) },
    { label: "Retrieval", value: response.retrieval_used ? "enabled" : "disabled" },
    { label: "Cards rendered", value: String(response.items.length) },
    { label: "As of date", value: response.as_of_date || "n/a" },
    {
      label: "Режим",
      value: state.view === "admin" ? "banner admin" : state.view === "users" ? "user admin" : "showcase",
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

export function renderPlacements(elements) {
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

function renderMidroll(elements, items) {
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

function renderShowcaseOverview(elements, state) {
  const items = state.response.items || [];
  const averageCtr = items.length
    ? items.reduce((sum, item) => sum + Number(item.pred_ctr || 0), 0) / items.length
    : 0;
  const activeBanners = state.banners.items.filter((banner) => banner.is_active).length;
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
  elements.inventoryTotal.textContent = String(state.banners.items.length);
  elements.inventoryCopy.textContent = `${activeBanners} активных из ${state.banners.items.length}`;
  elements.leaderboardCode.textContent = topItem ? `feature: ${topItem.banner_id}` : "inventory syncing...";
  elements.railCode.textContent =
    state.banners.items[0]
      ? `${state.banners.items[0].banner_format} / ${state.banners.items[0].brand}`
      : "awaiting inventory";
}

export function renderShowcase(elements, state, requestPayload) {
  const items = state.response.items || [];
  renderHero(elements, items[0], state.loading.recommendations);
  renderBriefing(elements, items, state.loading.recommendations);
  renderFeed(elements, items, state.loading.recommendations);
  renderNativeBand(elements, items);
  renderSponsoredRail(elements, items);
  renderProfile(elements, state.response, requestPayload);
  renderMeta(elements, state);
  renderPlacements(elements);
  renderMidroll(elements, items);
  renderShowcaseOverview(elements, state);
}

function getFilteredBanners(state) {
  const query = state.banners.filters.query.trim().toLowerCase();
  const filtered = state.banners.items.filter((banner) => {
    const matchesStatus =
      state.banners.filters.status === "all"
        ? true
        : state.banners.filters.status === "active"
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

  return sortBanners(filtered, state.banners.filters.sort);
}

function getFilteredUsers(state) {
  const query = state.users.filters.query.trim().toLowerCase();
  const filtered = state.users.items.filter((user) => {
    const matchesPremium =
      state.users.filters.premium === "all"
        ? true
        : state.users.filters.premium === "premium"
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

  return sortUsers(filtered, state.users.filters.sort);
}

export function renderBannerStats(elements, state) {
  const activeCount = state.banners.items.filter((banner) => banner.is_active).length;
  const averageCpm = state.banners.items.length
    ? state.banners.items.reduce((sum, banner) => sum + Number(banner.cpm_bid || 0), 0) / state.banners.items.length
    : 0;
  const selectedBanner = state.banners.items.find((banner) => banner.banner_id === state.banners.selectedId);

  elements.adminTotalBanners.textContent = String(state.banners.items.length);
  elements.adminActiveBanners.textContent = String(activeCount);
  elements.adminAverageCpm.textContent = formatMoney(averageCpm);
  elements.adminSelectedBanner.textContent = selectedBanner ? selectedBanner.banner_id : "Новый";
}

export function renderBannersTable(elements, state) {
  const filtered = getFilteredBanners(state);
  const page = paginate(filtered, state.banners.filters.page, state.banners.filters.pageSize);
  state.banners.filters.page = page.page;

  elements.bannersListCaption.textContent =
    state.banners.source === "api"
      ? `Показано ${filtered.length} из ${state.banners.items.length} баннеров из backend`
      : `Показано ${filtered.length} demo-баннеров во встроенном режиме`;
  elements.bannersPaginationCaption.textContent = page.totalItems
    ? `Строки ${page.start}-${page.end} из ${page.totalItems}`
    : "Нет строк для отображения";
  elements.bannersPageIndicator.textContent = `${page.page} / ${page.totalPages}`;
  elements.bannersPrevPage.disabled = page.page <= 1;
  elements.bannersNextPage.disabled = page.page >= page.totalPages;
  elements.bannersTableEmpty.hidden = filtered.length > 0;

  if (state.banners.loading) {
    elements.bannersTableBody.innerHTML = `
      <tr><td colspan="7"><div class="table-loading">Загружаем inventory…</div></td></tr>
    `;
    elements.bannersMobileList.innerHTML = `<div class="record-card"><div class="table-loading">Загружаем inventory…</div></div>`;
    return;
  }

  elements.bannersTableBody.innerHTML = page.items
    .map((banner) => {
      const isSelected = banner.banner_id === state.banners.selectedId;
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

  elements.bannersMobileList.innerHTML = page.items
    .map((banner) => {
      const isSelected = banner.banner_id === state.banners.selectedId;
      return `
        <article class="record-card ${isSelected ? "is-selected" : ""}">
          <div class="record-card-header">
            <button class="table-link" type="button" data-select-banner="${escapeHtml(banner.banner_id)}">${escapeHtml(banner.banner_id)}</button>
            <span class="table-pill ${banner.is_active ? "is-active" : "is-inactive"}">${banner.is_active ? "active" : "inactive"}</span>
          </div>
          <div class="record-card-grid">
            <span>Brand</span><strong>${escapeHtml(banner.brand)}</strong>
            <span>Category</span><strong>${escapeHtml(formatCategory(banner))}</strong>
            <span>Format</span><strong>${escapeHtml(humanize(banner.banner_format))}</strong>
            <span>CPM</span><strong>${escapeHtml(formatMoney(banner.cpm_bid))}</strong>
          </div>
        </article>
      `;
    })
    .join("");
}

export function renderUsersStats(elements, state) {
  const premiumCount = state.users.items.filter((user) => user.is_premium).length;
  const averageAge = state.users.items.length
    ? state.users.items.reduce((sum, user) => sum + Number(user.age || 0), 0) / state.users.items.length
    : 0;
  const selectedUser = state.users.items.find((user) => user.user_id === state.users.selectedId);

  elements.adminTotalUsers.textContent = String(state.users.items.length);
  elements.adminPremiumUsers.textContent = String(premiumCount);
  elements.adminAverageAge.textContent = formatWholeNumber(averageAge);
  elements.adminSelectedUser.textContent = selectedUser ? selectedUser.user_id : "Новый";
}

export function renderUsersTable(elements, state) {
  const filtered = getFilteredUsers(state);
  const page = paginate(filtered, state.users.filters.page, state.users.filters.pageSize);
  state.users.filters.page = page.page;

  elements.usersListCaption.textContent =
    state.users.source === "api"
      ? `Показано ${filtered.length} из ${state.users.items.length} пользователей из backend`
      : `Показано ${filtered.length} demo-пользователей во встроенном режиме`;
  elements.usersPaginationCaption.textContent = page.totalItems
    ? `Строки ${page.start}-${page.end} из ${page.totalItems}`
    : "Нет строк для отображения";
  elements.usersPageIndicator.textContent = `${page.page} / ${page.totalPages}`;
  elements.usersPrevPage.disabled = page.page <= 1;
  elements.usersNextPage.disabled = page.page >= page.totalPages;
  elements.usersTableEmpty.hidden = filtered.length > 0;

  if (state.users.loading) {
    elements.usersTableBody.innerHTML = `
      <tr><td colspan="7"><div class="table-loading">Загружаем users…</div></td></tr>
    `;
    elements.usersMobileList.innerHTML = `<div class="record-card"><div class="table-loading">Загружаем users…</div></div>`;
    return;
  }

  elements.usersTableBody.innerHTML = page.items
    .map((user) => {
      const isSelected = user.user_id === state.users.selectedId;
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

  elements.usersMobileList.innerHTML = page.items
    .map((user) => {
      const isSelected = user.user_id === state.users.selectedId;
      return `
        <article class="record-card ${isSelected ? "is-selected" : ""}">
          <div class="record-card-header">
            <button class="table-link" type="button" data-select-user="${escapeHtml(user.user_id)}">${escapeHtml(user.user_id)}</button>
            <span class="table-pill ${user.is_premium ? "is-premium" : "is-standard"}">${user.is_premium ? "premium" : "standard"}</span>
          </div>
          <div class="record-card-grid">
            <span>Age</span><strong>${escapeHtml(String(user.age))}</strong>
            <span>Platform</span><strong>${escapeHtml(user.platform)}</strong>
            <span>Country</span><strong>${escapeHtml(user.country)}</strong>
            <span>Activity</span><strong>${escapeHtml(user.activity_segment)}</strong>
          </div>
        </article>
      `;
    })
    .join("");
}

export function renderBannerPreview(elements, banner, mode) {
  if (!banner) {
    elements.bannerPreview.innerHTML = `
      <div class="entity-preview-card">
        <p class="slot-label">Preview</p>
        <h3>Новый баннер</h3>
        <p>Заполните форму, чтобы увидеть, как объект выглядит для редактора и QA.</p>
      </div>
    `;
    return;
  }

  elements.bannerPreview.innerHTML = `
    <div class="entity-preview-card">
      <div class="record-card-header">
        <div>
          <p class="slot-label">${escapeHtml(mode === "edit" ? "Editing" : "Draft")}</p>
          <h3>${escapeHtml(banner.brand || "Без бренда")}</h3>
        </div>
        <span class="table-pill ${banner.is_active ? "is-active" : "is-inactive"}">${banner.is_active ? "active" : "inactive"}</span>
      </div>
      <p class="entity-preview-copy">${escapeHtml(formatCategory(banner))} / ${escapeHtml(formatSubcategory(banner.subcategory || ""))}</p>
      <div class="record-card-grid">
        <span>Banner ID</span><strong>${escapeHtml(banner.banner_id || "draft")}</strong>
        <span>Format</span><strong>${escapeHtml(humanize(banner.banner_format || "static"))}</strong>
        <span>Goal</span><strong>${escapeHtml(humanize(banner.campaign_goal || "awareness"))}</strong>
        <span>CPM</span><strong>${escapeHtml(formatMoney(banner.cpm_bid || 0))}</strong>
        <span>Quality</span><strong>${escapeHtml(formatScore(banner.quality_score || 0))}</strong>
        <span>Audience</span><strong>${escapeHtml(`${banner.target_gender || "U"}, ${banner.target_age_min || 0}-${banner.target_age_max || 0}`)}</strong>
      </div>
    </div>
  `;
}

export function renderUserPreview(elements, user, mode) {
  if (!user) {
    elements.userPreview.innerHTML = `
      <div class="entity-preview-card">
        <p class="slot-label">Preview</p>
        <h3>Новый пользователь</h3>
        <p>Заполните форму, чтобы увидеть краткое summary профиля перед сохранением.</p>
      </div>
    `;
    return;
  }

  elements.userPreview.innerHTML = `
    <div class="entity-preview-card">
      <div class="record-card-header">
        <div>
          <p class="slot-label">${escapeHtml(mode === "edit" ? "Editing" : "Draft")}</p>
          <h3>${escapeHtml(user.user_id || "new-user")}</h3>
        </div>
        <span class="table-pill ${user.is_premium ? "is-premium" : "is-standard"}">${user.is_premium ? "premium" : "standard"}</span>
      </div>
      <div class="record-card-grid">
        <span>Age</span><strong>${escapeHtml(String(user.age || 0))}</strong>
        <span>Platform</span><strong>${escapeHtml(user.platform || "-")}</strong>
        <span>Country</span><strong>${escapeHtml(user.country || "-")}</strong>
        <span>Tier</span><strong>${escapeHtml(user.city_tier || "-")}</strong>
        <span>Income</span><strong>${escapeHtml(user.income_band || "-")}</strong>
        <span>Interests</span><strong>${escapeHtml([user.interest_1, user.interest_2, user.interest_3].filter(Boolean).join(", "))}</strong>
      </div>
    </div>
  `;
}
