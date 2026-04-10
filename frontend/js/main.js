import {
  buildLocalResponse,
  createBanner,
  createUser,
  deleteBanner,
  deleteUser,
  fetchBanners,
  fetchRecommendations,
  fetchUsers,
  updateBanner,
  updateUser,
} from "./api.js";
import {
  DEFAULT_RETRIEVAL_ARTIFACTS_DIR,
  DEMO_BANNERS,
  DEMO_USERS,
} from "./constants.js";
import { elements } from "./dom.js";
import {
  renderBannerPreview,
  renderBannersTable,
  renderBannerStats,
  renderPlacements,
  renderShowcase,
  renderUserPreview,
  renderUsersStats,
  renderUsersTable,
  setInlineFeedback,
  setStatus,
  showToast,
} from "./renderers.js";
import { createInitialState } from "./state.js";
import { loadPersistedState, savePersistedState } from "./storage.js";
import { formatMoney, formatScore } from "./utils.js";

const state = createInitialState(loadPersistedState());

function syncControlsFromDom() {
  state.controls = {
    userId: elements.userId.value.trim(),
    topK: Number(elements.topK.value),
    candidateMode: elements.candidateMode.value,
    scoreMode: elements.scoreMode.value,
    retrievalTopN: Number(elements.retrievalTopN.value),
    retrievalArtifactsDir: elements.retrievalArtifactsDir.value.trim(),
    onlyActive: elements.onlyActive.checked,
    excludeSeen: elements.excludeSeen.checked,
  };
  savePersistedState(state);
}

function syncBannerFiltersFromDom() {
  state.banners.filters.query = elements.bannerSearch.value;
  state.banners.filters.status = elements.bannerStatusFilter.value;
  state.banners.filters.sort = elements.bannerSort.value;
  state.banners.filters.pageSize = Number(elements.bannerPageSize.value);
  savePersistedState(state);
}

function syncUserFiltersFromDom() {
  state.users.filters.query = elements.userSearch.value;
  state.users.filters.premium = elements.userPremiumFilter.value;
  state.users.filters.sort = elements.userSort.value;
  state.users.filters.pageSize = Number(elements.userPageSize.value);
  savePersistedState(state);
}

function applyStateToControls() {
  elements.userId.value = state.controls.userId;
  elements.topK.value = String(state.controls.topK);
  elements.candidateMode.value = state.controls.candidateMode;
  elements.scoreMode.value = state.controls.scoreMode;
  elements.retrievalTopN.value = String(state.controls.retrievalTopN);
  elements.retrievalArtifactsDir.value = state.controls.retrievalArtifactsDir || DEFAULT_RETRIEVAL_ARTIFACTS_DIR;
  elements.onlyActive.checked = state.controls.onlyActive;
  elements.excludeSeen.checked = state.controls.excludeSeen;
  elements.bannerSearch.value = state.banners.filters.query;
  elements.bannerStatusFilter.value = state.banners.filters.status;
  elements.bannerSort.value = state.banners.filters.sort;
  elements.bannerPageSize.value = String(state.banners.filters.pageSize);
  elements.userSearch.value = state.users.filters.query;
  elements.userPremiumFilter.value = state.users.filters.premium;
  elements.userSort.value = state.users.filters.sort;
  elements.userPageSize.value = String(state.users.filters.pageSize);
}

function buildRequestPayload() {
  syncControlsFromDom();
  const payload = {
    user_id: state.controls.userId,
    top_k: Number(state.controls.topK),
    score_mode: state.controls.scoreMode,
    retrieval_top_n: Number(state.controls.retrievalTopN),
    only_active: state.controls.onlyActive,
    exclude_seen: state.controls.excludeSeen,
  };

  if (state.controls.candidateMode === "retrieval + ranking") {
    payload.retrieval_artifacts_dir = state.controls.retrievalArtifactsDir || DEFAULT_RETRIEVAL_ARTIFACTS_DIR;
  }

  return payload;
}

function renderCurrentShowcase() {
  renderShowcase(elements, state, {
    user_id: state.controls.userId,
  });
}

function renderCurrentBannerAdmin() {
  renderBannerStats(elements, state);
  renderBannersTable(elements, state);
}

function renderCurrentUsersAdmin() {
  renderUsersStats(elements, state);
  renderUsersTable(elements, state);
}

function syncBannerIntoState(savedBanner) {
  const index = state.banners.items.findIndex((item) => item.banner_id === savedBanner.banner_id);
  if (index === -1) {
    state.banners.items = [savedBanner, ...state.banners.items];
  } else {
    state.banners.items[index] = savedBanner;
  }
}

function syncUserIntoState(savedUser) {
  const index = state.users.items.findIndex((item) => item.user_id === savedUser.user_id);
  if (index === -1) {
    state.users.items = [savedUser, ...state.users.items];
  } else {
    state.users.items[index] = savedUser;
  }
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
  renderBannerPreview(elements, banner, "edit");
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
  renderUserPreview(elements, user, "edit");
}

function collectBannerDraft() {
  return {
    banner_id: elements.bannerId.value.trim(),
    brand: elements.bannerBrand.value.trim(),
    category: elements.bannerCategory.value.trim(),
    subcategory: elements.bannerSubcategory.value.trim(),
    banner_format: elements.bannerFormat.value,
    campaign_goal: elements.bannerGoal.value,
    target_gender: elements.bannerGender.value,
    target_age_min: Number(elements.bannerAgeMin.value || 0),
    target_age_max: Number(elements.bannerAgeMax.value || 0),
    cpm_bid: Number(elements.bannerCpm.value || 0),
    quality_score: Number(elements.bannerQuality.value || 0),
    created_at: elements.bannerCreatedAt.value,
    is_active: elements.bannerIsActive.checked,
    landing_page: elements.bannerLandingPage.value.trim(),
  };
}

function collectUserDraft() {
  return {
    user_id: elements.manageUserId.value.trim(),
    age: Number(elements.manageUserAge.value || 0),
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
    signup_days_ago: Number(elements.manageUserSignupDaysAgo.value || 0),
    is_premium: elements.manageUserIsPremium.checked,
  };
}

function resetBannerForm() {
  state.banners.selectedId = null;
  state.banners.formMode = "create";
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
  setInlineFeedback(elements.bannerFormErrors);
  renderBannerPreview(elements, collectBannerDraft(), "create");
  renderCurrentBannerAdmin();
}

function resetUserForm() {
  state.users.selectedId = null;
  state.users.formMode = "create";
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
  setInlineFeedback(elements.userFormErrors);
  renderUserPreview(elements, collectUserDraft(), "create");
  renderCurrentUsersAdmin();
}

function selectBanner(bannerId) {
  const banner = state.banners.items.find((item) => item.banner_id === bannerId);
  if (!banner) {
    resetBannerForm();
    return;
  }

  state.banners.selectedId = banner.banner_id;
  state.banners.formMode = "edit";
  fillBannerForm(banner);
  elements.bannerId.readOnly = true;
  elements.bannerFormTitle.textContent = `Редактирование ${banner.banner_id}`;
  elements.bannerFormCaption.textContent = "Изменения сохраняются через PATCH /api/v1/banners/{banner_id}.";
  elements.deleteBannerButton.disabled = false;
  renderCurrentBannerAdmin();
}

function selectUser(userId) {
  const user = state.users.items.find((item) => item.user_id === userId);
  if (!user) {
    resetUserForm();
    return;
  }

  state.users.selectedId = user.user_id;
  state.users.formMode = "edit";
  fillUserForm(user);
  elements.manageUserId.readOnly = true;
  elements.userFormTitle.textContent = `Редактирование ${user.user_id}`;
  elements.userFormCaption.textContent = "Изменения сохраняются через PATCH /api/v1/users/{user_id}.";
  elements.deleteUserButton.disabled = false;
  renderCurrentUsersAdmin();
}

function buildBannerPayloadFromForm() {
  const payload = collectBannerDraft();

  if (!payload.banner_id) {
    throw new Error("banner_id обязателен");
  }
  if (payload.banner_id.length > 20) {
    throw new Error("banner_id должен быть не длиннее 20 символов");
  }
  if (!payload.brand || payload.brand.length > 50) {
    throw new Error("brand обязателен и должен быть не длиннее 50 символов");
  }
  if (!payload.category || payload.category.length > 50) {
    throw new Error("category обязателен и должен быть не длиннее 50 символов");
  }
  if (!payload.subcategory || payload.subcategory.length > 100) {
    throw new Error("subcategory обязателен и должен быть не длиннее 100 символов");
  }
  if (payload.target_age_max < payload.target_age_min) {
    throw new Error("target_age_max должен быть больше или равен target_age_min");
  }
  if (payload.quality_score < 0 || payload.quality_score > 1) {
    throw new Error("quality_score должен быть в диапазоне от 0 до 1");
  }
  if (!payload.created_at) {
    throw new Error("created_at обязателен");
  }
  try {
    new URL(payload.landing_page);
  } catch {
    throw new Error("landing_page должен быть корректным URL");
  }

  return payload;
}

function buildUserPayloadFromForm() {
  const payload = collectUserDraft();

  if (!payload.user_id) {
    throw new Error("user_id обязателен");
  }
  if (payload.user_id.length > 20) {
    throw new Error("user_id должен быть не длиннее 20 символов");
  }
  if (payload.country.length !== 2) {
    throw new Error("country должен содержать 2 символа");
  }
  if (payload.city_tier.length > 10 || payload.income_band.length > 10 || payload.activity_segment.length > 10) {
    throw new Error("city_tier, income_band и activity_segment должны быть не длиннее 10 символов");
  }

  return payload;
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

async function loadRecommendations() {
  const payload = buildRequestPayload();
  state.loading.recommendations = true;
  elements.refreshButton.disabled = true;
  elements.demoButton.disabled = true;
  setStatus(elements, "Loading recommendations");
  setInlineFeedback(elements.recommendationFeedback, "Запросили рекомендации у backend и подготовили fallback на случай ошибки.", "info");
  renderCurrentShowcase();

  try {
    state.response = await fetchRecommendations(payload, state.controls);
    state.source = "api";
    setStatus(elements, "API ready", "ok");
    setInlineFeedback(elements.recommendationFeedback, `Рекомендации обновлены для ${state.response.user_id}.`, "success");
  } catch (error) {
    state.response = buildLocalResponse(payload, state.controls);
    state.source = "local";
    setStatus(elements, "Local fallback", "demo");
    setInlineFeedback(elements.recommendationFeedback, `Backend недоступен: ${error.message}`, "warning");
    showToast(elements, `Recommendations fallback: ${error.message}`, "warning");
  } finally {
    state.loading.recommendations = false;
    renderCurrentShowcase();
    elements.refreshButton.disabled = false;
    elements.demoButton.disabled = false;
    savePersistedState(state);
  }
}

async function loadBanners({ preserveSelection = true } = {}) {
  state.banners.loading = true;
  state.banners.error = "";
  elements.reloadBannersButton.disabled = true;
  elements.newBannerButton.disabled = true;
  setStatus(elements, "Syncing banners");
  renderCurrentBannerAdmin();

  const previousSelection = preserveSelection ? state.banners.selectedId : null;

  try {
    state.banners.items = await fetchBanners();
    state.banners.source = "api";
    setStatus(elements, "Inventory synced", "ok");
  } catch (error) {
    state.banners.items = [...DEMO_BANNERS];
    state.banners.source = "demo";
    state.banners.error = error.message;
    setStatus(elements, "Demo inventory", "demo");
    showToast(elements, `Banners fallback: ${error.message}`, "warning");
  } finally {
    state.banners.loading = false;
    renderCurrentBannerAdmin();
    renderCurrentShowcase();
    if (previousSelection && state.banners.items.some((banner) => banner.banner_id === previousSelection)) {
      selectBanner(previousSelection);
    } else {
      resetBannerForm();
    }
    elements.reloadBannersButton.disabled = false;
    elements.newBannerButton.disabled = false;
    savePersistedState(state);
  }
}

async function loadUsers({ preserveSelection = true } = {}) {
  state.users.loading = true;
  state.users.error = "";
  elements.reloadUsersButton.disabled = true;
  elements.newUserButton.disabled = true;
  setStatus(elements, "Syncing users");
  renderCurrentUsersAdmin();

  const previousSelection = preserveSelection ? state.users.selectedId : null;

  try {
    state.users.items = await fetchUsers();
    state.users.source = "api";
    setStatus(elements, "Users synced", "ok");
  } catch (error) {
    state.users.items = [...DEMO_USERS];
    state.users.source = "demo";
    state.users.error = error.message;
    setStatus(elements, "Demo users", "demo");
    showToast(elements, `Users fallback: ${error.message}`, "warning");
  } finally {
    state.users.loading = false;
    renderCurrentUsersAdmin();
    if (previousSelection && state.users.items.some((user) => user.user_id === previousSelection)) {
      selectUser(previousSelection);
    } else {
      resetUserForm();
    }
    elements.reloadUsersButton.disabled = false;
    elements.newUserButton.disabled = false;
    savePersistedState(state);
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
  renderCurrentShowcase();
  savePersistedState(state);
}

async function handleBannerSubmit(event) {
  event.preventDefault();
  elements.saveBannerButton.disabled = true;
  setInlineFeedback(elements.bannerFormErrors);

  try {
    const payload = buildBannerPayloadFromForm();
    const saved =
      state.banners.formMode === "create"
        ? await createBanner(payload)
        : await updateBanner(payload.banner_id, buildBannerPatchPayload(payload));
    syncBannerIntoState(saved);
    state.banners.source = "api";
    selectBanner(saved.banner_id);
    setInlineFeedback(
      elements.bannerFormErrors,
      state.banners.formMode === "create" ? `Баннер ${saved.banner_id} создан.` : `Баннер ${saved.banner_id} обновлен.`,
      "success",
    );
    showToast(elements, `Баннер ${saved.banner_id} сохранен`, "success");
  } catch (error) {
    setInlineFeedback(elements.bannerFormErrors, error.message, "error");
    showToast(elements, `Не удалось сохранить: ${error.message}`, "error");
  } finally {
    renderCurrentBannerAdmin();
    renderCurrentShowcase();
    elements.saveBannerButton.disabled = false;
    savePersistedState(state);
  }
}

async function handleDeleteBanner() {
  if (!state.banners.selectedId) {
    return;
  }

  const bannerId = state.banners.selectedId;
  if (!window.confirm(`Удалить баннер ${bannerId}?`)) {
    return;
  }

  elements.deleteBannerButton.disabled = true;
  try {
    await deleteBanner(bannerId);
    state.banners.items = state.banners.items.filter((banner) => banner.banner_id !== bannerId);
    resetBannerForm();
    showToast(elements, `Баннер ${bannerId} удален`, "success");
  } catch (error) {
    setInlineFeedback(elements.bannerFormErrors, error.message, "error");
    showToast(elements, `Не удалось удалить: ${error.message}`, "error");
  } finally {
    renderCurrentBannerAdmin();
    renderCurrentShowcase();
    elements.deleteBannerButton.disabled = state.banners.formMode !== "edit";
    savePersistedState(state);
  }
}

async function handleUserSubmit(event) {
  event.preventDefault();
  elements.saveUserButton.disabled = true;
  setInlineFeedback(elements.userFormErrors);

  try {
    const payload = buildUserPayloadFromForm();
    const saved =
      state.users.formMode === "create"
        ? await createUser(payload)
        : await updateUser(payload.user_id, buildUserPatchPayload(payload));
    syncUserIntoState(saved);
    state.users.source = "api";
    selectUser(saved.user_id);
    setInlineFeedback(
      elements.userFormErrors,
      state.users.formMode === "create" ? `Пользователь ${saved.user_id} создан.` : `Пользователь ${saved.user_id} обновлен.`,
      "success",
    );
    showToast(elements, `Пользователь ${saved.user_id} сохранен`, "success");
  } catch (error) {
    setInlineFeedback(elements.userFormErrors, error.message, "error");
    showToast(elements, `Не удалось сохранить пользователя: ${error.message}`, "error");
  } finally {
    renderCurrentUsersAdmin();
    elements.saveUserButton.disabled = false;
    savePersistedState(state);
  }
}

async function handleDeleteUser() {
  if (!state.users.selectedId) {
    return;
  }

  const userId = state.users.selectedId;
  if (!window.confirm(`Удалить пользователя ${userId}?`)) {
    return;
  }

  elements.deleteUserButton.disabled = true;
  try {
    await deleteUser(userId);
    state.users.items = state.users.items.filter((user) => user.user_id !== userId);
    resetUserForm();
    showToast(elements, `Пользователь ${userId} удален`, "success");
  } catch (error) {
    setInlineFeedback(elements.userFormErrors, error.message, "error");
    showToast(elements, `Не удалось удалить пользователя: ${error.message}`, "error");
  } finally {
    renderCurrentUsersAdmin();
    elements.deleteUserButton.disabled = state.users.formMode !== "edit";
    savePersistedState(state);
  }
}

function handleBannerDraftChange() {
  renderBannerPreview(elements, collectBannerDraft(), state.banners.formMode);
}

function handleUserDraftChange() {
  renderUserPreview(elements, collectUserDraft(), state.users.formMode);
}

function activateQuickUsers() {
  document.querySelectorAll("[data-user-id]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.userId.value = button.dataset.userId || "";
      syncControlsFromDom();
      loadRecommendations();
    });
  });
}

function bindEvents() {
  elements.controlsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadRecommendations();
  });

  [
    elements.userId,
    elements.topK,
    elements.candidateMode,
    elements.scoreMode,
    elements.retrievalTopN,
    elements.retrievalArtifactsDir,
    elements.onlyActive,
    elements.excludeSeen,
  ].forEach((element) => {
    element.addEventListener("change", syncControlsFromDom);
  });

  elements.refreshButton.addEventListener("click", () => {
    loadRecommendations();
  });

  elements.demoButton.addEventListener("click", () => {
    const payload = buildRequestPayload();
    state.response = buildLocalResponse(payload, state.controls);
    state.source = "demo";
    setStatus(elements, "Demo mode", "demo");
    setInlineFeedback(elements.recommendationFeedback, "Витрина переключена во встроенный demo режим.", "info");
    renderCurrentShowcase();
    showToast(elements, "Витрина переключена в demo режим", "info");
  });

  elements.showcaseTab.addEventListener("click", () => switchView("showcase"));
  elements.adminTab.addEventListener("click", () => switchView("admin"));
  elements.usersTab.addEventListener("click", () => switchView("users"));

  [
    elements.bannerSearch,
    elements.bannerStatusFilter,
    elements.bannerSort,
    elements.bannerPageSize,
  ].forEach((element) => {
    element.addEventListener("input", () => {
      state.banners.filters.page = 1;
      syncBannerFiltersFromDom();
      renderCurrentBannerAdmin();
    });
    element.addEventListener("change", () => {
      state.banners.filters.page = 1;
      syncBannerFiltersFromDom();
      renderCurrentBannerAdmin();
    });
  });

  elements.bannersPrevPage.addEventListener("click", () => {
    state.banners.filters.page = Math.max(1, state.banners.filters.page - 1);
    renderCurrentBannerAdmin();
    savePersistedState(state);
  });

  elements.bannersNextPage.addEventListener("click", () => {
    state.banners.filters.page += 1;
    renderCurrentBannerAdmin();
    savePersistedState(state);
  });

  elements.reloadBannersButton.addEventListener("click", () => loadBanners());
  elements.newBannerButton.addEventListener("click", () => {
    resetBannerForm();
    showToast(elements, "Форма готова для создания нового баннера", "info");
  });
  elements.bannerForm.addEventListener("submit", handleBannerSubmit);
  elements.resetBannerButton.addEventListener("click", resetBannerForm);
  elements.deleteBannerButton.addEventListener("click", handleDeleteBanner);
  elements.bannersTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-banner]");
    if (button) {
      selectBanner(button.dataset.selectBanner);
    }
  });
  elements.bannersMobileList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-banner]");
    if (button) {
      selectBanner(button.dataset.selectBanner);
    }
  });

  Array.from(elements.bannerForm.elements).forEach((field) => {
    if (field instanceof HTMLElement && "addEventListener" in field) {
      field.addEventListener("input", handleBannerDraftChange);
      field.addEventListener("change", handleBannerDraftChange);
    }
  });

  [
    elements.userSearch,
    elements.userPremiumFilter,
    elements.userSort,
    elements.userPageSize,
  ].forEach((element) => {
    element.addEventListener("input", () => {
      state.users.filters.page = 1;
      syncUserFiltersFromDom();
      renderCurrentUsersAdmin();
    });
    element.addEventListener("change", () => {
      state.users.filters.page = 1;
      syncUserFiltersFromDom();
      renderCurrentUsersAdmin();
    });
  });

  elements.usersPrevPage.addEventListener("click", () => {
    state.users.filters.page = Math.max(1, state.users.filters.page - 1);
    renderCurrentUsersAdmin();
    savePersistedState(state);
  });

  elements.usersNextPage.addEventListener("click", () => {
    state.users.filters.page += 1;
    renderCurrentUsersAdmin();
    savePersistedState(state);
  });

  elements.reloadUsersButton.addEventListener("click", () => loadUsers());
  elements.newUserButton.addEventListener("click", () => {
    resetUserForm();
    showToast(elements, "Форма готова для создания нового пользователя", "info");
  });
  elements.userForm.addEventListener("submit", handleUserSubmit);
  elements.resetUserButton.addEventListener("click", resetUserForm);
  elements.deleteUserButton.addEventListener("click", handleDeleteUser);
  elements.usersTableBody.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-user]");
    if (button) {
      selectUser(button.dataset.selectUser);
    }
  });
  elements.usersMobileList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-select-user]");
    if (button) {
      selectUser(button.dataset.selectUser);
    }
  });

  Array.from(elements.userForm.elements).forEach((field) => {
    if (field instanceof HTMLElement && "addEventListener" in field) {
      field.addEventListener("input", handleUserDraftChange);
      field.addEventListener("change", handleUserDraftChange);
    }
  });

  activateQuickUsers();
}

function initialize() {
  applyStateToControls();
  bindEvents();
  renderPlacements(elements);
  resetBannerForm();
  resetUserForm();
  renderCurrentShowcase();
  renderCurrentBannerAdmin();
  renderCurrentUsersAdmin();
  switchView(state.view);
  loadRecommendations();
  loadBanners({ preserveSelection: false });
  loadUsers({ preserveSelection: false });
}

initialize();
