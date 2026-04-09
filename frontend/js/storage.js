import {
  DEFAULT_BANNER_FILTERS,
  DEFAULT_CONTROLS,
  DEFAULT_USER_FILTERS,
  STORAGE_KEY,
} from "./constants.js";

export function loadPersistedState() {
  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    if (!rawValue) {
      return null;
    }

    const parsed = JSON.parse(rawValue);
    return {
      view: parsed.view || "showcase",
      controls: { ...DEFAULT_CONTROLS, ...(parsed.controls || {}) },
      bannerFilters: { ...DEFAULT_BANNER_FILTERS, ...(parsed.bannerFilters || {}) },
      userFilters: { ...DEFAULT_USER_FILTERS, ...(parsed.userFilters || {}) },
    };
  } catch {
    return null;
  }
}

export function savePersistedState(state) {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        view: state.view,
        controls: state.controls,
        bannerFilters: state.banners.filters,
        userFilters: state.users.filters,
      }),
    );
  } catch {
    // Ignore storage failures in private browsing or blocked environments.
  }
}
