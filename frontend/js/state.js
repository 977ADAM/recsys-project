import {
  DEFAULT_BANNER_FILTERS,
  DEFAULT_CONTROLS,
  DEFAULT_USER_FILTERS,
  DEMO_BANNERS,
  DEMO_RESPONSE,
  DEMO_USERS,
} from "./constants.js";

export function createInitialState(persistedState) {
  return {
    view: persistedState?.view || "showcase",
    source: "demo",
    response: structuredClone(DEMO_RESPONSE),
    controls: { ...DEFAULT_CONTROLS, ...(persistedState?.controls || {}) },
    banners: {
      items: [...DEMO_BANNERS],
      source: "demo",
      selectedId: null,
      formMode: "create",
      filters: { ...DEFAULT_BANNER_FILTERS, ...(persistedState?.bannerFilters || {}) },
      loading: false,
      error: "",
    },
    users: {
      items: [...DEMO_USERS],
      source: "demo",
      selectedId: null,
      formMode: "create",
      filters: { ...DEFAULT_USER_FILTERS, ...(persistedState?.userFilters || {}) },
      loading: false,
      error: "",
    },
    loading: {
      recommendations: false,
    },
  };
}
