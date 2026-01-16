// lib/api.ts
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

interface FetchOptions extends Omit<RequestInit, "body" | "headers"> {
  data?: any;
  headers?: Record<string, string>;
}

export async function fetchApi(fullUrl: string, options: FetchOptions = {}) {
  const { data, headers: customHeaders = {}, ...restOptions } = options;

  // Get token from localStorage
  let token = "";
  if (typeof window !== "undefined") {
    token = localStorage.getItem("token") || "";
  }

  const headers: Record<string, string> = {
    ...(token && { Authorization: `Bearer ${token}` }),
    ...customHeaders,
  };

  if (!headers["Content-Type"] && data && !(data instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const config: RequestInit = {
    ...restOptions,
    headers,
  };

  if (data) {
    if (data instanceof FormData) config.body = data;
    else if (headers["Content-Type"] === "application/json") config.body = JSON.stringify(data);
    else if (headers["Content-Type"] === "application/x-www-form-urlencoded")
      config.body = typeof data === "string" ? data : new URLSearchParams(data).toString();
    else config.body = data;
  }

  try {
    const response = await fetch(fullUrl, config);

    if (response.status === 401) {
      if (typeof window !== "undefined") {
        localStorage.removeItem("token");
        window.location.href = "/login";
      }
      throw new ApiError(401, "Unauthorized - Please log in again");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(response.status, errorData.message || errorData.detail || "An error occurred");
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(500, "Network error or server is unreachable");
  }
}

export const api = {
  get: (url: string, options?: Omit<FetchOptions, "method">) =>
    fetchApi(`http://127.0.0.1:8000${url}`, { ...options, method: "GET" }),

  post: (url: string, data?: any, options?: Omit<FetchOptions, "method">) =>
    fetchApi(`http://127.0.0.1:8000${url}`, { ...options, method: "POST", data }),

  put: (url: string, data?: any, options?: Omit<FetchOptions, "method">) =>
    fetchApi(`http://127.0.0.1:8000${url}`, { ...options, method: "PUT", data }),

  delete: (url: string, options?: Omit<FetchOptions, "method">) =>
    fetchApi(`http://127.0.0.1:8000${url}`, { ...options, method: "DELETE" }),

  patch: (url: string, data?: any, options?: Omit<FetchOptions, "method">) =>
    fetchApi(`http://127.0.0.1:8000${url}`, { ...options, method: "PATCH", data }),
};
