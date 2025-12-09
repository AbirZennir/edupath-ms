const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8082";

async function request(path, { method = "GET", body, token, headers } = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const message = await safeParseError(response);
    throw new Error(message || `Erreur ${response.status}`);
  }

  return response.status === 204 ? null : response.json();
}

async function safeParseError(response) {
  try {
    const data = await response.json();
    return data?.message || data?.error;
  } catch (err) {
    return null;
  }
}

export const api = {
  login: (payload) => request("/auth/login", { method: "POST", body: payload }),
  register: (payload) => request("/auth/register", { method: "POST", body: payload }),
  getDashboard: (studentId, token) =>
    request(`/dashboard/${studentId}`, { token }),
  listCourses: (params = {}, token) => {
    const query = new URLSearchParams(params).toString();
    const suffix = query ? `?${query}` : "";
    return request(`/courses${suffix}`, { token });
  },
};

export { API_BASE };
