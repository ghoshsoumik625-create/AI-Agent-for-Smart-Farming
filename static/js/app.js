/**
 * app.js — KisanMitra Smart Farming Advisor front-end logic.
 *
 * Responsibilities:
 *  - Chat message send/receive
 *  - Quick action buttons
 *  - Geolocation detection
 *  - Live weather/price sidebar cards
 *  - Status badge update
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const chatMessages   = document.getElementById("chat-messages");
const userInput      = document.getElementById("user-input");
const btnSend        = document.getElementById("btn-send");
const btnLocate      = document.getElementById("btn-locate");
const locationInput  = document.getElementById("location");
const latInput       = document.getElementById("lat");
const lonInput       = document.getElementById("lon");
const langSelect     = document.getElementById("language");
const statusBadge    = document.getElementById("status-badge");
const loadingOverlay = document.getElementById("loading-overlay");
const weatherCard    = document.getElementById("weather-card");
const weatherContent = document.getElementById("weather-content");
const priceCard      = document.getElementById("price-card");
const priceContent   = document.getElementById("price-content");

// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      statusBadge.textContent = "● Online";
      statusBadge.classList.add("online");
    } else {
      statusBadge.textContent = "⚠ Degraded";
    }
  } catch {
    statusBadge.textContent = "✕ Offline";
  }
}
checkHealth();

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatBotResponse(text) {
  // Convert markdown-like formatting to HTML
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g,     "<em>$1</em>")
    .replace(/^(\d+\.\s)/gm,   "$1")
    .replace(/^•\s/gm,         "• ")
    .replace(/\n\n/g,          "</p><p>")
    .replace(/\n/g,            "<br>");
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setLoading(state) {
  loadingOverlay.classList.toggle("hidden", !state);
  btnSend.disabled = state;
  userInput.disabled = state;
}

// ---------------------------------------------------------------------------
// Message renderers
// ---------------------------------------------------------------------------
function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message message-user";
  div.innerHTML = `
    <div class="message-avatar">👨‍🌾</div>
    <div class="message-bubble">${escapeHtml(text)}</div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message message-bot typing-indicator";
  div.id = "typing-indicator";
  div.innerHTML = `
    <div class="message-avatar">🌾</div>
    <div class="message-bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function appendBotMessage(data) {
  const { response, intent, detected_language, latency_ms } = data;

  const intentLabel = intent ? `<span class="intent-tag">${intent.replace("_", " ")}</span><br>` : "";
  const langNote = detected_language && detected_language !== "en"
    ? `<em style="font-size:0.78rem;color:#5a6b5a;">[Detected: ${detected_language.toUpperCase()}]</em><br>`
    : "";

  const div = document.createElement("div");
  div.className = "message message-bot";
  div.innerHTML = `
    <div class="message-avatar">🌾</div>
    <div class="message-bubble">
      ${intentLabel}${langNote}
      <p>${formatBotResponse(response)}</p>
      <div class="latency">⚡ ${latency_ms} ms · IBM Granite</div>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function appendErrorMessage(msg) {
  const div = document.createElement("div");
  div.className = "message message-bot";
  div.innerHTML = `
    <div class="message-avatar">⚠️</div>
    <div class="message-bubble" style="border-color:#ef9a9a;background:#fff3e0;">
      <strong>Error:</strong> ${escapeHtml(msg)}
      <br><small>Please check your connection or try again.</small>
    </div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------
async function sendMessage(queryText) {
  const text = (queryText || userInput.value).trim();
  if (!text) return;

  userInput.value = "";
  appendUserMessage(text);
  appendTypingIndicator();
  setLoading(true);

  const payload = {
    query: text,
    location:            locationInput.value.trim() || null,
    latitude:            latInput.value ? parseFloat(latInput.value) : null,
    longitude:           lonInput.value ? parseFloat(lonInput.value) : null,
    preferred_language:  langSelect.value || null,
    commodity:           extractCommodity(text),
    crop:                extractCrop(text),
  };

  try {
    const res = await fetch("/api/query", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    removeTypingIndicator();
    setLoading(false);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendErrorMessage(err.detail || "Unknown server error.");
      return;
    }

    const data = await res.json();
    appendBotMessage(data);

    // Update sidebar cards if tool data present
    if (data.tool_data?.weather) updateWeatherCard(data.tool_data.weather);
    if (data.tool_data?.mandi_price) updatePriceCard(data.tool_data.mandi_price);

  } catch (err) {
    removeTypingIndicator();
    setLoading(false);
    appendErrorMessage(err.message);
  }
}

// ---------------------------------------------------------------------------
// Sidebar card updaters
// ---------------------------------------------------------------------------
function updateWeatherCard(w) {
  weatherCard.classList.remove("hidden");
  weatherContent.innerHTML = `
    <strong>${w.location || "–"}</strong><br>
    🌡 ${w.temperature_c ?? "–"}°C &nbsp;|&nbsp; 💧 ${w.humidity_pct ?? "–"}%<br>
    ${w.description ?? "–"}<br>
    💨 Wind: ${w.wind_speed_kmh ?? "–"} km/h
    ${w.note ? `<br><small style="color:#888">${w.note}</small>` : ""}
  `;
}

function updatePriceCard(p) {
  priceCard.classList.remove("hidden");
  priceContent.innerHTML = `
    <strong>${p.commodity || "–"}</strong><br>
    📅 ${p.date || "–"}<br>
    Min: ₹${p.min ?? "–"} &nbsp;|&nbsp; Max: ₹${p.max ?? "–"}<br>
    Modal: <strong>₹${p.modal ?? "–"}</strong> /qtl
    ${p.note ? `<br><small style="color:#888">${p.note}</small>` : ""}
  `;
}

// ---------------------------------------------------------------------------
// Commodity & crop extraction (simple keyword lookup)
// ---------------------------------------------------------------------------
const COMMODITIES = [
  "tomato","potato","onion","wheat","rice","cotton","sugarcane",
  "maize","soybean","groundnut","mustard","chilli","turmeric","garlic","ginger"
];
const CROPS = [...COMMODITIES, "paddy", "jowar", "bajra", "sunflower", "sesame"];

function extractCommodity(text) {
  const lower = text.toLowerCase();
  return COMMODITIES.find(c => lower.includes(c)) || null;
}

function extractCrop(text) {
  const lower = text.toLowerCase();
  return CROPS.find(c => lower.includes(c)) || null;
}

// ---------------------------------------------------------------------------
// Geolocation
// ---------------------------------------------------------------------------
btnLocate.addEventListener("click", () => {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    return;
  }
  btnLocate.textContent = "Locating…";
  navigator.geolocation.getCurrentPosition(
    pos => {
      latInput.value = pos.coords.latitude.toFixed(4);
      lonInput.value = pos.coords.longitude.toFixed(4);
      btnLocate.textContent = "✓ Location Set";
      setTimeout(() => (btnLocate.textContent = "📡 Use My Location"), 2000);
    },
    err => {
      alert("Could not get location: " + err.message);
      btnLocate.textContent = "📡 Use My Location";
    }
  );
});

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
btnSend.addEventListener("click", () => sendMessage());

userInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Quick action buttons
document.querySelectorAll(".quick-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const query = btn.dataset.query;
    userInput.value = query;
    sendMessage(query);
  });
});
