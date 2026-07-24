(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const composer = $("#composer");
  const input = $("#input");
  const sendButton = $("#sendBtn");
  const messages = $("#messages");
  const emptyState = $("#emptyState");
  const chatScroll = $("#chatScroll");
  const themeToggle = $("#themeToggle");
  const readinessBadge = $("#readinessBadge");
  const chatList = $("#chatList");
  const sidebar = $("#sidebar");
  const sidebarBackdrop = $("#sidebarBackdrop");
  const sourceDrawer = $("#sourceDrawer");
  const sourceDrawerContent = $("#sourceDrawerContent");

  let controller = null;
  let progressTimer = null;
  let lastQuestion = "";
  let currentSessionId = crypto.randomUUID();
  const sessions = new Map();
  const conversationHistory = [];
  const sourceRegistry = new Map();

  const escapeHtml = (value) => String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const progressSteps = ["질문 분석", "약관 검색", "근거 검증", "답변 작성"];

  function setBusy(busy) {
    sendButton.classList.toggle("stop", busy);
    sendButton.disabled = !busy && input.value.trim().length < 2;
    input.disabled = false;
    sendButton.setAttribute("aria-label", busy ? "분석 중지" : "분석 시작");
  }

  function autoresize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  }

  function statusLabel(status) {
    return {
      completed: "분석 완료",
      limited: "제한된 답변",
      needs_information: "추가 정보 필요",
      insufficient_evidence: "근거 부족",
      blocked: "분석 제한",
      partial: "일부 완료",
      failed: "분석 실패",
    }[status] || status;
  }

  function scrollToLatest() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function saveCurrentSession() {
    const existing = sessions.get(currentSessionId) || {};
    sessions.set(currentSessionId, {
      ...existing,
      id: currentSessionId,
      title: existing.title || "새 대화",
      html: messages.innerHTML,
      history: conversationHistory.map((item) => ({...item})),
    });
    renderChatList();
  }

  function renderChatList() {
    chatList.innerHTML = [...sessions.values()].map((session) => `
      <button class="chat-item ${session.id === currentSessionId ? "active" : ""}"
              data-session-id="${escapeHtml(session.id)}">
        <span class="chat-item-title">${escapeHtml(session.title)}</span>
        <span class="chat-del" data-delete-session="${escapeHtml(session.id)}" aria-label="대화 삭제">✕</span>
      </button>
    `).join("");
  }

  function resetConversation() {
    if (controller) controller.abort();
    saveCurrentSession();
    currentSessionId = crypto.randomUUID();
    conversationHistory.splice(0);
    messages.innerHTML = "";
    emptyState.classList.remove("hidden");
    sessions.set(currentSessionId, {
      id: currentSessionId,
      title: "새 대화",
      html: "",
      history: [],
    });
    renderChatList();
    input.focus();
  }

  function switchSession(id) {
    if (id === currentSessionId || !sessions.has(id)) return;
    if (controller) controller.abort();
    saveCurrentSession();
    const session = sessions.get(id);
    currentSessionId = id;
    messages.innerHTML = session.html;
    conversationHistory.splice(0, conversationHistory.length, ...session.history);
    emptyState.classList.toggle("hidden", Boolean(session.html));
    renderChatList();
    closeSidebar();
    scrollToLatest();
  }

  function appendUserMessage(question) {
    messages.insertAdjacentHTML(
      "beforeend",
      `<article class="msg user"><div class="msg-content">${escapeHtml(question)}</div></article>`,
    );
    emptyState.classList.add("hidden");
    const session = sessions.get(currentSessionId);
    if (session && session.title === "새 대화") {
      session.title = question.length > 28 ? `${question.slice(0, 28)}…` : question;
      renderChatList();
    }
    scrollToLatest();
  }

  function appendLoading() {
    const loadingId = `analysis-${Date.now()}`;
    messages.insertAdjacentHTML(
      "beforeend",
      `<article class="msg assistant analysis-loading" id="${loadingId}">
        <span class="msg-spark thinking"></span>
        <div class="progress-panel">
          <strong>보험 약관을 분석하고 있습니다</strong>
          <ol>${progressSteps.map((step, index) =>
            `<li class="${index === 0 ? "active" : ""}">${escapeHtml(step)}</li>`
          ).join("")}</ol>
        </div>
      </article>`,
    );
    let index = 0;
    progressTimer = window.setInterval(() => {
      index = Math.min(index + 1, progressSteps.length - 1);
      document.querySelectorAll(`#${loadingId} li`).forEach((item, itemIndex) => {
        item.classList.toggle("done", itemIndex < index);
        item.classList.toggle("active", itemIndex === index);
      });
    }, 5500);
    scrollToLatest();
    return loadingId;
  }

  function clearLoading(loadingId) {
    window.clearInterval(progressTimer);
    progressTimer = null;
    document.getElementById(loadingId)?.remove();
  }

  function collapsibleSection(title, values, className = "") {
    if (!values?.length) return "";
    return `<details class="answer-section ${className}">
      <summary>${escapeHtml(title)} <span>${values.length}</span></summary>
      <ul>${values.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>
    </details>`;
  }

  function registerSources(sources) {
    if (!sources?.length) return "";
    const sourceId = crypto.randomUUID();
    sourceRegistry.set(sourceId, sources);
    return `<button class="inline-action source-action" data-source-id="${sourceId}">근거 보기 (${sources.length})</button>`;
  }

  function renderSubAnswers(subAnswers) {
    if (!subAnswers?.length) return "";
    return `<details class="answer-section sub-answers">
      <summary>세부 분석 <span>${subAnswers.length}</span></summary>
      ${subAnswers.map((item) => `
        <section class="sub-answer">
          <div class="result-meta">
            <span>${escapeHtml(statusLabel(item.status))}</span>
            <span>합의도 ${escapeHtml(item.model_agreement || "확인 불가")}</span>
          </div>
          <h3>${escapeHtml(item.question)}</h3>
          <p>${escapeHtml(item.answer || "답변을 생성하지 못했습니다.")}</p>
          ${registerSources(item.sources)}
        </section>
      `).join("")}
    </details>`;
  }

  function followUpQuestions(result) {
    const questions = [];
    if (result.missing_information?.length) questions.push("추가로 필요한 정보를 구체적으로 알려줘.");
    if (result.important_exceptions?.length) questions.push("예외 조건을 사례와 함께 자세히 설명해줘.");
    if (result.sources?.length) questions.push("근거 조항별로 핵심 문장을 정리해줘.");
    questions.push("보험금 청구 전에 담당자에게 확인할 내용을 정리해줘.");
    return [...new Set(questions)].slice(0, 3);
  }

  function informationGuidance(result) {
    if (!["needs_information", "insufficient_evidence", "failed"].includes(result.status)) {
      return "";
    }
    if (result.status === "failed") {
      return `<section class="information-guidance system-failure">
        <strong>입력하신 정보의 문제가 아닙니다</strong>
        <p>분석 과정에서 일시적인 오류가 발생했습니다. 같은 질문으로 다시 시도하거나 잠시 후 이용해 주세요.</p>
        <button class="retry-action">같은 질문 다시 시도</button>
      </section>`;
    }
    const missing = result.missing_information?.length
      ? result.missing_information
      : [
          "보험상품명 또는 약관명",
          "확인하려는 보장·특약명",
          "보험금 지급 여부와 관련된 상황",
        ];
    const title = result.status === "insufficient_evidence"
      ? "등록된 약관에서 충분한 근거를 찾지 못했습니다"
      : "정확한 답변을 위해 정보가 더 필요합니다";
    const description = result.status === "insufficient_evidence"
      ? "질문이 잘못된 것은 아닙니다. 아래 정보 중 아는 내용을 추가하면 관련 조항을 더 정확히 찾을 수 있습니다."
      : "아래 항목을 알려주시면 앞선 질문과 연결해 다시 분석하겠습니다.";
    return `<section class="information-guidance">
      <strong>${title}</strong>
      <p>${description}</p>
      <ul>${missing.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <div class="context-actions">
        <button class="context-fill" data-template="보험상품명은 [상품명]입니다. 앞 질문을 다시 확인해 주세요.">상품명 추가</button>
        <button class="context-fill" data-template="확인하려는 보장 또는 특약은 [보장·특약명]입니다.">보장·특약 추가</button>
        <button class="context-fill" data-template="제 상황은 [가입 시기, 사고 또는 진단 내용]입니다. 지급 조건을 확인해 주세요.">상황 추가</button>
      </div>
    </section>`;
  }

  function renderResult(question, result, loadingId) {
    clearLoading(loadingId);
    const messageId = crypto.randomUUID();
    const notices = [
      result.requires_disclaimer ? "면책 안내가 필요한 답변입니다." : null,
      result.requires_human_review ? "사람의 추가 검토가 필요합니다." : null,
      ...(result.warnings || []),
    ].filter(Boolean);
    const answer = result.answer || result.key_points?.join("\n") || "제공된 문서에서 충분한 답변 근거를 찾지 못했습니다.";
    const followUps = followUpQuestions(result);
    const agreementLabel = {
      high: "핵심 판단이 대체로 일치합니다.",
      medium: "일부 판단에서 관점 차이가 있습니다.",
      low: "중요한 판단에서 AI들의 의견이 갈렸습니다.",
      indeterminate: "비교할 판단 근거가 충분하지 않습니다.",
    }[result.model_agreement] || "AI 판단의 합의 정도를 확인할 수 없습니다.";
    const disagreements = result.disagreements || [];
    messages.insertAdjacentHTML("beforeend", `
      <article class="msg assistant result-card" id="message-${messageId}" data-question="${escapeHtml(question)}">
        <span class="msg-spark"></span>
        <div class="msg-body">
          <div class="result-meta">
            <span class="status-pill ${escapeHtml(result.status)}">${escapeHtml(statusLabel(result.status))}</span>
            ${result.model_agreement ? `<span>모델 합의도 ${escapeHtml(result.model_agreement)}</span>` : ""}
          </div>
          <section class="verdict-overview">
            <p class="eyebrow">이견 분석</p>
            <h2>${escapeHtml(agreementLabel)}</h2>
            <div class="verdict-grid">
              <div class="verdict-panel consensus">
                <strong>AI들이 동의한 내용</strong>
                ${result.key_points?.length
                  ? `<ul>${result.key_points.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
                  : "<p>확인된 공통 판단이 없습니다.</p>"}
              </div>
              <div class="verdict-panel divergence">
                <strong>AI들의 판단이 갈린 내용</strong>
                ${disagreements.length
                  ? `<ul>${disagreements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
                  : `<p>${result.model_agreement === "high" ? "핵심 판단에서 확인된 이견이 없습니다." : "구체적인 이견을 확인할 근거가 부족합니다."}</p>`}
              </div>
            </div>
            ${result.applicable_conditions?.length
              ? `<div class="decision-conditions"><strong>판단에 중요한 조건</strong><ul>${result.applicable_conditions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`
              : ""}
          </section>
          <section class="final-summary">
            <p class="eyebrow">최종 정리</p>
            <div class="answer-text">${escapeHtml(answer)}</div>
          </section>
          ${informationGuidance(result)}
          <div class="primary-actions">
            ${registerSources(result.sources)}
            <button class="inline-action copy-action">복사</button>
            <button class="inline-action regenerate-action">다시 생성</button>
            <button class="inline-action feedback-action" data-feedback="up" aria-label="좋아요">👍</button>
            <button class="inline-action feedback-action" data-feedback="down" aria-label="싫어요">👎</button>
          </div>
          ${collapsibleSection("중요한 예외와 제한", result.important_exceptions, "exceptions")}
          ${collapsibleSection("추가로 필요한 정보", result.missing_information)}
          ${renderSubAnswers(result.sub_answers)}
          ${collapsibleSection("안내", notices, "notices")}
          ${result.reason_code ? `<details class="technical-details"><summary>진단 정보</summary><code>${escapeHtml(result.reason_code)}</code></details>` : ""}
          <div class="follow-ups">
            ${followUps.map((item) => `<button class="follow-up-chip">${escapeHtml(item)}</button>`).join("")}
          </div>
        </div>
      </article>
    `);
    scrollToLatest();
    saveCurrentSession();
  }

  function renderError(message, loadingId, question) {
    clearLoading(loadingId);
    messages.insertAdjacentHTML(
      "beforeend",
      `<article class="error-card" data-question="${escapeHtml(question)}">
        <strong>분석을 완료하지 못했습니다.</strong>
        <p>${escapeHtml(message)}</p>
        <button class="retry-action">다시 시도</button>
      </article>`,
    );
    saveCurrentSession();
    scrollToLatest();
  }

  function assistantHistoryText(result) {
    return result.answer
      || result.missing_information?.join(" ")
      || result.key_points?.join(" ")
      || statusLabel(result.status);
  }

  async function analyze(question) {
    lastQuestion = question;
    controller = new AbortController();
    setBusy(true);
    appendUserMessage(question);
    const loadingId = appendLoading();
    const userContext = {conversation_history: conversationHistory.slice(-8)};
    try {
      const response = await fetch("/insurance/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, user_context: userContext}),
        signal: controller.signal,
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.message || `서버 응답 오류 (${response.status})`);
      renderResult(question, result, loadingId);
      conversationHistory.push(
        {role: "user", content: question},
        {role: "assistant", content: assistantHistoryText(result)},
      );
      saveCurrentSession();
    } catch (error) {
      if (error.name === "AbortError") {
        renderError("분석을 중지했습니다.", loadingId, question);
      } else {
        renderError(error.message || "서버에 연결하지 못했습니다.", loadingId, question);
      }
    } finally {
      controller = null;
      setBusy(false);
      input.focus();
    }
  }

  function submitQuestion(question) {
    const cleanQuestion = question.trim();
    if (cleanQuestion.length < 2 || controller) return;
    input.value = "";
    autoresize();
    analyze(cleanQuestion);
  }

  function openSources(sourceId) {
    const sources = sourceRegistry.get(sourceId) || [];
    sourceDrawerContent.innerHTML = sources.map((source, index) => {
      const location = [
        source.company, source.product, source.article, source.title,
        source.page_start ? `${source.page_start}쪽` : null, source.source_file,
      ].filter(Boolean);
      return `<article class="source-card">
        <span>근거 ${index + 1}</span>
        <strong>${escapeHtml(location.join(" · ") || source.claim_id)}</strong>
        <p>Claim ID: ${escapeHtml(source.claim_id)}</p>
      </article>`;
    }).join("") || "<p>표시할 근거가 없습니다.</p>";
    sourceDrawer.classList.add("open");
    sourceDrawer.setAttribute("aria-hidden", "false");
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("show");
  }

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (controller) {
      controller.abort();
      return;
    }
    submitQuestion(input.value);
  });

  input.addEventListener("input", () => {
    autoresize();
    if (!controller) sendButton.disabled = input.value.trim().length < 2;
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  messages.addEventListener("click", async (event) => {
    const sourceButton = event.target.closest(".source-action");
    if (sourceButton) return openSources(sourceButton.dataset.sourceId);
    const followUp = event.target.closest(".follow-up-chip");
    if (followUp) return submitQuestion(followUp.textContent);
    const retry = event.target.closest(".retry-action");
    if (retry) return submitQuestion(retry.closest("[data-question]").dataset.question);
    const contextFill = event.target.closest(".context-fill");
    if (contextFill) {
      input.value = contextFill.dataset.template;
      autoresize();
      input.focus();
      sendButton.disabled = false;
      return;
    }
    const regenerate = event.target.closest(".regenerate-action");
    if (regenerate) return submitQuestion(regenerate.closest("[data-question]").dataset.question);
    const copy = event.target.closest(".copy-action");
    if (copy) {
      const text = copy.closest(".msg-body").querySelector(".answer-text").textContent;
      await navigator.clipboard.writeText(text);
      copy.textContent = "복사됨";
      return;
    }
    const feedback = event.target.closest(".feedback-action");
    if (feedback) {
      feedback.closest(".primary-actions").querySelectorAll(".feedback-action")
        .forEach((button) => button.classList.remove("selected"));
      feedback.classList.add("selected");
    }
  });

  chatList.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-session]");
    if (deleteButton) {
      event.stopPropagation();
      const id = deleteButton.dataset.deleteSession;
      sessions.delete(id);
      if (id === currentSessionId) resetConversation();
      else renderChatList();
      return;
    }
    const item = event.target.closest("[data-session-id]");
    if (item) switchSession(item.dataset.sessionId);
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => submitQuestion(chip.dataset.prompt));
  });
  $("#newChatBtn").addEventListener("click", resetConversation);
  $("#sidebarOpen").addEventListener("click", () => {
    sidebar.classList.add("open");
    sidebarBackdrop.classList.add("show");
  });
  $("#sidebarClose").addEventListener("click", closeSidebar);
  sidebarBackdrop.addEventListener("click", closeSidebar);
  $("#sourceDrawerClose").addEventListener("click", () => {
    sourceDrawer.classList.remove("open");
    sourceDrawer.setAttribute("aria-hidden", "true");
  });

  themeToggle.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark";
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    themeToggle.textContent = dark ? "🌙" : "☀️";
  });

  fetch("/insurance/readiness")
    .then((response) => response.json())
    .then((value) => {
      readinessBadge.textContent = value.status === "ready" ? "준비됨" : "설정 확인";
      readinessBadge.classList.toggle("ready", value.status === "ready");
    })
    .catch(() => { readinessBadge.textContent = "상태 확인 불가"; });

  sessions.set(currentSessionId, {id: currentSessionId, title: "새 대화", html: "", history: []});
  renderChatList();
  input.focus();
})();
