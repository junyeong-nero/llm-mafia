"use strict";

const els = {
    runState: document.getElementById("run-state"),
    runStateIcon: document.querySelector("#run-state .state-chip-icon"),
    runStateText: document.querySelector("#run-state .state-chip-text"),
    cycleIcon: document.getElementById("cycle-icon"),
    cycleCurrent: document.getElementById("cycle-current"),
    aliveCount: document.getElementById("alive-count"),
    playerTable: document.getElementById("player-status"),
    playerTableBody: document.querySelector("#player-status tbody"),
    playerTableEmpty: document.getElementById("player-status-empty"),
    speechQueueProgress: document.getElementById("speech-queue-progress"),
    speechQueueProgressFill: document.querySelector("#speech-queue-progress .progress-fill"),
    speechQueueProgressCaption: document.querySelector("#speech-queue-progress .progress-caption"),
    speechQueueCards: document.getElementById("speech-queue-cards"),
    speechQueueEmpty: document.getElementById("speech-queue-empty"),
    configPath: document.getElementById("config-path"),
    seed: document.getElementById("seed"),
    maxRounds: document.getElementById("max-rounds"),
    runner: document.getElementById("runner"),
    runButton: document.getElementById("run-match"),
    runButtonIcon: document.querySelector("#run-match .button-icon"),
    runButtonText: document.querySelector("#run-match .button-text"),
    matchStatus: document.getElementById("match-status"),
    matchStatusText: document.querySelector("#match-status .status-line-text"),
    retriesSection: document.getElementById("retries-section"),
    retriesList: document.getElementById("retries-list"),
    matchProgress: document.getElementById("match-progress"),
    matchProgressFill: document.querySelector("#match-progress .progress-fill"),
    matchProgressCaption: document.querySelector("#match-progress .progress-caption"),
    artifacts: document.getElementById("artifacts"),
    artifactLog: document.getElementById("artifact-log"),
    artifactEvents: document.getElementById("artifact-events"),
    artifactSummary: document.getElementById("artifact-summary"),
    chatFeed: document.getElementById("chat-feed"),
    chatEmpty: document.getElementById("chat-empty"),
    jumpLatest: document.getElementById("jump-latest"),
    consoleLive: document.getElementById("console-live"),
    consoleLiveText: document.querySelector("#console-live .console-live-text"),
};

const MAX_PROGRESS_PERCENT = 95;
const SCROLL_STICK_THRESHOLD_PX = 80;

const RUN_STATES = {
    idle: { icon: "circle-check", text: "Idle" },
    running: { icon: "loader-circle", text: "Running" },
    complete: { icon: "flag", text: "Complete" },
    error: { icon: "circle-alert", text: "Error" },
};

const ROLE_ICONS = {
    mafia: "venetian-mask",
    police: "shield-check",
    doctor: "stethoscope",
    citizen: "user",
};

const CHAT_KIND_LABELS = {
    setup: "Setup",
    speech: "Speech",
    mafia_chat: "Mafia chat",
    night_result: "Night result",
    vote_result: "Vote result",
    game_end: "Game end",
};

let eventSource = null;

// ---------- helpers ----------

function createElement(tag, className, textContent) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (textContent !== undefined && textContent !== null) {
        node.textContent = textContent;
    }
    return node;
}

function setIcon(container, name, className) {
    container.replaceChildren(icon(name, className));
}

function setStatus(message, isError) {
    els.matchStatus.hidden = message === null;
    if (message !== null) {
        els.matchStatusText.textContent = message;
        els.matchStatus.classList.toggle("error", Boolean(isError));
    }
}

function setRunState(state) {
    const config = RUN_STATES[state] || RUN_STATES.idle;
    els.runState.dataset.state = state;
    els.runStateText.textContent = config.text;
    setIcon(els.runStateIcon, config.icon, state === "running" ? "icon-spin" : "");
    const isLive = state === "running";
    els.consoleLive.dataset.live = String(isLive);
    els.consoleLiveText.textContent = isLive ? "Live" : "Offline";
}

function setProgress(percent, text) {
    els.matchProgressFill.style.width = `${percent}%`;
    els.matchProgressCaption.textContent = text;
}

function setRunning(isRunning) {
    els.runButton.disabled = isRunning;
    els.runButtonText.textContent = isRunning ? "Match running" : "Run Full Match";
    setIcon(els.runButtonIcon, isRunning ? "loader-circle" : "play", isRunning ? "icon-spin" : "");
    if (isRunning) setRunState("running");
}

// ---------- state rendering ----------

function cycleIconName(current) {
    const text = (current || "").toLowerCase();
    if (text.includes("voting")) return "gavel";
    if (text.startsWith("night")) return "moon";
    if (text.startsWith("day")) return "sun";
    if (text.includes("over")) return "flag";
    return "settings";
}

function renderCycle(current) {
    els.cycleCurrent.textContent = current;
    setIcon(els.cycleIcon, cycleIconName(current));
}

function iconLabel(iconName, text, className) {
    const node = createElement("span", className);
    node.appendChild(icon(iconName));
    node.appendChild(createElement("span", null, text));
    return node;
}

function renderPlayers(playersStatus) {
    const players = playersStatus || [];
    els.playerTable.hidden = players.length === 0;
    els.playerTableEmpty.hidden = players.length > 0;
    els.playerTableBody.replaceChildren();
    for (const player of players) {
        const row = document.createElement("tr");
        row.className = player.alive ? "player-row" : "player-row player-row--dead";
        row.appendChild(createElement("td", "player-name", player.name));
        row.appendChild(createElement("td", "player-model", player.model_name));

        const roleCell = document.createElement("td");
        roleCell.appendChild(
            iconLabel(ROLE_ICONS[player.role] || "user", player.role, `role-chip role-chip--${player.role}`),
        );
        row.appendChild(roleCell);

        const statusCell = createElement("td", player.alive ? "status-alive" : "status-dead");
        const statusText = player.alive ? "Alive" : "Dead";
        statusCell.title = statusText;
        statusCell.appendChild(icon(player.alive ? "heart-pulse" : "skull"));
        statusCell.appendChild(createElement("span", "visually-hidden", statusText));
        row.appendChild(statusCell);
        els.playerTableBody.appendChild(row);
    }

    const aliveCount = players.filter((player) => player.alive).length;
    els.aliveCount.hidden = players.length === 0;
    els.aliveCount.textContent = `${aliveCount} of ${players.length} players alive`;
}

function renderSpeechQueue(queueState) {
    const queue = (queueState && queueState.queue) || [];
    const active = (queueState && queueState.active_speaker) || null;
    const total = (queueState && queueState.total_slots) || 0;
    const completed = (queueState && queueState.completed_slots) || 0;

    const hasSlots = total > 0;
    els.speechQueueProgress.hidden = !hasSlots;
    if (hasSlots) {
        els.speechQueueProgressFill.style.width = `${(completed / total) * 100}%`;
        els.speechQueueProgressCaption.textContent = `Completed ${completed}/${total}`;
    }

    els.speechQueueCards.replaceChildren();
    if (active) {
        const card = createElement("div", "speech-queue-card speech-queue-card--active");
        card.appendChild(iconLabel("mic", "Now", "speech-queue-index"));
        card.appendChild(createElement("span", "speech-queue-name", active));
        els.speechQueueCards.appendChild(card);
    }

    queue.forEach((speaker, index) => {
        const card = createElement("div", "speech-queue-card speech-queue-card--pending");
        card.appendChild(createElement("span", "speech-queue-index", String(index + 1)));
        card.appendChild(createElement("span", "speech-queue-name", speaker));
        els.speechQueueCards.appendChild(card);
    });

    let emptyText = "No pending speakers.";
    if (hasSlots && queue.length === 0 && !active) {
        emptyText = "All queued speeches finished.";
    }
    els.speechQueueEmpty.textContent = emptyText;
    els.speechQueueEmpty.hidden = queue.length > 0 || Boolean(active);
}

// ---------- chat rendering ----------

function voteTable(rows) {
    const table = document.createElement("table");
    table.className = "chat-vote-table";
    const head = table.createTHead().insertRow();
    for (const label of ["Type", "Voter", "Target"]) {
        head.appendChild(createElement("th", null, label));
    }
    const body = table.createTBody();
    for (const row of rows) {
        const tr = body.insertRow();
        tr.appendChild(createElement("td", null, row.type));
        tr.appendChild(createElement("td", null, row.voter));
        tr.appendChild(createElement("td", null, row.target));
    }
    return table;
}

function isFeedNearBottom() {
    const feed = els.chatFeed;
    return feed.scrollHeight - feed.scrollTop - feed.clientHeight < SCROLL_STICK_THRESHOLD_PX;
}

function scrollFeedToBottom() {
    els.chatFeed.scrollTop = els.chatFeed.scrollHeight;
    els.jumpLatest.hidden = true;
}

function resetChatFeed() {
    els.chatFeed.replaceChildren(els.chatEmpty);
    els.chatEmpty.hidden = false;
    els.jumpLatest.hidden = true;
}

function renderChatEvent(event) {
    const shouldStick = isFeedNearBottom();
    els.chatEmpty.hidden = true;

    const entryClass = `chat-entry chat-entry--${event.kind} chat-entry--${event.role}`;
    const entry = createElement("div", entryClass);
    const avatar = createElement("div", "chat-avatar");
    avatar.appendChild(icon(event.avatar));
    entry.appendChild(avatar);

    const body = createElement("div", "chat-body");
    const header = createElement("div", "chat-header");
    header.appendChild(createElement("span", "chat-speaker", event.speaker));
    header.appendChild(
        createElement("span", "chat-kind", CHAT_KIND_LABELS[event.kind] || event.kind.replace(/_/g, " ")),
    );
    body.appendChild(header);
    body.appendChild(createElement("div", "chat-meta", `Turn ${event.turn} / ${event.phase}`));

    const hasVoteRows = event.vote_summary_rows && event.vote_summary_rows.length > 0;
    if (hasVoteRows) {
        body.appendChild(voteTable(event.vote_summary_rows));
        if (event.vote_result_text) {
            body.appendChild(createElement("div", "chat-result", `Result: ${event.vote_result_text}`));
        }
    } else {
        body.appendChild(createElement("div", "chat-text", event.message));
    }

    entry.appendChild(body);
    els.chatFeed.appendChild(entry);
    if (shouldStick) {
        scrollFeedToBottom();
    } else {
        els.jumpLatest.hidden = false;
    }
}

// ---------- SSE event handling ----------

function applyState(state) {
    renderCycle(state.current);
    renderPlayers(state.players_status);
    renderSpeechQueue(state.speech_queue);
}

function applyRetryEvent(event) {
    els.retriesSection.hidden = false;
    els.retriesList.replaceChildren();
    const log = (event.log && event.log.length) ? event.log : [event.message];
    for (const line of log) {
        els.retriesList.appendChild(createElement("li", null, line));
    }
}

function applyResult(result) {
    els.artifacts.hidden = false;
    els.artifactLog.textContent = `Logs: ${result.output_dir}`;
    els.artifactLog.title = result.output_dir;
    els.artifactEvents.textContent = `Events: ${result.events_path}`;
    els.artifactSummary.textContent = `Summary: ${result.summary_path}`;
    renderPlayers(result.players_status);
    renderCycle(result.current);
}

function markFinished(kind, message) {
    const isError = kind === "match_error";
    setProgress(100, isError ? "Match ended with error" : "Match complete");
    setStatus(
        message ||
            (isError
                ? "Stopped due to an error."
                : null),
        isError,
    );
    setRunning(false);
    setRunState(isError ? "error" : "complete");
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
}

function handleEvent(event) {
    switch (event.type) {
        case "progress":
            if (event.percent > MAX_PROGRESS_PERCENT) {
                break;
            }
            setProgress(event.percent, event.text);
            break;
        case "retry":
            applyRetryEvent(event);
            break;
        case "chat":
            renderChatEvent(event);
            break;
        case "state":
            applyState(event);
            break;
        case "match_complete":
            applyResult(event);
            markFinished("match_complete");
            break;
        case "match_error":
            markFinished("match_error", `Error: ${event.message}`);
            break;
        default:
            break;
    }
}

// ---------- networking ----------

function startListening() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource("/api/match/stream");
    eventSource.onmessage = (message) => {
        try {
            handleEvent(JSON.parse(message.data));
        } catch (error) {
            console.error("Failed to parse SSE payload", error);
        }
    };
}

async function applySnapshot(snapshot, connect) {
    if (snapshot.view) {
        applyState(snapshot.view);
    }
    if (snapshot.error_message) {
        setStatus(`Error: ${snapshot.error_message}`, true);
        setRunState("error");
    }
    if (snapshot.result) {
        applyResult(snapshot.result);
        setRunState("complete");
    }
    if (snapshot.status === "running") {
        setRunning(true);
        setProgress(0, "Running match");
        if (connect) startListening();
    } else {
        setRunning(false);
    }
}

async function loadState() {
    try {
        const response = await fetch("/api/match/state");
        if (!response.ok) {
            console.error("Failed to load match state", response.status);
            return;
        }
        const snapshot = await response.json();
        await applySnapshot(snapshot, snapshot.status === "running");
    } catch (error) {
        console.error("Failed to load match state", error);
    }
}

async function runMatch() {
    const configPath = els.configPath.value.trim() || "config.yaml";
    const seedRaw = els.seed.value.trim();
    const maxRoundsRaw = Number.parseInt(els.maxRounds.value, 10);

    setRunning(true);
    setStatus(null);
    resetChatFeed();
    els.retriesSection.hidden = true;
    els.artifacts.hidden = true;
    setProgress(0, "Preparing match");

    try {
        const response = await fetch("/api/match/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                config_path: configPath,
                seed: seedRaw === "" ? null : Number.parseInt(seedRaw, 10),
                max_rounds: Number.isNaN(maxRoundsRaw) ? 10 : maxRoundsRaw,
                runner: els.runner.value,
            }),
        });
        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                const payload = await response.json();
                if (payload && payload.detail) detail = payload.detail;
            } catch (error) {
                // keep default detail
            }
            setRunning(false);
            setRunState("error");
            setProgress(100, "Match not started");
            setStatus(`Error: ${detail}`, true);
            return;
        }
        const snapshot = await response.json();
        await applySnapshot(snapshot, true);
    } catch (error) {
        setRunning(false);
        setRunState("error");
        setProgress(100, "Match not started");
        setStatus(`Error: ${error.message}`, true);
    }
}

els.runButton.addEventListener("click", runMatch);
els.jumpLatest.addEventListener("click", scrollFeedToBottom);
els.chatFeed.addEventListener("scroll", () => {
    if (isFeedNearBottom()) els.jumpLatest.hidden = true;
});
hydrateIcons();
loadState();
