const TEXT = {
  title: '\u6c5f\u9634\u9a6c\u8e44\u9165\u6570\u5b57\u4eba\u8bed\u97f3\u52a9\u624b',
  subtitle: '\u6d4f\u89c8\u5668\u7aef\u5b9e\u65f6\u8bed\u97f3\u5bf9\u8bdd\uff0c\u5df2\u63a5\u5165 ASR + LLM + \u6d41\u5f0f TTS \u540e\u7aef\u3002',
  eyebrow: '\u5b9e\u65f6 WebSocket \u8bed\u97f3\u4ea4\u4e92',
  toggle_console_hide: '\u9690\u85cf\u53f3\u4fa7\u63a7\u5236\u53f0',
  toggle_console_show: '\u663e\u793a\u53f3\u4fa7\u63a7\u5236\u53f0',
  hint_mic_label: '\u5f55\u97f3 / VAD',
  hint_tts_label: '\u64ad\u653e / \u53e3\u578b',
  hint_vad_label: '\u622a\u6b62\u53c2\u6570',
  hint_backend_label: '\u540e\u7aef\u6a21\u5f0f',
  chat_title: '\u5bf9\u8bdd\u5185\u5bb9',
  chat_meta: '\u5de6\u4fa7\u4e3a\u6570\u5b57\u4eba\uff0c\u53f3\u4fa7\u4e3a\u5b9e\u65f6\u63a7\u5236\u53f0\u3002',
  clear_chat: '\u6e05\u7a7a\u5bf9\u8bdd',
  transcript_title: '\u5b9e\u65f6\u8bc6\u522b',
  input_label: '\u8f93\u5165\u6587\u672c\uff08\u53ef\u76f4\u63a5\u53d1\u9001\uff0c\u4e5f\u53ef\u9884\u89c8\u6ce8\u97f3\u6b63\u5e38\u5316\u7ed3\u679c\uff09',
  input_placeholder: '\u8f93\u5165\u4e00\u53e5\u8bdd\uff0c\u6216\u70b9\u51fb\u201c\u5f00\u59cb\u5b9e\u65f6\u5f55\u97f3\u201d\u76f4\u63a5\u5bf9\u8bdd...',
  send_text: '\u53d1\u9001\u6587\u672c',
  start_mic: '\u5f00\u59cb\u5b9e\u65f6\u5f55\u97f3',
  stop_mic: '\u7ed3\u675f\u5f55\u97f3',
  preview_normalize: '\u9884\u89c8\u6ce8\u97f3\u89c4\u5219',
  interrupt: '\u6253\u65ad\u56de\u590d',
  metric_ttfa: 'TTFA',
  metric_llm: 'LLM',
  metric_rtf: 'RTF',
  metric_tts: 'TTS',
  console_title: 'Realtime Console',
  console_subtitle: '\u9ed8\u8ba4\u9690\u85cf\u53c2\u6570\uff0c\u4ec5\u5728\u9700\u8981\u65f6\u5c55\u5f00\u3002',
  reconnect: '\u91cd\u8fde\u540e\u7aef',
  console_runtime: '\u8fd0\u884c\u72b6\u6001',
  console_normalized: '\u6b63\u5e38\u5316 / \u6ce8\u97f3\u9884\u89c8',
  advanced_params: '\u9ad8\u7ea7\u53c2\u6570',
  field_language: '\u8bed\u8a00',
  field_chunk: 'TTS chunk_size',
  field_preroll: 'pre_roll_sec',
  field_min_utt: 'min_utterance_sec',
  field_silence: 'silence_sec',
  transcript_empty: '\u7b49\u5f85\u8bf4\u8bdd...',
  runtime_idle: '\u5f85\u547d',
  runtime_connecting: '\u6b63\u5728\u8fde\u63a5\u540e\u7aef...',
  runtime_ready: '\u540e\u7aef\u5df2\u5c31\u7eea',
  runtime_listening: '\u6b63\u5728\u76d1\u542c',
  runtime_speech: '\u68c0\u6d4b\u5230\u8bf4\u8bdd',
  runtime_thinking: '\u6b63\u5728\u601d\u8003',
  runtime_speaking: '\u6b63\u5728\u64ad\u62a5',
  runtime_error: '\u53d1\u751f\u9519\u8bef',
  playback_speaking: '\u6d41\u5f0f\u64ad\u653e\u4e2d',
  playback_idle: '\u6682\u65e0\u8bed\u97f3',
  log_user: '\u7528\u6237',
  log_assistant: '\u6570\u5b57\u4eba',
  log_system: '\u7cfb\u7edf',
  msg_connected: '\u5df2\u8fde\u63a5 WebSocket \u540e\u7aef',
  msg_disconnected: '\u540e\u7aef\u8fde\u63a5\u5df2\u65ad\u5f00',
  msg_sent_text: '\u5df2\u53d1\u9001\u6587\u672c\u8bf7\u6c42',
  msg_interrupt: '\u5df2\u53d1\u9001\u6253\u65ad\u6307\u4ee4',
  msg_mic_started: '\u5b9e\u65f6\u5f55\u97f3\u5df2\u5f00\u542f',
  msg_mic_stopped: '\u5f55\u97f3\u7ed3\u675f\uff0c\u7b49\u5f85\u6700\u540e\u4e00\u6bb5\u8bc6\u522b',
  msg_clear: '\u5bf9\u8bdd\u5df2\u6e05\u7a7a',
  msg_normalized: '\u5df2\u5237\u65b0\u6ce8\u97f3\u9884\u89c8',
  msg_reconnect: '\u6b63\u5728\u91cd\u65b0\u8fde\u63a5\u540e\u7aef',
  msg_error_prefix: '\u9519\u8bef',
  ready_prefix: '\u540e\u7aef',
  listening_armed: '\u5df2\u5f00\u542f\u8bed\u97f3\u68c0\u6d4b\uff0c\u53ef\u4ee5\u5f00\u53e3\u8bf4\u8bdd',
  listening_speech_start: '\u68c0\u6d4b\u5230\u8bf4\u8bdd\uff0c\u6b63\u5728\u5b9e\u65f6\u4e0a\u4f20',
  listening_speech_end: '\u68c0\u6d4b\u5230\u53e5\u5c3e\u9759\u97f3\uff0c\u5f00\u59cb\u8bc6\u522b',
  listening_idle: '\u5f85\u547d',
  backend_unknown: '\u672a\u77e5',
  metric_done: '\u672c\u8f6e\u6307\u6807\u5df2\u66f4\u65b0',
  no_normalized: '\u9884\u89c8\u533a\u57df\u6682\u65e0\u5185\u5bb9',
  connection_connected: '\u8fde\u63a5\u6b63\u5e38',
  connection_disconnected: '\u672a\u8fde\u63a5'
};

const state = {
  ws: null,
  wsReady: false,
  connecting: false,
  micActive: false,
  assistantBuffers: new Map(),
  assistantNodes: new Map(),
  audioContext: null,
  audioGain: null,
  nextPlaybackTime: 0,
  playbackActive: false,
  lipEvents: [],
  mouthEnergy: 0,
  mediaStream: null,
  mediaSource: null,
  processor: null,
  captureContext: null,
  captureSink: null,
  flushTimer: null,
  logLimit: 120
};

const el = {
  toggleConsoleBtn: document.getElementById('toggleConsoleBtn'),
  assistantStateText: document.getElementById('assistantStateText'),
  connectionText: document.getElementById('connectionText'),
  listeningStateText: document.getElementById('listeningStateText'),
  playbackStateText: document.getElementById('playbackStateText'),
  vadSummaryText: document.getElementById('vadSummaryText'),
  backendSummaryText: document.getElementById('backendSummaryText'),
  chatLog: document.getElementById('chatLog'),
  transcriptBox: document.getElementById('transcriptBox'),
  textInput: document.getElementById('textInput'),
  sendTextBtn: document.getElementById('sendTextBtn'),
  toggleMicBtn: document.getElementById('toggleMicBtn'),
  previewBtn: document.getElementById('previewBtn'),
  interruptBtn: document.getElementById('interruptBtn'),
  ttfaStat: document.getElementById('ttfaStat'),
  llmStat: document.getElementById('llmStat'),
  rtfStat: document.getElementById('rtfStat'),
  ttsStat: document.getElementById('ttsStat'),
  reconnectBtn: document.getElementById('reconnectBtn'),
  runtimeStatus: document.getElementById('runtimeStatus'),
  normalizedPreview: document.getElementById('normalizedPreview'),
  consoleLog: document.getElementById('consoleLog'),
  clearChatBtn: document.getElementById('clearChatBtn'),
  languageSelect: document.getElementById('languageSelect'),
  chunkSizeInput: document.getElementById('chunkSizeInput'),
  preRollInput: document.getElementById('preRollInput'),
  minUtteranceInput: document.getElementById('minUtteranceInput'),
  silenceInput: document.getElementById('silenceInput'),
  mouthInner: document.getElementById('mouthInner'),
  smileLine: document.getElementById('smileLine'),
  eyeLeft: document.getElementById('eyeLeft'),
  eyeRight: document.getElementById('eyeRight'),
  faceGroup: document.getElementById('faceGroup')
};

function applyI18n() {
  document.title = TEXT.title;
  document.querySelectorAll('[data-i18n]').forEach((node) => {
    const key = node.getAttribute('data-i18n');
    if (Object.prototype.hasOwnProperty.call(TEXT, key)) node.textContent = TEXT[key];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((node) => {
    const key = node.getAttribute('data-i18n-placeholder');
    if (Object.prototype.hasOwnProperty.call(TEXT, key)) node.setAttribute('placeholder', TEXT[key]);
  });
}

function setBodyState(name) {
  document.body.classList.remove('state-idle', 'state-ready', 'state-listening', 'state-thinking', 'state-speaking', 'state-error');
  document.body.classList.add('state-' + name);
}

function setAssistantState(textKey, stateName) {
  el.assistantStateText.textContent = TEXT[textKey] || textKey;
  setBodyState(stateName);
}

function updateConnectionStatus() {
  if (state.connecting) {
    el.connectionText.textContent = TEXT.runtime_connecting;
    return;
  }
  el.connectionText.textContent = state.wsReady ? TEXT.connection_connected : TEXT.connection_disconnected;
}

function updateMicButton() { el.toggleMicBtn.textContent = state.micActive ? TEXT.stop_mic : TEXT.start_mic; }

function updateVadSummary() {
  const minUtt = Number(el.minUtteranceInput.value || 0).toFixed(2);
  const silence = Number(el.silenceInput.value || 0).toFixed(2);
  const preRoll = Number(el.preRollInput.value || 0).toFixed(2);
  el.vadSummaryText.textContent = 'min ' + minUtt + 's / end ' + silence + 's / pre ' + preRoll + 's';
}

function updateRuntimeCard(extra) {
  const lines = [
    'ws_ready: ' + String(state.wsReady),
    'mic_active: ' + String(state.micActive),
    'playback_active: ' + String(state.playbackActive),
    'language: ' + el.languageSelect.value,
    'chunk_size: ' + Number(el.chunkSizeInput.value || 8),
    'pre_roll_sec: ' + Number(el.preRollInput.value || 0.2).toFixed(2),
    'min_utterance_sec: ' + Number(el.minUtteranceInput.value || 0.35).toFixed(2),
    'silence_sec: ' + Number(el.silenceInput.value || 0.45).toFixed(2)
  ];
  if (extra) lines.push(extra);
  el.runtimeStatus.textContent = lines.join('\n');
}

function addConsoleLog(text) {
  const row = document.createElement('div');
  row.className = 'log-row';
  row.textContent = '[' + new Date().toLocaleTimeString() + '] ' + text;
  el.consoleLog.prepend(row);
  while (el.consoleLog.children.length > state.logLimit) el.consoleLog.removeChild(el.consoleLog.lastElementChild);
}

function scrollChatToBottom() { el.chatLog.scrollTop = el.chatLog.scrollHeight; }

function addMessage(role, text, turnId) {
  const wrap = document.createElement('div');
  wrap.className = 'message ' + role;
  const roleNode = document.createElement('div');
  roleNode.className = 'message-role';
  roleNode.textContent = role === 'user' ? TEXT.log_user : role === 'assistant' ? TEXT.log_assistant : TEXT.log_system;
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text || '';
  wrap.appendChild(roleNode);
  wrap.appendChild(body);
  el.chatLog.appendChild(wrap);
  scrollChatToBottom();
  if (role === 'assistant' && turnId != null) state.assistantNodes.set(String(turnId), body);
  return body;
}

function clearConversation() {
  el.chatLog.innerHTML = '';
  el.transcriptBox.textContent = TEXT.transcript_empty;
  state.assistantBuffers.clear();
  state.assistantNodes.clear();
  el.normalizedPreview.textContent = TEXT.no_normalized;
  el.ttfaStat.textContent = '--';
  el.llmStat.textContent = '--';
  el.rtfStat.textContent = '--';
  el.ttsStat.textContent = '--';
}

function ensureAssistantNode(turnId) {
  const key = String(turnId);
  let node = state.assistantNodes.get(key);
  if (!node) node = addMessage('assistant', '', turnId);
  return node;
}

function flushAssistantBuffers() {
  state.assistantBuffers.forEach((delta, turnId) => {
    if (!delta) return;
    const node = ensureAssistantNode(turnId);
    node.textContent += delta;
    state.assistantBuffers.set(turnId, '');
  });
  scrollChatToBottom();
}

function formatMetric(value, suffix, digits) {
  if (value == null || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(digits) + suffix;
}

function updateMetrics(metrics) {
  el.ttfaStat.textContent = metrics && metrics.ttfa_ms != null ? String(metrics.ttfa_ms) + ' ms' : '--';
  el.llmStat.textContent = metrics && metrics.llm_first_token_ms != null ? String(metrics.llm_first_token_ms) + ' / ' + String(metrics.llm_done_ms || '--') + ' ms' : '--';
  el.rtfStat.textContent = metrics && metrics.rtf != null ? Number(metrics.rtf).toFixed(3) : '--';
  el.ttsStat.textContent = metrics ? formatMetric(metrics.tts_synth_s, ' s', 2) + ' / ' + formatMetric(metrics.tts_audio_s, ' s', 2) : '--';
}

function base64ToBytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function bytesToBase64(bytes) {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

function float32ToPcm16Bytes(float32Array) {
  const samples = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    samples[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return new Uint8Array(samples.buffer);
}

function pcm16Base64ToFloat32(b64) {
  const bytes = base64ToBytes(b64);
  const out = new Float32Array(Math.floor(bytes.length / 2));
  for (let i = 0, j = 0; i + 1 < bytes.length; i += 2, j += 1) {
    let value = bytes[i] | (bytes[i + 1] << 8);
    if (value & 0x8000) value -= 0x10000;
    out[j] = value / 32768;
  }
  return out;
}

function computeEnergy(samples) {
  if (!samples || !samples.length) return 0;
  let acc = 0;
  const step = Math.max(1, Math.floor(samples.length / 1200));
  let count = 0;
  for (let i = 0; i < samples.length; i += step) {
    const v = samples[i];
    acc += v * v;
    count += 1;
  }
  return Math.sqrt(acc / Math.max(1, count));
}

async function ensurePlaybackContext() {
  if (!state.audioContext) {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    state.audioGain = state.audioContext.createGain();
    state.audioGain.gain.value = 1.0;
    state.audioGain.connect(state.audioContext.destination);
  }
  if (state.audioContext.state === 'suspended') await state.audioContext.resume();
  return state.audioContext;
}

async function enqueuePcmAudio(b64, sampleRate) {
  const ctx = await ensurePlaybackContext();
  const pcm = pcm16Base64ToFloat32(b64);
  if (!pcm.length) return;
  const buffer = ctx.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioGain);
  const now = ctx.currentTime;
  const when = Math.max(now + 0.015, state.nextPlaybackTime || 0);
  source.start(when);
  state.nextPlaybackTime = when + buffer.duration;
  state.playbackActive = true;
  el.playbackStateText.textContent = TEXT.playback_speaking;
  state.lipEvents.push({ start: when, end: when + buffer.duration, energy: Math.min(1, computeEnergy(pcm) * 5.6) });
  source.onended = () => {
    const current = state.audioContext ? state.audioContext.currentTime : 0;
    state.lipEvents = state.lipEvents.filter((item) => item.end > current - 0.15);
    if (state.audioContext && state.nextPlaybackTime - state.audioContext.currentTime < 0.03) {
      state.playbackActive = false;
      el.playbackStateText.textContent = TEXT.playback_idle;
      if (document.body.classList.contains('state-speaking')) setAssistantState('runtime_ready', 'ready');
    }
  };
}

function animateAvatar() {
  const audioNow = state.audioContext ? state.audioContext.currentTime : 0;
  state.lipEvents = state.lipEvents.filter((item) => item.end > audioNow - 0.18);
  let target = 0;
  for (const item of state.lipEvents) {
    if (audioNow >= item.start && audioNow <= item.end) target = Math.max(target, item.energy);
    else if (item.start > audioNow && item.start - audioNow < 0.04) target = Math.max(target, item.energy * 0.35);
  }
  const smoothing = target > state.mouthEnergy ? 0.08 : 0.05;
  state.mouthEnergy += (target - state.mouthEnergy) * smoothing;
  const openY = 0.58 + state.mouthEnergy * 1.05;
  const openX = 0.88 + state.mouthEnergy * 0.15;
  const faceBob = Math.sin(performance.now() / 850) * 1.2;
  const eyeScale = 1 - Math.min(0.18, state.mouthEnergy * 0.12);
  el.mouthInner.setAttribute('transform', 'translate(0 ' + (state.mouthEnergy * 2.6).toFixed(2) + ') scale(' + openX.toFixed(3) + ' ' + openY.toFixed(3) + ')');
  el.smileLine.setAttribute('transform', 'translate(0 ' + (state.mouthEnergy * 1.5).toFixed(2) + ')');
  el.faceGroup.setAttribute('transform', 'translate(0 ' + faceBob.toFixed(2) + ')');
  el.eyeLeft.setAttribute('transform', 'scale(1 ' + eyeScale.toFixed(3) + ')');
  el.eyeRight.setAttribute('transform', 'scale(1 ' + eyeScale.toFixed(3) + ')');
  requestAnimationFrame(animateAvatar);
}

function socketUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return proto + '//' + window.location.host + '/ws/assistant';
}

function safeJsonSend(payload) {
  if (!state.wsReady || !state.ws) {
    addConsoleLog(TEXT.msg_error_prefix + ': websocket not ready');
    return false;
  }
  state.ws.send(JSON.stringify(payload));
  return true;
}

function connectWebSocket() {
  if (state.ws) {
    try { state.ws.close(); } catch (err) { console.warn(err); }
  }
  state.connecting = true;
  state.wsReady = false;
  updateConnectionStatus();
  updateRuntimeCard('ws_url: ' + socketUrl());
  setAssistantState('runtime_connecting', 'idle');
  const ws = new WebSocket(socketUrl());
  state.ws = ws;
  ws.onopen = () => {
    state.connecting = false;
    state.wsReady = true;
    updateConnectionStatus();
    addConsoleLog(TEXT.msg_connected);
    setAssistantState('runtime_ready', 'ready');
  };
  ws.onclose = () => {
    state.connecting = false;
    state.wsReady = false;
    updateConnectionStatus();
    addConsoleLog(TEXT.msg_disconnected);
    if (state.micActive) void stopMicStreaming();
    else setAssistantState('runtime_idle', 'idle');
  };
  ws.onerror = () => {
    state.connecting = false;
    state.wsReady = false;
    updateConnectionStatus();
    setAssistantState('runtime_error', 'error');
  };
  ws.onmessage = async (event) => { await handleServerEvent(JSON.parse(event.data)); };
}

async function handleServerEvent(msg) {
  if (!msg || !msg.type) return;
  if (msg.type === 'ready') {
    const status = msg.status || {};
    const backend = [status.language || TEXT.backend_unknown, status.rag_backend || 'none'];
    if (status.rag_collection) backend.push(status.rag_collection);
    el.backendSummaryText.textContent = backend.join(' / ');
    addConsoleLog(TEXT.ready_prefix + ': ' + (msg.message || 'ready'));
    updateRuntimeCard('backend: ' + backend.join(' | '));
    return;
  }
  if (msg.type === 'listening') {
    if (msg.silence_sec != null) el.silenceInput.value = Number(msg.silence_sec).toFixed(2);
    if (msg.min_utterance_sec != null) el.minUtteranceInput.value = Number(msg.min_utterance_sec).toFixed(2);
    updateVadSummary();
    if (msg.state === 'armed') {
      el.listeningStateText.textContent = TEXT.listening_armed;
      setAssistantState('runtime_listening', 'listening');
    } else if (msg.state === 'speech_start') {
      el.listeningStateText.textContent = TEXT.listening_speech_start;
      setAssistantState('runtime_speech', 'listening');
    } else if (msg.state === 'speech_end') {
      el.listeningStateText.textContent = TEXT.listening_speech_end;
      setAssistantState('runtime_thinking', 'thinking');
    } else if (msg.state === 'idle') {
      el.listeningStateText.textContent = TEXT.listening_idle;
      if (!state.playbackActive) setAssistantState('runtime_ready', 'ready');
    }
    return;
  }
  if (msg.type === 'transcript') {
    el.transcriptBox.textContent = msg.text ? msg.text : TEXT.transcript_empty;
    return;
  }
  if (msg.type === 'user_text') {
    addMessage('user', msg.text || '', msg.turn_id);
    setAssistantState('runtime_thinking', 'thinking');
    return;
  }
  if (msg.type === 'assistant_text') {
    const key = String(msg.turn_id || 'default');
    ensureAssistantNode(key);
    state.assistantBuffers.set(key, (state.assistantBuffers.get(key) || '') + String(msg.delta || ''));
    setAssistantState('runtime_thinking', 'thinking');
    return;
  }
  if (msg.type === 'audio_chunk') {
    if (msg.audio_pcm16_b64) {
      await enqueuePcmAudio(msg.audio_pcm16_b64, Number(msg.sample_rate || 24000));
      setAssistantState('runtime_speaking', 'speaking');
    }
    return;
  }
  if (msg.type === 'done') {
    flushAssistantBuffers();
    updateMetrics(msg.metrics || null);
    if (!state.playbackActive) setAssistantState('runtime_ready', 'ready');
    return;
  }
  if (msg.type === 'interrupted') {
    flushAssistantBuffers();
    addConsoleLog(TEXT.msg_interrupt);
    setAssistantState('runtime_ready', 'ready');
    return;
  }
  if (msg.type === 'normalized') {
    el.normalizedPreview.textContent = msg.normalized || TEXT.no_normalized;
    addConsoleLog(TEXT.msg_normalized);
    return;
  }
  if (msg.type === 'error') {
    addConsoleLog(TEXT.msg_error_prefix + ': ' + String(msg.message || 'unknown'));
    addMessage('system', TEXT.msg_error_prefix + ': ' + String(msg.message || 'unknown'));
    setAssistantState('runtime_error', 'error');
    return;
  }
  if (msg.type === 'request_done' && !state.playbackActive && !state.micActive) setAssistantState('runtime_ready', 'ready');
}

async function sendTextRequest() {
  const text = el.textInput.value.trim();
  if (!text) return;
  await ensurePlaybackContext();
  if (!safeJsonSend({
    type: 'text',
    text: text,
    language: el.languageSelect.value,
    chunk_size: Math.max(1, Number(el.chunkSizeInput.value || 8))
  })) return;
  addConsoleLog(TEXT.msg_sent_text);
  el.transcriptBox.textContent = TEXT.transcript_empty;
  setAssistantState('runtime_thinking', 'thinking');
}

function previewNormalize() {
  const text = el.textInput.value.trim();
  if (!text) {
    el.normalizedPreview.textContent = TEXT.no_normalized;
    return;
  }
  safeJsonSend({ type: 'normalize', text: text });
}

async function startMicStreaming() {
  if (state.micActive) return;
  if (!state.wsReady) {
    addConsoleLog(TEXT.msg_error_prefix + ': websocket not ready');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    addMessage('system', 'Browser does not support microphone capture.');
    return;
  }
  await ensurePlaybackContext();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  const captureContext = new (window.AudioContext || window.webkitAudioContext)();
  if (captureContext.state === 'suspended') await captureContext.resume();
  const source = captureContext.createMediaStreamSource(stream);
  const processor = captureContext.createScriptProcessor(4096, 1, 1);
  const sink = captureContext.createGain();
  sink.gain.value = 0;
  processor.onaudioprocess = (ev) => {
    if (!state.wsReady || !state.micActive) return;
    const input = ev.inputBuffer.getChannelData(0);
    safeJsonSend({
      type: 'audio_chunk',
      sample_rate: captureContext.sampleRate,
      audio_chunk_b64: bytesToBase64(float32ToPcm16Bytes(input))
    });
  };
  source.connect(processor);
  processor.connect(sink);
  sink.connect(captureContext.destination);
  state.mediaStream = stream;
  state.mediaSource = source;
  state.processor = processor;
  state.captureContext = captureContext;
  state.captureSink = sink;
  state.micActive = true;
  updateMicButton();
  const started = safeJsonSend({
    type: 'audio_start',
    language: el.languageSelect.value,
    chunk_size: Math.max(1, Number(el.chunkSizeInput.value || 8)),
    pre_roll_sec: Math.max(0.05, Number(el.preRollInput.value || 0.2)),
    min_utterance_sec: Math.max(0.1, Number(el.minUtteranceInput.value || 0.35)),
    silence_sec: Math.max(0.1, Number(el.silenceInput.value || 0.45))
  });
  if (!started) {
    await stopMicStreaming();
    return;
  }
  el.listeningStateText.textContent = TEXT.listening_armed;
  addConsoleLog(TEXT.msg_mic_started);
  updateRuntimeCard();
  setAssistantState('runtime_listening', 'listening');
}

async function stopMicStreaming() {
  if (!state.micActive) return;
  state.micActive = false;
  updateMicButton();
  if (state.processor) {
    state.processor.disconnect();
    state.processor.onaudioprocess = null;
  }
  if (state.mediaSource) state.mediaSource.disconnect();
  if (state.captureSink) state.captureSink.disconnect();
  if (state.mediaStream) state.mediaStream.getTracks().forEach((track) => track.stop());
  if (state.captureContext) {
    try { await state.captureContext.close(); } catch (err) { console.warn(err); }
  }
  state.processor = null;
  state.mediaSource = null;
  state.mediaStream = null;
  state.captureContext = null;
  state.captureSink = null;
  safeJsonSend({ type: 'audio_end' });
  addConsoleLog(TEXT.msg_mic_stopped);
  updateRuntimeCard();
}

function toggleConsole() {
  const hidden = document.body.classList.toggle('console-hidden');
  el.toggleConsoleBtn.textContent = hidden ? TEXT.toggle_console_show : TEXT.toggle_console_hide;
}

function bindEvents() {
  el.toggleConsoleBtn.addEventListener('click', toggleConsole);
  el.sendTextBtn.addEventListener('click', sendTextRequest);
  el.previewBtn.addEventListener('click', previewNormalize);
  el.reconnectBtn.addEventListener('click', () => { addConsoleLog(TEXT.msg_reconnect); connectWebSocket(); });
  el.interruptBtn.addEventListener('click', () => { if (safeJsonSend({ type: 'interrupt' })) addConsoleLog(TEXT.msg_interrupt); });
  el.clearChatBtn.addEventListener('click', () => { clearConversation(); addConsoleLog(TEXT.msg_clear); });
  el.toggleMicBtn.addEventListener('click', async () => { if (state.micActive) await stopMicStreaming(); else await startMicStreaming(); });
  [el.chunkSizeInput, el.preRollInput, el.minUtteranceInput, el.silenceInput, el.languageSelect].forEach((node) => {
    node.addEventListener('change', () => { updateVadSummary(); updateRuntimeCard(); });
  });
  el.textInput.addEventListener('keydown', (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
      ev.preventDefault();
      sendTextRequest();
    }
  });
  window.addEventListener('beforeunload', () => { try { if (state.ws) state.ws.close(); } catch (err) { console.warn(err); } });
}

applyI18n();
clearConversation();
updateMicButton();
updateVadSummary();
el.playbackStateText.textContent = TEXT.playback_idle;
el.listeningStateText.textContent = TEXT.listening_idle;
el.backendSummaryText.textContent = TEXT.backend_unknown;
updateRuntimeCard();
updateConnectionStatus();
state.flushTimer = window.setInterval(flushAssistantBuffers, 50);
bindEvents();
connectWebSocket();
animateAvatar();
