// The handful of values more than one feature both reads and writes.
//
// In here: state with no single owner -- the conversation, the composer's
// attachments, which engines /health says are up. Reads import the binding
// directly (ES module bindings are live); writes go through the setter, since
// an imported binding cannot be assigned to.
// Not in here: anything one module owns. That state stays in that module and
// is exported from there -- see chat/generation.js, voice/capture.js,
// util/engine.js. Adding to this file should feel like a small defeat.

// --- State ---
export let chatHistory = [];
export let attachedImage = null;   // base64 data URI
export let attachedDoc = null;     // {name, text, chars} — already extracted to text
export let thinkExpanded = false;  // think-block expand state across re-renders

export function setChatHistory(next) { chatHistory = next; }
export function setAttachedImage(next) { attachedImage = next; }
export function setAttachedDoc(next) { attachedDoc = next; }
export function setThinkExpanded(next) { thinkExpanded = next; }
// What the chat request should ask for: the id of the model actually resident,
// e.g. "qwen3-8b-int4-cw@NPU". Kept separate from the dropdown now that the
// dropdown lists models on *disk* — the two answer different questions, and
// conflating them would send requests naming a model that isn't loaded yet.
export let loadedModelId = '';

export function setLoadedModelId(next) { loadedModelId = next; }

// The most recent /health payload. Several features answer questions from it
// (which model is loaded, can this device call tools) and none of them should
// be issuing their own poll to find out.
export let lastHealthData = {};

export function setLastHealthData(next) { lastHealthData = next; }

// Which audio models the server actually has loaded. A control wired to a
// missing model is worse than no control, so these gate the UI rather than
// letting the user find out by getting a 503.
export let asrReady = false;
export let ttsReady = false;
export let vadReady = false;         // is a real VAD model loaded server-side

export function setAsrReady(next) { asrReady = next; }
export function setTtsReady(next) { ttsReady = next; }
export function setVadReady(next) { vadReady = next; }

// Push-to-talk is held in two places -- the chat tab's mic button and the
// voice tab's -- and the VAD socket has to know, because the server must stand
// its endpointing down while a human is holding the button.
export let pttActive = false;

export function setPttActive(next) { pttActive = next; }
