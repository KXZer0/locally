// locally Web UI — entry point.
//
// Device provenance is still on every reply, but it is carried by the engine's
// NAME in the mono face and by the fill of its dot -- solid NPU, ring GPU,
// hollow CPU -- rather than by a colour. The palette is monochrome by design;
// see static/css/style.css for the token system.
//
// This file does two things and nothing else: it pulls in every module (which
// is what registers their event listeners), and it starts the app. The import
// order below follows the order these features appeared in the old single
// app.js, so a listener registered before another one still is.
//
// If you are looking for behaviour, it is not here. Each module's header says
// what belongs in it and what does not:
//
//   core/       storage migration, element handles, shared state, formatting
//   markdown/   the escape-first renderer and the streaming painter
//   chat/       prompts, context accounting, the thread, sending, web search
//   system/     bootstrap and /health, the device rail, model swap, memory HUD
//   util/       the Tools tab: engines, read, images, search
//   audio/      WAV plumbing and the chat tab's microphone
//   voice/      the voice tab: state machine, capture, VAD, speech, a turn
//   ui/         tabs, composer, settings, the Code tab
//   palette.js  the Ctrl+K command palette

import './core/storage.js';
import './core/dom.js';
import './core/state.js';
import './core/format.js';
import './core/paint.js';

import './markdown/think-split.js';
import './markdown/render.js';
import './markdown/stream-painter.js';

import './chat/prompt.js';
import './chat/websearch.js';
import './chat/thread.js';
import './chat/send.js';
import './chat/sources-panel.js';
import './chat/context.js';
import './chat/generation.js';
import './chat/completion.js';
import './chat/attachments.js';

import { init } from './system/bootstrap.js';
import './system/devices.js';
import './system/swap.js';
import './system/memory-hud.js';

import './util/engine.js';
import './util/read.js';
import './util/images.js';
import './util/search.js';

import './audio/wav.js';
import './audio/asr.js';

import './ui/tabs.js';
import './voice/speaker.js';
import './voice/state.js';
import './voice/think-stream.js';
import './voice/capture.js';
import './voice/vad-socket.js';
import './voice/turn.js';
import './voice/speech-queue.js';

import './ui/code-tab.js';
import './ui/odysseus.js';
import './ui/settings.js';
import './ui/composer.js';
import './ui/rail-resize.js';

import './palette.js';

// --- Start ---
init();
