// The four modes: Chat, Voice, Tools, Code.
//
// In here: setMode() and the tab controls. Switching modes must never drop the
// conversation -- Chat and Voice share one history -- and must never leave a
// microphone open or audio playing.
// Not in here: what any panel contains. Each tab's feature owns its own DOM
// and its own state between visits.

import { input, panelChat, panelCode, panelUtil, panelVoice, tabChat, tabCode, tabUtil, tabVoice, utilNavBtns, utilViews } from '../core/dom.js';
import { revealSurface } from '../core/paint.js';
import { asrReady, ttsReady } from '../core/state.js';
import { enterCodeTab } from './code-tab.js';
import { selectUtilTask, utilTask } from '../util/engine.js';
import { cancelSpeakerRecording, disarmSpeakerForget } from '../voice/speaker.js';
import { setVoiceState, stopVoiceSession, voiceSession, warmAudio } from '../voice/state.js';

export let mode = 'chat';
// --- Tabs ---

// Switching modes must not drop history: Chat and Voice share chatHistory,
// while Util keeps its selected file and result intact between visits.
export function setMode(next) {
    if (!['chat', 'voice', 'util', 'code'].includes(next)) return;
    mode = next;
    document.body.dataset.mode = next;
    const isChat = next === 'chat';
    const isVoice = next === 'voice';
    const isUtil = next === 'util';
    const isCode = next === 'code';
    tabChat.classList.toggle('active', isChat);
    tabVoice.classList.toggle('active', isVoice);
    tabUtil.classList.toggle('active', isUtil);
    tabCode.classList.toggle('active', isCode);
    tabChat.setAttribute('aria-selected', String(isChat));
    tabVoice.setAttribute('aria-selected', String(isVoice));
    tabUtil.setAttribute('aria-selected', String(isUtil));
    tabCode.setAttribute('aria-selected', String(isCode));
    tabChat.tabIndex = isChat ? 0 : -1;
    tabVoice.tabIndex = isVoice ? 0 : -1;
    tabUtil.tabIndex = isUtil ? 0 : -1;
    tabCode.tabIndex = isCode ? 0 : -1;
    if (isChat) revealSurface(panelChat); else panelChat.hidden = true;
    if (isVoice) revealSurface(panelVoice); else panelVoice.hidden = true;
    if (isUtil) revealSurface(panelUtil); else panelUtil.hidden = true;
    if (isCode) { revealSurface(panelCode); enterCodeTab(); }
    else panelCode.hidden = true;

    // Leaving voice mode must not leave the mic live or audio playing. The
    // enrolment panel keeps its own stream, so it has to be told separately.
    if (!isVoice && voiceSession) stopVoiceSession();
    if (!isVoice) { cancelSpeakerRecording(); disarmSpeakerForget(); }
    if (isVoice) warmAudio();
    if (isChat) {
        input.focus();
    } else if (isVoice && (!ttsReady || !asrReady)) {
        const missing = [];
        if (!asrReady) missing.push('speech-to-text (--whisper-dir)');
        if (!ttsReady) missing.push('text-to-speech (--tts-dir)');
        setVoiceState('idle', `Voice needs ${missing.join(' and ')}.`);
    } else if (isUtil) {
        const view = utilViews.find(el => el.dataset.utilView === utilTask);
        view?.querySelector('button, input, textarea')?.focus();
    }
}
tabChat.addEventListener('click', () => setMode('chat'));
tabVoice.addEventListener('click', () => setMode('voice'));
// The Tools tab opens whichever utility was last selected; direct sidebar
// shortcuts enter Tools and select their workspace in one action.
tabUtil.addEventListener('click', () => setMode('util'));
tabCode.addEventListener('click', () => setMode('code'));
for (const btn of utilNavBtns) {
    btn.addEventListener('click', () => {
        setMode('util');
        selectUtilTask(btn.dataset.utilTask);
    });
}

for (const action of document.querySelectorAll('[data-home-prompt], [data-home-tool], [data-home-mode]')) {
    action.addEventListener('click', () => {
        if (action.dataset.homeTool) {
            setMode('util');
            selectUtilTask(action.dataset.homeTool);
            return;
        }
        if (action.dataset.homeMode) {
            setMode(action.dataset.homeMode);
            return;
        }
        input.value = action.dataset.homePrompt || '';
        input.dispatchEvent(new Event('input'));
        input.focus();
    });
}

export const modeTabs = [tabChat, tabVoice, tabUtil];
for (const [index, tab] of modeTabs.entries()) {
    tab.addEventListener('keydown', (e) => {
        let next = null;
        if (e.key === 'ArrowRight') next = (index + 1) % modeTabs.length;
        if (e.key === 'ArrowLeft') next = (index - 1 + modeTabs.length) % modeTabs.length;
        if (e.key === 'Home') next = 0;
        if (e.key === 'End') next = modeTabs.length - 1;
        if (next === null) return;
        e.preventDefault();
        modeTabs[next].click();
        modeTabs[next].focus();
    });
}
