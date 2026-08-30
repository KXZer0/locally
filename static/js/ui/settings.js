// The settings panel.
//
// In here: the persisted system and voice prompts, the thinking switch, and
// opening and closing the panel. null means "never set" (use the default) and
// "" is a deliberate opt-out that must survive a reload -- do not collapse the
// two with `|| DEFAULT`.
// Not in here: what the prompts SAY (chat/prompt.js) or the model control,
// which lives in the composer because that is where it is used.

import { invalidateContextCount, scheduleExactContextCount, updateContextDisplay } from '../chat/context.js';
import { DEFAULT_SYSTEM_PROMPT, DEFAULT_VOICE_PROMPT, getSystemPrompt, getVoicePrompt, syncThinkSwitch } from '../chat/prompt.js';
import { input, modelSelect, noThinkCheckbox, resetSystemPromptBtn, resetVoicePromptBtn, settingsBtn, settingsClose, settingsPanel, systemPromptInput, utilViews, voicePromptInput, voiceToggle } from '../core/dom.js';
import { revealSurface } from '../core/paint.js';
import { mode } from './tabs.js';
import { utilTask } from '../util/engine.js';

// Settings — the system prompt persists across sessions.
systemPromptInput.value = getSystemPrompt();
systemPromptInput.addEventListener('input', () => {
    localStorage.setItem('locally-system-prompt', systemPromptInput.value);
    invalidateContextCount(true);
    updateContextDisplay();
    scheduleExactContextCount();
});
resetSystemPromptBtn.addEventListener('click', () => {
    systemPromptInput.value = DEFAULT_SYSTEM_PROMPT;
    localStorage.setItem('locally-system-prompt', DEFAULT_SYSTEM_PROMPT);
    invalidateContextCount(true);
    updateContextDisplay();
    scheduleExactContextCount();
    systemPromptInput.focus();
});

voicePromptInput.value = getVoicePrompt();
voicePromptInput.addEventListener('input', () => {
    localStorage.setItem('locally-voice-prompt', voicePromptInput.value);
});
resetVoicePromptBtn.addEventListener('click', () => {
    voicePromptInput.value = DEFAULT_VOICE_PROMPT;
    localStorage.setItem('locally-voice-prompt', DEFAULT_VOICE_PROMPT);
    voicePromptInput.focus();
});
settingsBtn.addEventListener('click', () => {
    const opening = settingsPanel.hidden;
    if (opening) revealSurface(settingsPanel);
    else settingsPanel.hidden = true;
    settingsBtn.setAttribute('aria-expanded', String(!settingsPanel.hidden));
    if (!settingsPanel.hidden) modelSelect.focus();
    else if (mode === 'chat') input.focus();
    else if (mode === 'util') {
        const view = utilViews.find(el => el.dataset.utilView === utilTask);
        view?.querySelector('button, input, textarea')?.focus();
    }
    else voiceToggle.focus();
});
settingsClose?.addEventListener('click', () => settingsBtn.click());

// No-think defaults ON (slow devices + thinking models = runaway loops);
// the user's choice sticks across sessions.
noThinkCheckbox.checked = localStorage.getItem('locally-no-think') !== 'off';
syncThinkSwitch();
noThinkCheckbox.addEventListener('change', () => {
    syncThinkSwitch();
    localStorage.setItem('locally-no-think', noThinkCheckbox.checked ? 'on' : 'off');
    invalidateContextCount(true);
    updateContextDisplay();
    scheduleExactContextCount();
});
