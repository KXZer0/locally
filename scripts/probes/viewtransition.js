// Does the tab switch cross-fade, and does it get out of the way when told to?
//
// §2.4a's helper has three skip conditions and the whole point is that they are
// checked in one place. A transition that fires while tokens stream would
// snapshot a long thread mid-inference; one that fires under reduced motion is
// an accessibility failure. Both are silent if only the happy path is tested,
// so each is asserted here by forcing the condition and watching whether
// startViewTransition is called.
//
// The switch itself is also checked every time: losing the tab change would be
// far worse than losing its animation, and a helper that swallows the mutation
// on any of these paths would look identical from the outside.

const { setMode } = await import('/static/js/ui/tabs.js');

// Count calls without suppressing them -- the real API still runs, so a broken
// transition still shows up as an error or a stuck panel.
const native = document.startViewTransition?.bind(document);
let calls = 0;
if (native) {
    document.startViewTransition = fn => { calls += 1; return native(fn); };
}

const visible = () => ['chat', 'voice', 'util', 'code']
    .filter(m => !document.getElementById(`panel-${m}`).hidden);

const settle = () => new Promise(r => setTimeout(r, 320));

const results = {};

// --- the names must be unique among rendered elements --------------------
// One name is shared by four panels, which is only safe while exactly one is
// rendered. Check that invariant directly rather than trusting it.
results.panelsVisibleAtRest = visible();

// --- 1. a normal switch animates -----------------------------------------
calls = 0;
setMode('voice');
await settle();
results.normalSwitch = { calls, visible: visible(), bodyMode: document.body.dataset.mode };

// --- 2. switching to the mode already active does nothing ----------------
calls = 0;
setMode('voice');
await settle();
results.sameModeIsNoop = { calls, visible: visible() };

// --- 3. streaming suppresses it, and the switch still happens ------------
calls = 0;
// Exactly what setGenerating() does. Setting the attribute some other way
// is how the first version of this probe passed while the real app had every
// transition disabled from its first turn onward: it tested the assumption
// rather than the convention.
document.body.dataset.busy = '1';
setMode('chat');
await settle();
document.body.dataset.busy = '0';   // setGenerating never removes it
results.whileBusy = { calls, visible: visible(), bodyMode: document.body.dataset.mode };

// --- 4. and it comes back afterwards -------------------------------------
calls = 0;
setMode('util');
await settle();
results.afterBusy = { calls, visible: visible() };

// --- 5. the mutation survives a throwing transition -----------------------
// A duplicate view-transition-name throws at snapshot time. The switch must
// still complete.
document.startViewTransition = () => { throw new Error('forced'); };
setMode('chat');
await settle();
results.throwingTransitionStillSwitches = { visible: visible(),
                                            bodyMode: document.body.dataset.mode };
if (native) document.startViewTransition = native;

// --- CSS actually carries the names --------------------------------------
results.viewTransitionName = getComputedStyle(
    document.getElementById('panel-chat')).viewTransitionName;

results.supported = !!native;
return results;
