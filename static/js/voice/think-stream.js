// Keeping reasoning out of the speaker.
//
// In here: ThinkStream, the client-side twin of the server's _ThinkFilter, and
// the markdown stripping a sentence needs before it is read aloud. It decides
// from stream STATE, not from a regex over the text so far, because mid-stream
// there is no closing tag -- which is how a thinking model once had its
// reasoning spoken and its actual answer never queued at all.
// Not in here: the chat thread's collapsible think block
// (markdown/think-split.js). Different medium, different rules.

export const THINK_OPEN = '<think>';
export const THINK_CLOSE = '</think>';

// Reasoning must never be spoken, and the old regex could only tell reasoning
// from answer once the CLOSING tag existed. Mid-stream there is no closing tag,
// so a thinking model had its reasoning chunked and read aloud — and when
// </think> finally arrived the visible text shrank, which silently starved the
// real answer of ever being queued. This is the server's _ThinkFilter
// (locally.py) applied client-side: decide from the state, not from a regex
// over a half-written document.
export class ThinkStream {
    // expectThinking: the user has allowed reasoning, so treat everything
    // before </think> as reasoning. There is no other marker — a model that
    // reasons in plain prose looks exactly like one that is answering.
    constructor(expectThinking) {
        this.state = expectThinking ? 'thinking' : 'answering';
        this.answer = '';
        // True when the turn ended inside an unclosed think block, i.e. the
        // model reasoned until its budget ran out and never answered.
        this.discarded = false;
    }

    _split(full) {
        const close = full.lastIndexOf(THINK_CLOSE);
        if (close !== -1) {                 // covers <think>…</think> and the
            this.state = 'answering';       // stray-close models in models.json
            return full.slice(close + THINK_CLOSE.length);
        }
        if (this.state === 'thinking') return '';       // still reasoning
        const open = full.indexOf(THINK_OPEN);
        if (open !== -1) { this.state = 'thinking'; return full.slice(0, open); }
        return full;
    }

    // Cumulative text in, answer-only text out. Holds the tail back because a
    // tag arrives split across tokens ("<", "think", ">") and half a tag is
    // both visible garbage and, worse, spoken garbage.
    push(full) {
        const text = this._split(full);
        this.answer = text.length > THINK_CLOSE.length
            ? text.slice(0, -THINK_CLOSE.length)
            : '';
        return this.answer;
    }

    // End of stream: nothing more can arrive, so release the held-back tail.
    flush(full) {
        const text = this._split(full);
        // A small model can burn its whole budget reasoning and never close the
        // tag. The server makes the same call (_ThinkFilter.discarded): showing
        // the reasoning beats returning nothing, which just looks like a hang.
        // Callers that speak the result should check `discarded` first — the
        // argument for showing reasoning is not an argument for reading it out.
        this.discarded = !text.trim();
        this.answer = this.discarded ? full.split(THINK_OPEN).join(' ') : text;
        return this.answer;
    }
}

// The model's answer is written for the eye: code fences and markdown
// punctuation all sound like noise when read aloud. Think blocks are NOT
// handled here — ThinkStream owns that, because it needs stream state that a
// regex over the text so far cannot recover.
export function speakableText(text) {
    return text
        .replace(/```[\s\S]*?```/g, ' (code omitted) ')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/\*(.+?)\*/g, '$1')
        .replace(/^#{1,6}\s*/gm, '')
        .replace(/\s+/g, ' ')
        .trim();
}
