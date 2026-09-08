// css-oracle.js — the acceptance test for the stylesheet collapse.
//
// The rule the rebuild plan sets is "do not port by reading; port by
// measurement". This is the measurement. It records, for every element in the
// rendered page, a hash of every computed property that can change what is
// drawn (the WATCH list below, ~110 of the 543 the CSSOM exposes). Two
// stylesheet sets are equivalent if and only if every hash matches; anything
// that changed shows up as a named element rather than as "looks fine to me".
//
// Usage, in the browser console on the running app:
//   1. paste this file
//   2. cssOracle.capture('before')            // once per viewport width
//   3. swap the <link> tags
//   4. cssOracle.capture('after')             // at the same widths
//   5. cssOracle.diff('before', 'after')      // must be empty
//
// Capture once per REAL viewport width (phone / narrow / desktop): the app's
// layout is viewport- and container-conditional, and a collapse that is
// identical at 1440 can still be wrong at 375. The width must be a real
// viewport -- resizing documentElement does not move a media query, and in a
// hidden pane innerWidth is 0, which makes every max-width query match and
// every measurement worthless. capture() refuses to run in that state.
(() => {
    const KEY = name => `css-oracle:${name}`;

    // Path independent of CSS and of any class the collapse might rename.
    const path = el => {
        const parts = [];
        let n = el;
        while (n && n !== document.documentElement) {
            const i = [...n.parentNode.children].indexOf(n) + 1;
            parts.unshift(`${n.tagName.toLowerCase()}:nth-child(${i})`);
            n = n.parentElement;
        }
        return parts.join('>');
    };

    // FNV-1a. Cheap, stable across runs, and it moves if any property moves.
    const hash = s => {
        let h = 0x811c9dc5;
        for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
        return (h >>> 0).toString(36);
    };

    // Reading all 543 computed properties for 920 elements at three widths is
    // 1.5M CSSOM reads and takes longer than a minute. This list is every
    // property that can change what is on screen -- layout, box, paint, type,
    // motion -- which is the thing being tested. Anything outside it (the
    // -webkit aliases, the SVG-only properties, the print set) cannot alter the
    // rendering of this app without also moving one of these.
    const WATCH = ('display,position,top,right,bottom,left,float,clear,'
        + 'width,height,min-width,min-height,max-width,max-height,box-sizing,'
        + 'margin-top,margin-right,margin-bottom,margin-left,'
        + 'padding-top,padding-right,padding-bottom,padding-left,'
        + 'border-top-width,border-right-width,border-bottom-width,border-left-width,'
        + 'border-top-color,border-right-color,border-bottom-color,border-left-color,'
        + 'border-top-style,border-top-left-radius,border-top-right-radius,'
        + 'border-bottom-left-radius,border-bottom-right-radius,outline,box-shadow,'
        + 'background-color,background-image,background-size,background-position,'
        + 'background-repeat,background-clip,color,opacity,visibility,mix-blend-mode,'
        + 'font-family,font-size,font-weight,font-style,font-variant-numeric,'
        + 'line-height,letter-spacing,word-spacing,text-align,text-transform,'
        + 'text-decoration-line,text-overflow,white-space,overflow-wrap,'
        + 'flex-direction,flex-wrap,flex-grow,flex-shrink,flex-basis,order,'
        + 'align-items,align-self,align-content,justify-content,justify-items,justify-self,'
        + 'gap,row-gap,column-gap,grid-template-columns,grid-template-rows,'
        + 'grid-template-areas,grid-area,grid-auto-flow,grid-auto-rows,'
        + 'overflow-x,overflow-y,overscroll-behavior,z-index,'
        + 'transform,transform-origin,transition,animation,filter,backdrop-filter,'
        + 'cursor,pointer-events,user-select,content,contain,content-visibility,'
        + 'aspect-ratio,object-fit,inset,translate,scale,rotate,'
        + 'container-type,list-style-type,vertical-align,text-indent,'
        + 'stroke,stroke-width,fill,writing-mode,isolation,will-change').split(',');

    const snapshot = () => {
        const out = {};
        for (const el of document.querySelectorAll('*')) {
            if (['SCRIPT', 'STYLE', 'LINK'].includes(el.tagName)) continue;
            const cs = getComputedStyle(el);
            let s = '';
            for (const p of WATCH) s += p + ':' + cs.getPropertyValue(p) + ';';
            out[path(el)] = hash(s);
        }
        return out;
    };

    // The widths are emulated by shrinking the document element, not the OS
    // window: the runner may be inside a pane that cannot be resized, and a
    // media query reads the viewport either way.
    window.cssOracle = {
        // One call per viewport width; the caller resizes between calls.
        capture(name) {
            if (!innerWidth) throw new Error('viewport is 0 wide (pane hidden?) — every '
                + 'max-width query would match and the snapshot would be meaningless');
            const w = innerWidth;
            const all = JSON.parse(localStorage.getItem(KEY(name)) || '{}');
            all[w] = snapshot();
            localStorage.setItem(KEY(name), JSON.stringify(all));
            return { width: w, elements: Object.keys(all[w]).length,
                     widths: Object.keys(all).map(Number) };
        },
        diff(a, b) {
            const A = JSON.parse(localStorage.getItem(KEY(a)) || '{}');
            const B = JSON.parse(localStorage.getItem(KEY(b)) || '{}');
            const report = {};
            const widths = Object.keys(A).filter(w => w in B).map(Number);
            if (!widths.length) return { error: 'no width captured in both' };
            for (const w of widths) {
                const changed = [], gone = [], added = [];
                for (const k of Object.keys(A[w] || {})) {
                    if (!(k in (B[w] || {}))) gone.push(k);
                    else if (A[w][k] !== B[w][k]) changed.push(k);
                }
                for (const k of Object.keys(B[w] || {})) if (!(k in (A[w] || {}))) added.push(k);
                report[w] = { changed: changed.length, gone: gone.length, added: added.length,
                              sample: changed.slice(0, 20) };
            }
            return report;
        },
        // What actually differs on one element, once the diff has named it.
        explain(sel) {
            const el = document.querySelector(sel) || document.evaluate;
            const cs = getComputedStyle(el);
            const o = {};
            for (let i = 0; i < cs.length; i++) o[cs[i]] = cs.getPropertyValue(cs[i]);
            return o;
        },
    };
})();
