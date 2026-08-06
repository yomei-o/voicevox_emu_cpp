// A tar reader, small enough to read and big enough for the one archive here.
//
// The Open JTalk dictionary ships as a 23 MB tar.gz that expands to 103 MB; the
// page fetches it as one request, lets the browser's DecompressionStream do the
// gzip, and walks the tar with this.  Only what the archive actually contains is
// handled - regular files and directories, ustar with the prefix field - and a
// member type nobody produces is skipped rather than guessed at.
//
// Loadable three ways so the browser worker and the node rehearsal share it:
// importScripts, a plain script tag, or require.
(function (root) {
    'use strict';

    // `write(name, bytes)` is called for each regular file, in archive order.
    // `bytes` is a view into the input, not a copy: use it before the next call
    // if the consumer keeps references.
    function untar(bytes, write) {
        let off = 0;
        const dec = new TextDecoder();
        const field = (start, len) => {
            const s = bytes.subarray(off + start, off + start + len);
            const end = s.indexOf(0);
            return dec.decode(end < 0 ? s : s.subarray(0, end));
        };
        while (off + 512 <= bytes.length) {
            const name = field(0, 100);
            if (!name) {  // the two zero blocks that end an archive
                off += 512;
                continue;
            }
            const size = parseInt(field(124, 12).trim() || '0', 8);
            const type = String.fromCharCode(bytes[off + 156]);
            const prefix = field(345, 155);
            const full = prefix ? prefix + '/' + name : name;
            off += 512;
            // '0' and NUL are both "regular file"; '5' is a directory, which the
            // consumer creates implicitly, and anything else does not appear.
            if (type === '0' || type === '\0') write(full, bytes.subarray(off, off + size));
            off += Math.ceil(size / 512) * 512;
        }
    }

    root.untarBytes = untar;
    if (typeof module !== 'undefined' && module.exports) module.exports = untar;
})(typeof globalThis !== 'undefined' ? globalThis : this);
