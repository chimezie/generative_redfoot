'''
PDF text extraction via PyMuPDF, shared by every `PDF_*` block.

Approach: open the document from bytes, run OCR page by page (collapsing
whitespace so the text is compact enough for a prompt), time it, and memoize the result by SHA-256 of the bytes.

OCR (`get_textpage_ocr`) is performed by Tesseract, which must be installed on the system (macOS:
`brew install tesseract`; Debian/Ubuntu: `apt install tesseract-ocr`) or located via `TESSDATA_PREFIX`.
Text-layer extraction (`ocr=False`) needs only PyMuPDF.
'''
import hashlib
import io
import logging
import re
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

MAX_CACHED_DOCUMENTS = 32  # extracted texts kept in memory (least recently used are dropped)
MAX_CACHED_TEXT_BYTES = 1024 * 1024  # larger extractions are returned but not cached


class TextCache:
    '''Small in-memory LRU of extracted text; process-lifetime (no TTL). Callers run one at a time.'''

    def __init__(self, max_entries: int = MAX_CACHED_DOCUMENTS, max_text_bytes: int = MAX_CACHED_TEXT_BYTES):
        self.max_entries, self.max_text_bytes = max_entries, max_text_bytes
        self._items = OrderedDict()

    @staticmethod
    def key(file_bytes: bytes, ocr: bool) -> tuple[str, bool]:
        # The mode is part of the key: OCR and text-layer extraction of the same bytes differ
        return hashlib.sha256(file_bytes).hexdigest(), bool(ocr)

    def get(self, key) -> str | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)  # mark as recently used
        return self._items[key]

    def set(self, key, text: str) -> bool:
        '''Store text unless it is too large to be worth keeping; returns whether it was stored'''
        if self.max_entries <= 0 or len(text.encode('utf-8')) > self.max_text_bytes:
            return False
        self._items[key] = text
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        return True

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)


_cache = TextCache()


def clear_cache():
    _cache.clear()


def normalize_ocr_text(text: str) -> str:
    '''Collapse whitespace runs (including newlines) to single spaces'''
    return re.sub(r'\s+', ' ', text)


def extract_pdf_text(file_bytes: bytes, ocr: bool, cache: TextCache | None = None) -> str:
    '''
    Extract text from PDF bytes: OCR'd (whitespace-normalized) when `ocr`, otherwise from the text layer as is.
    Results are memoized in `cache` (the module-level cache by default).
    '''
    cache = _cache if cache is None else cache
    key = TextCache.key(file_bytes, ocr)
    if (cached := cache.get(key)) is not None:
        logger.debug(f'PDF text cache hit for {key[0][:12]} (ocr={ocr})')
        return cached
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        raise ImportError('PDF reading requires the pymupdf package to be installed') from None

    out = io.StringIO()
    start = time.perf_counter()
    try:
        # The explicit filetype lets PyMuPDF open a bare byte stream (it has no filename to infer the format from)
        with pymupdf.open(stream=file_bytes, filetype='pdf') as doc:
            for page in doc:
                if ocr:
                    textpage = page.get_textpage_ocr()
                    out.write(normalize_ocr_text(page.get_text(textpage=textpage)))
                else:
                    out.write(page.get_text())
    except Exception as e:  # PyMuPDF's FileDataError is a RuntimeError too, so sort everything out in one place
        error_str = str(e).lower()
        corrupt = isinstance(e, getattr(pymupdf, 'FileDataError', ())) or 'no objects found' in error_str
        if corrupt or 'code=7' in error_str:
            raise ValueError('The provided PDF content is invalid or corrupted and cannot be processed') from e
        if isinstance(e, RuntimeError) and ('tessdata' in error_str or 'tesseract' in error_str):
            raise RuntimeError(
                'OCR needs Tesseract: install it (macOS: `brew install tesseract`; Debian/Ubuntu: '
                '`apt install tesseract-ocr`) or set TESSDATA_PREFIX; for PDFs that have a text layer, '
                'use PDF_filename_txt / PDF_raw_read_txt instead'
            ) from e
        raise
    text = out.getvalue()
    logger.info(f'Extracted PDF text ({"OCR" if ocr else "text layer"}) in {time.perf_counter() - start:.3f}s')
    if not cache.set(key, text):
        logger.debug(f'Extracted text too large to cache for {key[0][:12]}')
    return text
