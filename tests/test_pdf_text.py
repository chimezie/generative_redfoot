import logging
import os
import shutil
from pathlib import Path

import pytest
import yaml

from generative_redfoot import pdf_text
from generative_redfoot.object_pdl_model import ParseDispatcher, PDFRead, PDLProgram, PDLText

TEST_PDF = Path(__file__).parent / 'Test.pdf'


@pytest.fixture(autouse=True)
def fresh_text_cache():
    pdf_text.clear_cache()
    yield
    pdf_text.clear_cache()


def run_pdf_block(mode, value, context=None):
    pdl = f'description: t\ntext:\n- {mode}: {value}\n  contribute: [context]\n'
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLText, PDFRead]
    p = PDLProgram(yaml.safe_load(pdl), dispatcher=dispatcher, initial_context=context or {})
    p.execute()
    return p.evaluation_environment['_'][0]['content']


# --- cache and helpers (no PyMuPDF needed) ---------------------------------------------------------------------------

def test_cache_key_includes_mode():
    assert pdf_text.TextCache.key(b'x', True) != pdf_text.TextCache.key(b'x', False)
    assert pdf_text.TextCache.key(b'x', True) == pdf_text.TextCache.key(b'x', True)


def test_cache_evicts_least_recently_used():
    cache = pdf_text.TextCache(max_entries=2)
    cache.set('a', '1')
    cache.set('b', '2')
    assert cache.get('a') == '1'  # 'a' is now more recent than 'b'
    cache.set('c', '3')
    assert cache.get('b') is None and cache.get('a') == '1' and cache.get('c') == '3'
    assert len(cache) == 2


def test_cache_skips_oversized_text():
    cache = pdf_text.TextCache(max_text_bytes=4)
    assert cache.set('big', 'toolong') is False
    assert cache.get('big') is None
    assert cache.set('ok', 'tiny') is True


def test_normalize_ocr_text_collapses_whitespace():
    assert pdf_text.normalize_ocr_text('a \n\n b\t c') == 'a b c'


def test_pdf_bytes_coercions(tmp_path):
    f = tmp_path / 'x.pdf'
    f.write_bytes(b'%PDF-1.4 data')
    assert PDFRead._pdf_bytes(b'raw') == b'raw'
    assert PDFRead._pdf_bytes(bytearray(b'raw')) == b'raw'
    assert PDFRead._pdf_bytes(str(f)) == str(f).encode()  # a str is content, never a path
    assert PDFRead._pdf_bytes('not a path') == b'not a path'
    assert PDFRead._pdf_bytes('%PDF-1.4 inline') == b'%PDF-1.4 inline'  # PDF content, never a path
    assert PDFRead._pdf_bytes(12) == b'12'


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match='does not exist'):
        run_pdf_block('PDF_filename_txt', '{ $f }', {'f': str(tmp_path / 'nope.pdf')})


def test_empty_path_is_reported():
    with pytest.raises(ValueError, match='empty'):
        run_pdf_block('PDF_filename_txt', '{ $f }', {'f': ''})


# --- real PyMuPDF on tests/Test.pdf (text layer; no Tesseract needed) ------------------------------------------------

@pytest.fixture
def pymupdf():
    return pytest.importorskip('pymupdf')


def test_filename_text_layer(pymupdf):
    text = run_pdf_block('PDF_filename_txt', '{ $f }', {'f': str(TEST_PDF)})
    assert text.split() == 'The quick brown fox jumped over the lazy moon'.split()


def test_raw_bytes_text_layer(pymupdf):
    text = run_pdf_block('PDF_raw_read_txt', '{ $doc }', {'doc': TEST_PDF.read_bytes()})
    assert 'quick brown fox' in text


def test_second_extraction_is_a_cache_hit(pymupdf, caplog):
    data = TEST_PDF.read_bytes()
    first = pdf_text.extract_pdf_text(data, ocr=False)
    with caplog.at_level(logging.DEBUG, logger='generative_redfoot.pdf_text'):
        second = pdf_text.extract_pdf_text(data, ocr=False)
    assert first == second and 'cache hit' in caplog.text


def test_corrupted_pdf_is_a_value_error(pymupdf):
    with pytest.raises(ValueError, match='invalid or corrupted'):
        pdf_text.extract_pdf_text(b'this is not a pdf', ocr=False)


@pytest.mark.skipif(shutil.which('tesseract') is not None or os.environ.get('TESSDATA_PREFIX'),
                    reason='checks the message shown when Tesseract is absent')
def test_ocr_without_tesseract_explains_how_to_fix(pymupdf):
    with pytest.raises(RuntimeError, match='brew install tesseract'):
        pdf_text.extract_pdf_text(TEST_PDF.read_bytes(), ocr=True)


@pytest.mark.integration  # runs Tesseract
@pytest.mark.skipif(shutil.which('tesseract') is None, reason='OCR needs tesseract (brew install tesseract)')
def test_ocr_from_file_and_bytes_agree(pymupdf):
    by_file = run_pdf_block('PDF_filename_ocr', '{ $f }', {'f': str(TEST_PDF)})
    pdf_text.clear_cache()
    by_bytes = run_pdf_block('PDF_raw_read_ocr', '{ $doc }', {'doc': TEST_PDF.read_bytes()})
    assert by_file == by_bytes and 'quick brown fox' in by_file
    assert '\n' not in by_file  # OCR text is whitespace-normalized


def test_pages_are_never_glued_together(pymupdf):
    doc = pymupdf.open()
    for text in ('the patient', 'was admitted'):
        doc.new_page().insert_text((72, 72), text)
    extracted = pdf_text.extract_pdf_text(doc.tobytes(), ocr=False)
    assert 'patientwas' not in extracted
    assert extracted.split() == ['the', 'patient', 'was', 'admitted']
