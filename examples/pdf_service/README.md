# PDF briefing service

A small web service: POST a PDF and get back a briefing on it (summary, key points, open questions).  It shows how one
PDL file declares a whole service:

| Feature | Where |
|---|---|
| HTTP service, request body bound to a variable | `server` section (`request_body_marker: document_text`) |
| Prompts kept in a WordLoom library, with file inclusion | `prompts/briefing.loom.toml` includes `prompts/briefing_guidelines.md` |
| Prefix cache built once at startup, cut at a WordLoom marker, KV-quantized | the `content_model` block in `cache` |
| OCR of the uploaded PDF | `PDF_raw_read_ocr: ${ document_text }` |
| Per-request generation that reuses the cached prefix | `model: briefing_cache` |

## Requirements

Apple Silicon, `generative_redfoot` installed with the `pdf` extra (`uv pip install 'generative_redfoot[pdf]'`, or
`uv pip install '.[pdf]'` from a checkout), and Tesseract for OCR (`brew install tesseract`; see the main README).  The
first run downloads `mlx-community/Llama-3.2-3B-Instruct-4bit` (about 1.8 GB).

## Run it

Run from this directory: the bare loom filename resolves to `./prompts/`.

```bash
cd examples/pdf_service
generative_redfoot --verbose service.pdl
```

At startup it renders the two prompts, cuts the user prompt at `{document_text}`, processes that prefix through the
model, and saves the cache to `briefing_prompt_cache.safetensors`.  Then it serves `http://127.0.0.1:8667/brief`.

From another terminal (any PDF works; this one is the repository's test file):

```bash
# raw body: the content type must be application/pdf
curl -X POST -H 'Content-Type: application/pdf' --data-binary @../../tests/Test.pdf http://127.0.0.1:8667/brief

# multipart form: the field name is the request_body_marker
curl -X POST -F 'document_text=@../../tests/Test.pdf;type=application/pdf' http://127.0.0.1:8667/brief

# wrong content type: rejected with 415
curl -i -X POST -H 'Content-Type: text/plain' --data 'hello' http://127.0.0.1:8667/brief
```

With `--verbose`, each request logs `Sending N tokens (M already cached)`: M is the cached prefix and N is only the
document.  The service log is also written to `/tmp/pdf_service.log`.

## Adapting it

- Change the prompts in `prompts/`; restart the service so the prefix cache is rebuilt.
- `quantized_kv_start` is low here so that even this short prefix is quantized.  For real prompts, a few thousand tokens
  is typical; remove `kv_bits` to keep the cache unquantized.  (`kv_bits` cannot be combined with `max_kv_size`.)
- To use the document's text layer instead of OCR (faster, no Tesseract), use `PDF_raw_read_txt`.
