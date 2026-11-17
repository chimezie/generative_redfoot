'''
Command-line interface for Generative Redfoot: a generative, conversational workflow and multi-agent system using
PDL (Prompt Declaration Language) and MLX.

This module provides:
1. Cache preparation: runs the `content_model` directive in a PDL `cache` section to build a named prefix cache
2. Model evaluation: wires the MLX backend and sampling defaults into the PDL program and executes it
3. Service orchestration: serves the program over FastAPI when the PDL file has a `server` section (`build_app`)
'''

import asyncio
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import click
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from generative_redfoot.extensions.loom import WorldLoomRead
from generative_redfoot.extensions.mlx import CachePrep, MLXAPSModel, MLXModelEvaluation
from generative_redfoot.generation import GenerationDefaults, MLXBackend
from generative_redfoot.object_pdl_model import (
    ParseDispatcher,
    PDFRead,
    PDLProgram,
    PDLRead,
    PDLRepeat,
    PDLText,
)


def make_dispatcher() -> ParseDispatcher:
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [
        PDLRead,
        WorldLoomRead,
        PDLRepeat,
        PDLText,
        CachePrep,
        MLXModelEvaluation,
        MLXAPSModel,
        PDFRead,
    ]
    return dispatcher


def make_program(
    program_yaml: dict,
    dispatcher,
    backend,
    defaults,
    cache_lookup: dict,
    initial_context: dict | None = None,
    base_dir=None,
) -> PDLProgram:
    '''Instantiate a PDL program with the runtime wiring its MLX blocks need'''
    program = PDLProgram(program_yaml, dispatcher=dispatcher, initial_context=initial_context, base_dir=base_dir)
    program.backend, program.defaults, program.cache_lookup = backend, defaults, cache_lookup
    return program


def prepare_caches(program: PDLProgram, ctx: dict, verbose: bool) -> dict:
    '''
    If the program's `cache` section defines a `content_model`, run it to create the prefix cache file and
    return the {cache name: cache file} lookup for model blocks to reference.
    '''
    if not program.cache or isinstance(program.cache, str):
        return {}
    cache_block = program.cache[0]['text'][-1]  # the content_model block is the last item of the cache text
    prefix_marker = cache_block.get('prefix_marker')
    # Tautological substitution, so the marker survives wordloom rendering and the prompt can be cut there
    ctx.update({prefix_marker: f'{{{prefix_marker}}}'} if prefix_marker else {})
    program.cache_lookup = {cache_block['name']: cache_block['file']}
    PDLText(program.cache[0], program).execute(ctx, verbose=verbose)  # instantiates CachePrep via the dispatcher
    return program.cache_lookup


def build_app(
    program_yaml: dict,
    dispatcher,
    backend,
    defaults,
    cache_lookup: dict,
    base_context: dict | None = None,
    verbose: bool = False,
    logger: logging.Logger | None = None,
    base_dir=None,
) -> FastAPI:
    '''Create the FastAPI app that runs the PDL program once per POST request (request body -> context marker)'''
    server_config = program_yaml['server']
    request_body_marker = server_config['request_body_marker']
    expected_content_type = server_config.get('content_type')
    logger = logger or logging.getLogger(f'{__name__}.generative_redfoot_service')
    base_context = base_context or {}
    generation_lock = asyncio.Lock()  # MLX generation runs one request at a time
    # One fixed worker thread: MLX streams are thread-local, so all model work for every request stays on it
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='redfoot-generation')
    app = FastAPI()

    @app.post(server_config['path'])
    async def process_request(request: Request) -> Response:
        '''
        Run the PDL program with the request body bound to `request_body_marker` (and, for form posts, each form
        field bound to its name) and return the final assistant message as plain text.
        Raises 415 for an unexpected content type and 500 for errors executing the program.
        '''
        start_time = time.time()
        request_id = id(request)  # unique identifier for log correlation
        content_type = request.headers.get('content-type', '')
        form_data = {}
        logger.info(f'Request {request_id}: Received request with content type {content_type}')
        if 'multipart/form-data' in content_type.lower():
            for key, value in (await request.form()).items():
                if hasattr(value, 'file') and hasattr(value, 'filename'):  # an UploadFile
                    try:
                        file_content = await value.read()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f'Request {request_id}: Error reading file upload {key}: {e}')
                        form_data[key] = str(value)
                        continue
                    if 'pdf' in (getattr(value, 'content_type', '') or '').lower():
                        form_data[key] = file_content  # PDFRead wants the raw bytes
                    else:
                        try:
                            form_data[key] = file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            form_data[key] = file_content  # binary: leave it to the PDL program
                else:
                    form_data[key] = str(value)
            body_content = form_data.get(request_body_marker, '')
        elif 'application/x-www-form-urlencoded' in content_type.lower():
            form_data = {key: str(value) for key, value in (await request.form()).items()}
            body_content = form_data.get(request_body_marker, '')
        else:
            if expected_content_type is not None and expected_content_type.lower() not in content_type.lower():
                raise HTTPException(
                    status_code=415, detail=f'Unsupported Media Type. Expected {expected_content_type}.'
                )
            body_content = await request.body()
            try:
                body_content = body_content.decode('utf-8')
            except UnicodeDecodeError:
                logger.debug(f'Request {request_id}: Body is not valid UTF-8; keeping raw bytes (e.g. a PDF)')

        initial_context = {**base_context, **form_data, request_body_marker: body_content}
        logger.debug(
            f'Request {request_id}: Form data: { ({k: v for k, v in form_data.items() if isinstance(v, str)}) }'
        )
        logger.info(f'Request {request_id}: Started processing document of length {len(body_content):,} characters')
        try:
            request_program = make_program(
                program_yaml, dispatcher, backend, defaults, cache_lookup, initial_context, base_dir=base_dir
            )
            logger.debug(f'Request {request_id}: Initial context variable names: {list(initial_context)}')
            logger.debug(f'Request {request_id}: Named K/V caches (& corresponding cache file): {cache_lookup}')
            # Model work is blocking; run it off the event loop (one request at a time, in a single worker thread)
            async with generation_lock:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, partial(request_program.execute, verbose=verbose))
        except Exception as e:
            logger.error(f'Request {request_id}: Error loading or executing PDL program: {e}')
            logger.exception(f'Request {request_id}: Full traceback for PDL program error:')
            raise HTTPException(status_code=500, detail=f'Error loading or executing PDL program: {e}') from e
        logger.info(f'Request {request_id}: Completed in {time.time() - start_time:.1f} seconds')

        messages = request_program.evaluation_environment.get('_') or []
        if not messages or messages[-1]['role'] != 'assistant':
            raise HTTPException(status_code=500, detail='PDL program did not end with an assistant message')
        return Response(str(messages[-1]['content']), media_type='text/plain')

    return app


def configure_service_logging(server_config: dict) -> tuple[logging.Logger, str]:
    log_level = str(server_config.get('log_level', 'debug')).upper()
    log_path = server_config.get('log_path', str(Path(tempfile.gettempdir()) / 'server.log'))
    logger = logging.getLogger(f'{__name__}.generative_redfoot_service')
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, log_level


@click.command()
@click.option('-t', '--temperature', default=1, type=float)
@click.option(
    '-rp',
    '--repetition-penalty',
    default=0,
    type=float,
    help='The penalty factor for repeating tokens (none if not used)',
)
@click.option('--top-k', default=0, type=int, help='Sampling top_k (0 disables)')
@click.option('--top-p', default=0.95, type=float, help='Sampling top_p')
@click.option('--max-tokens', default=800, type=int, help='Max tokens')
@click.option('--min-p', default=0, type=float, help='Sampling min-p')
@click.option(
    '--trust-remote-code/--no-trust-remote-code',
    default=False,
    help='Allow models that execute custom code from their repository',
)
@click.option('--verbose/--no-verbose', default=False)
@click.option('--variables', '-v', 'variables', type=(str, str), multiple=True)
@click.argument('pdl_file')
def main(
    temperature, repetition_penalty, top_k, top_p, max_tokens, min_p, trust_remote_code, verbose, variables, pdl_file
):
    '''
    Execute a PDL (Prompt Declaration Language) file: prepare any prefix caches defined in its `cache` section, run
    its blocks against MLX models, or start a web service if it has a `server` section.

    \b
    VARIABLES are key/value pairs injected into the PDL context.
    PDL_FILE is the path to the PDL program.
    '''
    defaults = GenerationDefaults(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        trust_remote_code=trust_remote_code,
    )
    backend = MLXBackend(trust_remote_code=trust_remote_code)
    dispatcher = make_dispatcher()
    with open(pdl_file) as file:
        program_yaml = yaml.safe_load(file)
    base_dir = Path(pdl_file).resolve().parent
    ctx = dict(variables) if variables else {}
    program = make_program(program_yaml, dispatcher, backend, defaults, {}, ctx, base_dir=base_dir)
    cache_lookup = prepare_caches(program, ctx, verbose)
    program.evaluation_environment = ctx

    server_config = program_yaml.get('server')
    if not server_config:
        program.execute(verbose=verbose)
        if verbose:
            print(program.evaluation_environment)
        return 0

    logger, log_level = configure_service_logging(server_config)
    if variables:
        logger.debug(f'Using variables: {variables} as initial_context')
    app = build_app(
        program_yaml, dispatcher, backend, defaults, cache_lookup, dict(variables), verbose, logger, base_dir=base_dir
    )
    host, port = server_config['host'], int(server_config['port'])
    logger.info(f'Starting generative redfoot service with config: {pdl_file}')
    logger.info(f'Service will be available at: http://{host}:{port}{server_config["path"]}')
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())  # noqa: S104 (host comes from the PDL file)
    return 0


if __name__ == '__main__':
    main()
