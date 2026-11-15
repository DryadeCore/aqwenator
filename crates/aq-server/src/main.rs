//! aq-serve: OpenAI-compatible `/v1/completions` front end for the aqwenator engine.
//!
//! The engine runs on its own OS thread, which owns the CUDA context and loops
//! `step()` while there is work. HTTP handlers (tokio) submit requests over a channel and
//! receive token events over a per-request channel; detokenization happens on the tokio
//! side so the engine thread never touches text.
//!
//! Supported: prompt as text or token ids, `max_tokens`, `ignore_eos`, `stream` (SSE),
//! `return_token_ids`. Greedy only for now: `temperature` must be absent or 0.
//!
//!     aq-serve <model> [--port 8000] [--max-seqs 64] [--max-ctx 32768] [--max-tokens 8192]
//!              [--mixed-prefill N] [--max-running N] [--session-cache true|false] [--kv-gb X]
//!              [--spec-k N]
//!
//! `--spec-k` (default 4, 0 = off): exact speculative decoding with prompt-lookup drafts;
//! outputs are token-identical to plain greedy decoding.
//!
//! `--max-seqs` is the number of state slots (running + parked sessions); `--max-running`
//! caps how many run at once.

use anyhow::{Context, Result};
use aq_engine::engine::{Engine, Request, TokenEvent};
use aq_engine::model::{Limits, Model};
use aq_engine::weights::resolve_model;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokenizers::tokenizer::pipeline::{EncodeOptions, PipelineDecodeStream, PipelineTokenizer};
use tokio::sync::mpsc;

enum Msg {
    Token(TokenEvent),
    Error(String),
}

struct Submit {
    req: Request,
    tx: mpsc::UnboundedSender<Msg>,
}

struct App {
    submit: std::sync::mpsc::Sender<Submit>,
    tok: Arc<PipelineTokenizer>,
    next_id: AtomicU64,
    model: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Ids(Vec<u32>),
    Text(String),
}

#[derive(Deserialize)]
struct CompletionReq {
    prompt: Prompt,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    ignore_eos: Option<bool>,
    stream: Option<bool>,
    return_token_ids: Option<bool>,
}

fn bad(msg: impl Into<String>) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

/// Incremental detokenizer state for one response.
struct Detok {
    tok: Arc<PipelineTokenizer>,
    stream: PipelineDecodeStream,
}

impl Detok {
    fn push(&mut self, id: u32) -> String {
        self.stream.step(&self.tok, &[id]).ok().flatten().unwrap_or_default()
    }
}

async fn completions(State(app): State<Arc<App>>, Json(body): Json<CompletionReq>) -> Response {
    if body.temperature.is_some_and(|t| t > 0.0) {
        return bad("only greedy decoding (temperature 0) is supported for now");
    }
    let prompt = match body.prompt {
        Prompt::Ids(ids) => ids,
        Prompt::Text(text) => {
            let opts = EncodeOptions { add_special_tokens: false, ..Default::default() };
            match app.tok.encode(text, &opts).wait() {
                Ok(encs) => encs.first().map(|e| e.ids().iter().map(|&t| u32::from(t)).collect()).unwrap_or_default(),
                Err(e) => return bad(format!("tokenization failed: {e}")),
            }
        }
    };
    if prompt.is_empty() {
        return bad("empty prompt");
    }
    let id = app.next_id.fetch_add(1, Ordering::Relaxed);
    let req = Request {
        id,
        prompt,
        max_new_tokens: body.max_tokens.unwrap_or(16),
        ignore_eos: body.ignore_eos.unwrap_or(false),
    };
    let (tx, rx) = mpsc::unbounded_channel();
    if app.submit.send(Submit { req, tx }).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine stopped").into_response();
    }
    let with_ids = body.return_token_ids.unwrap_or(false);
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let rid = format!("cmpl-{id}");
    let model = app.model.clone();
    let mut detok = Detok { tok: app.tok.clone(), stream: PipelineDecodeStream::new(Vec::new(), false) };

    if body.stream.unwrap_or(false) {
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).flat_map(move |msg| {
            let events: Vec<Result<Event, Infallible>> = match msg {
                Msg::Error(e) => vec![
                    Ok(Event::default().data(json!({"error": {"message": e}}).to_string())),
                    Ok(Event::default().data("[DONE]")),
                ],
                Msg::Token(e) => {
                    let mut choice = json!({"index": 0, "text": detok.push(e.token), "logprobs": null,
                        "finish_reason": if !e.finished { json!(null) } else if e.stop { json!("stop") } else { json!("length") }});
                    if with_ids {
                        choice["token_ids"] = json!([e.token]);
                    }
                    let chunk = json!({"id": rid, "object": "text_completion", "created": created, "model": model, "choices": [choice]});
                    let mut v = vec![Ok(Event::default().data(chunk.to_string()))];
                    if e.finished {
                        v.push(Ok(Event::default().data("[DONE]")));
                    }
                    v
                }
            };
            futures::stream::iter(events)
        });
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }

    let mut rx = rx;
    let mut toks = Vec::new();
    let mut text = String::new();
    let mut finish = "length";
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Error(e) => return bad(e),
            Msg::Token(e) => {
                toks.push(e.token);
                text.push_str(&detok.push(e.token));
                if e.finished {
                    if e.stop {
                        finish = "stop";
                    }
                    break;
                }
            }
        }
    }
    let mut choice = json!({"index": 0, "text": text, "logprobs": null, "finish_reason": finish});
    if with_ids {
        choice["token_ids"] = json!(toks);
    }
    Json(json!({"id": rid, "object": "text_completion", "created": created, "model": app.model, "choices": [choice],
        "usage": {"completion_tokens": toks.len()}}))
    .into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    Json(json!({"object": "list", "data": [{"id": app.model, "object": "model", "owned_by": "aqwenator"}]}))
}

/// Engine thread: owns the CUDA context; steps while there is work, blocks on the channel
/// when idle.
#[derive(Clone, Copy)]
struct EngineOpts {
    mixed: Option<usize>,
    max_running: Option<usize>,
    session_cache: bool,
    spec_k: usize,
    policy: aq_engine::engine::Policy,
    max_snapshots: Option<usize>,
    itl_target_ms: Option<f64>,
    adaptive: Option<(usize, usize)>,
    prewarm: bool,
    /// Burst prefill for long prompts: tokens per burst step (0 = off) and steps between bursts.
    burst: (usize, usize),
}

fn engine_loop(dir: PathBuf, limits: Limits, opts: EngineOpts, eos: Vec<u32>, rx: std::sync::mpsc::Receiver<Submit>, ready: std::sync::mpsc::Sender<Result<()>>) {
    let t0 = Instant::now();
    let model = match Model::load(&dir, limits) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let mut model = model;
    if opts.prewarm {
        let t = Instant::now();
        match model.prewarm_gemms() {
            Ok(n) => eprintln!("prewarmed GEMM plans for {n} M buckets in {:.1}s", t.elapsed().as_secs_f64()),
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        }
    }
    let mut eng = Engine::new(model, eos);
    if let Some(m) = opts.mixed {
        eng.mixed_prefill_budget = m;
    }
    if let Some(r) = opts.max_running {
        eng.max_running = r;
    }
    eng.session_cache = opts.session_cache;
    eng.spec_k = opts.spec_k;
    eng.policy = opts.policy;
    // AQ_BURST_PREFILL / AQ_BURST_EVERY (read by the engine) win over the flags when set.
    if std::env::var("AQ_BURST_PREFILL").is_err() {
        eng.burst_prefill = opts.burst.0;
    }
    if std::env::var("AQ_BURST_EVERY").is_err() {
        eng.burst_every = opts.burst.1;
    }
    if let Some(n) = opts.max_snapshots {
        eng.max_snapshots = n;
    }
    eng.itl_target_us = opts.itl_target_ms.map(|m| m * 1000.0);
    eng.mixed_prefill_adaptive = opts.adaptive;
    eprintln!("policy: {:?} max_snapshots {} itl_target_ms {:?}", eng.policy, eng.max_snapshots, opts.itl_target_ms);
    let mut last_report = Instant::now();
    eprintln!("engine ready in {:.2}s", t0.elapsed().as_secs_f64());
    let _ = ready.send(Ok(()));
    let mut sinks: HashMap<u64, mpsc::UnboundedSender<Msg>> = HashMap::new();
    let mut accept = |eng: &mut Engine, s: Submit, sinks: &mut HashMap<u64, mpsc::UnboundedSender<Msg>>| {
        let id = s.req.id;
        match eng.add(s.req) {
            Ok(()) => {
                sinks.insert(id, s.tx);
            }
            Err(e) => {
                let _ = s.tx.send(Msg::Error(e.to_string()));
            }
        }
    };
    loop {
        if !eng.has_work() {
            match rx.recv() {
                Ok(s) => accept(&mut eng, s, &mut sinks),
                Err(_) => return,
            }
        }
        while let Ok(s) = rx.try_recv() {
            accept(&mut eng, s, &mut sinks);
        }
        if last_report.elapsed().as_secs() >= 10 {
            let c = eng.cache;
            eprintln!(
                "[stats] running {} parked {} | admitted {} hits {} (forks {}) reused {} prefilled {} snapshots {} checkpoints {} evictions {} (snapshot {}) deferred {} host {:.0}us/step | spec drafted {} accepted {} (cross-session {} / {}, mtp {} / {}) tok/step {:.2}",
                eng.num_running(), eng.num_parked(), c.admitted, c.hits, c.forks, c.reused_tokens, c.prefilled_tokens, c.snapshots, c.checkpoints, c.evictions,
                c.snapshot_evictions, c.deferred, c.host_us as f64 / c.steps.max(1) as f64,
                eng.spec.drafted, eng.spec.accepted, eng.spec.global_accepted, eng.spec.global_drafted, eng.spec.mtp_accepted, eng.spec.mtp_drafted, eng.spec.emitted as f64 / eng.spec.decode_entries.max(1) as f64
            );
            last_report = Instant::now();
        }
        match eng.step() {
            Ok(events) => {
                for e in events {
                    // A failed send means the handler dropped its receiver: the client went
                    // away. Stop generating for it instead of running to its token limit.
                    let gone = sinks.get(&e.id).is_some_and(|tx| tx.send(Msg::Token(e)).is_err());
                    if gone && !e.finished {
                        eng.cancel(e.id);
                    }
                    if gone || e.finished {
                        sinks.remove(&e.id);
                    }
                }
                // Clients that left before their first token (still queued or prefilling).
                let closed: Vec<u64> = sinks.iter().filter(|(_, tx)| tx.is_closed()).map(|(&id, _)| id).collect();
                for id in closed {
                    eng.cancel(id);
                    sinks.remove(&id);
                }
            }
            Err(e) => {
                eprintln!("engine step failed: {e:#}");
                for (_, tx) in sinks.drain() {
                    let _ = tx.send(Msg::Error(format!("engine error: {e}")));
                }
                return;
            }
        }
    }
}

fn eos_ids(dir: &std::path::Path) -> Vec<u32> {
    let mut out = Vec::new();
    for f in ["generation_config.json", "config.json"] {
        if let Ok(s) = std::fs::read_to_string(dir.join(f)) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                for key in [&v["eos_token_id"], &v["text_config"]["eos_token_id"]] {
                    match key {
                        serde_json::Value::Number(n) => out.extend(n.as_u64().map(|x| x as u32)),
                        serde_json::Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32))),
                        _ => {}
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("usage: aq-serve <model> [--port N] [--max-seqs N] [--max-ctx N] [--max-tokens N] [--mixed-prefill N]")?;
    let (mut port, mut max_seqs, mut max_ctx, mut max_tokens) = (8000u16, 64usize, 32768usize, 8192usize);
    let mut opts = EngineOpts { mixed: None, max_running: None, session_cache: true, spec_k: 4, policy: Default::default(), max_snapshots: None, itl_target_ms: None, adaptive: None, prewarm: false, burst: (4096, 3) };
    // Shortest remaining prefill first by default (`--srpt false` restores admission order):
    // on real agent traces it keeps small turns from queueing behind a long tool result, at
    // the same throughput (Spark, 16 agents: TTFT p90 2633 -> 1506 ms).
    opts.policy.srpt = true;
    let mut kv_gb = 0f64;
    while let Some(k) = args.next() {
        let v = args.next().with_context(|| format!("missing value for {k}"))?;
        match k.as_str() {
            "--port" => port = v.parse()?,
            "--max-seqs" => max_seqs = v.parse()?,
            "--max-ctx" => max_ctx = v.parse()?,
            "--max-tokens" => max_tokens = v.parse()?,
            "--mixed-prefill" => opts.mixed = Some(v.parse()?),
            "--max-running" => opts.max_running = Some(v.parse()?),
            "--session-cache" => opts.session_cache = v.parse()?,
            "--spec-k" => opts.spec_k = v.parse()?,
            "--evict" => {
                opts.policy.evict = match v.as_str() {
                    "lru" => aq_engine::engine::Evict::Lru,
                    "mru" => aq_engine::engine::Evict::Mru,
                    "idle" => aq_engine::engine::Evict::Idle,
                    _ => anyhow::bail!("--evict lru|mru|idle"),
                }
            }
            "--pin-snapshots" => opts.policy.pin_snapshots = v.parse()?,
            "--load-control" => opts.policy.load_control = v.parse()?,
            "--srpt" => opts.policy.srpt = v.parse()?,
            // A long prompt gets a whole step of up to N prefill tokens to itself every
            // `--burst-every` steps (default 4096 every 3; 0 = off): on real traces it halves
            // the time to first token of turns with 3k+ new tokens for the same throughput.
            "--burst-prefill" => opts.burst.0 = v.parse()?,
            "--burst-every" => opts.burst.1 = v.parse()?,
            "--max-wait-ms" => opts.policy.max_wait_ms = v.parse()?,
            "--dead-after-ms" => opts.policy.dead_after_ms = v.parse()?,
            "--quantum" => opts.policy.quantum = v.parse()?,
            "--prewarm" => opts.prewarm = v.parse()?,
            "--mixed-prefill-adaptive" => {
                // "HI:THRESHOLD", e.g. 8192:16384
                let (a, b) = v.split_once(':').context("--mixed-prefill-adaptive HI:THRESHOLD")?;
                opts.adaptive = Some((a.parse()?, b.parse()?));
            }
            "--max-snapshots" => opts.max_snapshots = Some(v.parse()?),
            "--itl-target-ms" => opts.itl_target_ms = Some(v.parse()?),
            "--kv-gb" => kv_gb = v.parse()?,
            _ => anyhow::bail!("unknown option {k}"),
        }
    }
    let dir = resolve_model(&model)?;
    // Hub tokenizer.json files are format 1.0; tokenizers v1 reads 2.0, so upgrade in memory.
    let json = tokenizers::canonicalize_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let tok = Arc::new(tokenizers::from_json(&json).map_err(|e| anyhow::anyhow!("{e}"))?);
    let eos = eos_ids(&dir);
    eprintln!("eos ids: {eos:?}");
    let limits = Limits { max_tokens, max_seqs, max_ctx, block_size: 16, kv_budget_bytes: (kv_gb * (1u64 << 30) as f64) as usize, kv_blocks: 0 };
    let (tx, rx) = std::sync::mpsc::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_loop(dir, limits, opts, eos, rx, ready_tx))?;
    ready_rx.recv()??;

    let app = Arc::new(App { submit: tx, tok, next_id: AtomicU64::new(0), model });
    let router = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(|| async { "ok" }))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    eprintln!("listening on :{port}");
    axum::serve(listener, router).await?;
    Ok(())
}
