use crossbeam_channel::{bounded, Receiver, Sender};
use ffmpeg_next as ffmpeg;
use ffmpeg_next::ffi;
use ffmpeg_next::format::input;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context as Scaler, flag::Flags};
use ffmpeg_next::util::frame::video::Video;
use std::collections::HashMap;
use std::thread::{self, JoinHandle};

const CACHE_SIZE: usize = 30;

enum DecoderCommand {
    RequestFrame(usize),
    Shutdown,
}

#[allow(dead_code)]
pub enum DecoderResponse {
    Frame(usize, Vec<u32>),
    NotAvailable(usize),
}

pub struct VideoMeta {
    pub path: String,
    pub src_width: u32,
    pub src_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    stream_idx: usize,
    all_pts: Vec<i64>,
    pts_to_idx: HashMap<i64, usize>,
    keyframe_pts: Vec<i64>,
    keyframe_indices: Vec<usize>,
}

impl VideoMeta {
    pub fn open(path: &str, out_width: u32, out_height: u32) -> Option<Self> {
        let start = std::time::Instant::now();
        let mut ictx = input(&path).ok()?;
        let stream = ictx.streams().best(Type::Video)?;
        let stream_idx = stream.index();
        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters()).ok()?;
        let decoder = ctx.decoder().video().ok()?;
        let src_width = decoder.width();
        let src_height = decoder.height();

        let mut all_pts: Vec<i64> = Vec::new();
        let mut pts_to_idx: HashMap<i64, usize> = HashMap::new();
        let mut keyframe_pts: Vec<i64> = Vec::new();
        let mut keyframe_indices: Vec<usize> = Vec::new();
        let mut first_kf_idx = None;
        let mut second_kf_idx = None;

        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_idx { continue; }
            let pts = packet.pts().or(packet.dts()).unwrap_or(0);
            all_pts.push(pts);
            pts_to_idx.entry(pts).or_insert(all_pts.len() - 1);
            if packet.is_key() {
                keyframe_pts.push(pts);
                keyframe_indices.push(all_pts.len() - 1);
                if first_kf_idx.is_none() { first_kf_idx = Some(all_pts.len() - 1); }
                else if second_kf_idx.is_none() { second_kf_idx = Some(all_pts.len() - 1); }
            }
        }

        let keyframe_spacing = match (first_kf_idx, second_kf_idx) {
            (Some(first), Some(second)) => second - first,
            _ => 5,
        };
        eprintln!("    {} frames, {} keyframes, GOP ~{} ({:.1}s)",
            all_pts.len(), keyframe_pts.len(), keyframe_spacing, start.elapsed().as_secs_f64());

        Some(Self {
            path: path.to_string(), src_width, src_height, out_width, out_height,
            stream_idx, all_pts, pts_to_idx, keyframe_pts, keyframe_indices,
        })
    }

    #[allow(dead_code)]
    pub fn frame_count(&self) -> usize {
        self.all_pts.len()
    }
}

pub struct FrameResult<'a> {
    pub frame: &'a Vec<u32>,
    pub frame_idx: usize,  // The actual frame index being returned
}

#[allow(dead_code)]
pub struct DecoderHandle {
    pub meta: VideoMeta,
    cmd_tx: Sender<DecoderCommand>,
    resp_rx: Receiver<DecoderResponse>,
    cache: HashMap<usize, Vec<u32>>,
    cache_order: Vec<usize>,
    pending_request: Option<usize>,
    worker: Option<JoinHandle<()>>,
}

impl DecoderHandle {
    pub fn spawn(meta: VideoMeta) -> Self {
        let (cmd_tx, cmd_rx) = bounded::<DecoderCommand>(4);
        let (resp_tx, resp_rx) = bounded::<DecoderResponse>(4);

        let path = meta.path.clone();
        let out_width = meta.out_width;
        let out_height = meta.out_height;
        let stream_idx = meta.stream_idx;
        let all_pts = meta.all_pts.clone();
        let pts_to_idx = meta.pts_to_idx.clone();
        let keyframe_pts = meta.keyframe_pts.clone();
        let keyframe_indices = meta.keyframe_indices.clone();
        let src_width = meta.src_width;
        let src_height = meta.src_height;

        let worker = thread::spawn(move || {
            decoder_worker(cmd_rx, resp_tx, path, stream_idx, src_width, src_height,
                out_width, out_height, all_pts, pts_to_idx, keyframe_pts, keyframe_indices);
        });

        Self {
            meta, cmd_tx, resp_rx,
            cache: HashMap::new(), cache_order: Vec::new(),
            pending_request: None, worker: Some(worker),
        }
    }

    pub fn request_frame(&mut self, frame_idx: usize) -> Option<FrameResult<'_>> {
        while let Ok(resp) = self.resp_rx.try_recv() {
            match resp {
                DecoderResponse::Frame(idx, data) => { self.cache_insert(idx, data); }
                DecoderResponse::NotAvailable(_) => {}
            }
        }
        if self.cache.contains_key(&frame_idx) {
            self.pending_request = None;
            return self.cache.get(&frame_idx).map(|f| FrameResult { frame: f, frame_idx });
        }
        if self.pending_request != Some(frame_idx) {
            let _ = self.cmd_tx.try_send(DecoderCommand::RequestFrame(frame_idx));
            self.pending_request = Some(frame_idx);
        }
        self.find_nearest_cached_with_idx(frame_idx)
            .map(|(idx, f)| FrameResult { frame: f, frame_idx: idx })
    }

    fn cache_insert(&mut self, idx: usize, buf: Vec<u32>) {
        if self.cache.contains_key(&idx) { return; }
        while self.cache.len() >= CACHE_SIZE {
            if let Some(oldest) = self.cache_order.first().cloned() {
                self.cache.remove(&oldest);
                self.cache_order.remove(0);
            } else { break; }
        }
        self.cache.insert(idx, buf);
        self.cache_order.push(idx);
    }

    fn find_nearest_cached_with_idx(&self, frame_idx: usize) -> Option<(usize, &Vec<u32>)> {
        let mut best_idx = None;
        let mut best_dist = usize::MAX;
        for &cached_idx in self.cache.keys() {
            let dist = (cached_idx as isize - frame_idx as isize).unsigned_abs();
            if dist < best_dist { best_dist = dist; best_idx = Some(cached_idx); }
        }
        best_idx.and_then(|idx| self.cache.get(&idx).map(|f| (idx, f)))
    }
}

impl Drop for DecoderHandle {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(DecoderCommand::Shutdown);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

fn decoder_worker(
    cmd_rx: Receiver<DecoderCommand>, resp_tx: Sender<DecoderResponse>,
    path: String, stream_idx: usize, src_width: u32, src_height: u32,
    out_width: u32, out_height: u32,
    all_pts: Vec<i64>, pts_to_idx: HashMap<i64, usize>,
    keyframe_pts: Vec<i64>, keyframe_indices: Vec<usize>,
) {
    let mut state = WorkerState {
        path, stream_idx, src_width, src_height, out_width, out_height,
        all_pts, pts_to_idx, keyframe_pts, keyframe_indices,
        ictx: None, decoder: None, scaler: None,
        decode_base_idx: None, decoded_since_reset: 0,
    };
    loop {
        match cmd_rx.recv() {
            Ok(DecoderCommand::RequestFrame(frame_idx)) => {
                if let Some(data) = state.decode_frame(frame_idx) {
                    let _ = resp_tx.send(DecoderResponse::Frame(frame_idx, data));
                } else {
                    let _ = resp_tx.send(DecoderResponse::NotAvailable(frame_idx));
                }
            }
            Ok(DecoderCommand::Shutdown) | Err(_) => break,
        }
    }
}

struct WorkerState {
    path: String,
    stream_idx: usize,
    src_width: u32,
    src_height: u32,
    out_width: u32,
    out_height: u32,
    all_pts: Vec<i64>,
    pts_to_idx: HashMap<i64, usize>,
    keyframe_pts: Vec<i64>,
    keyframe_indices: Vec<usize>,
    ictx: Option<ffmpeg::format::context::Input>,
    decoder: Option<ffmpeg::decoder::Video>,
    scaler: Option<Scaler>,
    decode_base_idx: Option<usize>,
    decoded_since_reset: usize,
}

impl WorkerState {
    fn pts_to_frame_idx(&self, pts: i64) -> Option<usize> {
        self.pts_to_idx.get(&pts).copied()
    }

    fn reset_decoder(&mut self, start_frame: usize) {
        self.ictx = None;
        self.decoder = None;
        self.scaler = None;
        self.decode_base_idx = None;
        self.decoded_since_reset = 0;

        let mut ictx = match input(&self.path) { Ok(c) => c, Err(_) => return };

        if start_frame > 0 {
            let target_pts = self.all_pts.get(start_frame).copied().unwrap_or(0);
            let kf_pos = self.keyframe_pts.iter().rposition(|&pts| pts <= target_pts);
            let (kf_pts, kf_idx) = match kf_pos {
                Some(pos) => (self.keyframe_pts[pos], self.keyframe_indices[pos]),
                None if !self.keyframe_pts.is_empty() => (self.keyframe_pts[0], self.keyframe_indices[0]),
                None => (0, 0),
            };
            unsafe {
                let _ = ffi::avformat_seek_file(
                    ictx.as_mut_ptr(), self.stream_idx as i32, i64::MIN, kf_pts, kf_pts, 1);
            }
            self.decode_base_idx = Some(kf_idx);
        } else {
            self.decode_base_idx = Some(0);
        }

        let stream = match ictx.streams().best(Type::Video) { Some(s) => s, None => return };
        let ctx = match ffmpeg::codec::context::Context::from_parameters(stream.parameters()) { Ok(c) => c, Err(_) => return };
        let decoder = match ctx.decoder().video() { Ok(d) => d, Err(_) => return };
        let scaler = match Scaler::get(decoder.format(), self.src_width, self.src_height,
            Pixel::RGBA, self.out_width, self.out_height, Flags::BILINEAR) { Ok(s) => s, Err(_) => return };

        self.ictx = Some(ictx);
        self.decoder = Some(decoder);
        self.scaler = Some(scaler);
    }

    fn decode_one_frame(&mut self) -> Option<(usize, Vec<u32>)> {
        let stream_idx = self.stream_idx;
        let ictx = self.ictx.as_mut()?;
        let decoder = self.decoder.as_mut()?;
        let mut decoded = Video::empty();
        loop {
            let (stream, packet) = ictx.packets().next()?;
            if stream.index() != stream_idx { continue; }
            if decoder.send_packet(&packet).is_ok() {
                if decoder.receive_frame(&mut decoded).is_ok() {
                    let frame_pts = decoded.pts().unwrap_or(i64::MIN);
                    let frame_idx = self.pts_to_frame_idx(frame_pts)
                        .or_else(|| self.decode_base_idx.map(|b| b + self.decoded_since_reset))?;
                    self.decoded_since_reset += 1;
                    let scaler = self.scaler.as_mut()?;
                    let mut rgb = Video::empty();
                    let _ = scaler.run(&decoded, &mut rgb);
                    let buf: Vec<u32> = rgb.data(0).chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[2], c[1], c[0], c[3]]))
                        .collect();
                    return Some((frame_idx, buf));
                }
            }
        }
    }

    fn decode_frame(&mut self, target_frame: usize) -> Option<Vec<u32>> {
        let can_continue = self.ictx.is_some()
            && self.decode_base_idx.map_or(false, |base| {
                let current = base + self.decoded_since_reset;
                target_frame >= current && target_frame <= current + 50
            });
        if !can_continue { self.reset_decoder(target_frame); }
        for _ in 0..100 {
            if let Some((idx, buf)) = self.decode_one_frame() {
                if idx == target_frame { return Some(buf); }
                if idx > target_frame { return None; }
            } else { return None; }
        }
        None
    }
}
