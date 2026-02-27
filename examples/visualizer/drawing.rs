//! Overlay drawing for the visualizer.

use crate::dataset::Clip;
use crate::{OverlaySettings, TrackResults, TrackedObject};
use eframe::egui::{self, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use jamtrack_rs::debug_info::{AssociationStageInfo, FrameDebugInfo};

/// Generate a color for a track ID using golden angle for visual distinction
pub fn track_color(track_id: u64) -> Color32 {
    let golden_angle = 137.508;
    let hue = ((track_id as f32) * golden_angle) % 360.0;
    hsv_to_rgb(hue, 0.75, 0.95)
}

/// Convert HSV to RGB Color32
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color32 {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

// Color constants for ByteTrack visualization
const COLOR_HIGH_CONF: Color32 = Color32::from_rgb(100, 149, 237); // Cornflower blue
const COLOR_LOW_CONF: Color32 = Color32::from_rgb(255, 165, 0);    // Orange
const COLOR_FINAL: Color32 = Color32::from_rgb(220, 50, 50);       // Red
const COLOR_HIGHLIGHT: Color32 = Color32::from_rgb(255, 255, 0);   // Yellow highlight

/// Draw all overlays for the current frame
/// Returns the track_id being hovered (if any)
///
/// # Arguments
/// * `display_frame_idx` - Index for looking up track results (after sampling)
/// * `original_frame_idx` - Index for looking up detections from clip (before sampling)
pub fn draw_overlays(
    painter: &Painter,
    video_rect: Rect,
    clip: &Clip,
    display_frame_idx: usize,
    original_frame_idx: usize,
    settings: &OverlaySettings,
    track_results: Option<&TrackResults>,
    mechanics_stage: usize,
    mouse_pos: Option<Pos2>,
    current_hovered: Option<usize>,
) -> Option<usize> {
    let scale_x = video_rect.width() / clip.video_width as f32;
    let scale_y = video_rect.height() / clip.video_height as f32;

    // Draw raw detections (faded, if enabled)
    if settings.show_detections {
        let detections = clip.get_detections(original_frame_idx);
        for det in detections {
            if det.confidence < settings.detection_min_confidence {
                continue;
            }

            // Filter by class if any classes are enabled
            if !settings.enabled_classes.is_empty() && !settings.enabled_classes.contains(&det.class) {
                continue;
            }

            let (x, y, w, h) = det.to_pixel_rect(clip.video_width, clip.video_height);
            let box_rect = Rect::from_min_size(
                Pos2::new(
                    video_rect.min.x + x * scale_x,
                    video_rect.min.y + y * scale_y,
                ),
                Vec2::new(w * scale_x, h * scale_y),
            );

            // Draw detection box in gray/yellow (untracked)
            painter.rect_stroke(box_rect, 0.0, Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 0, 100)));

            // Draw class label for detections
            if settings.show_class_labels {
                let label_pos = Pos2::new(box_rect.max.x - 2.0, box_rect.max.y + 2.0);
                painter.text(
                    label_pos,
                    egui::Align2::RIGHT_TOP,
                    &det.class,
                    FontId::proportional(10.0),
                    Color32::from_rgba_unmultiplied(255, 255, 0, 150),
                );
            }
        }
    }

    let mut new_hovered: Option<usize> = None;

    // ByteTrack mechanics step-by-step visualization
    if settings.show_bytetrack_mechanics {
        if let Some(results) = track_results {
            if let Some(debug_info) = results.debug_by_frame.get(&display_frame_idx) {
                new_hovered = draw_bytetrack_mechanics(
                    painter, video_rect, debug_info, mechanics_stage, scale_x, scale_y,
                    mouse_pos, current_hovered,
                );
                return new_hovered; // Don't draw normal tracks when showing mechanics
            }
        }
    }

    // Draw ByteTrack results (solid, if enabled) - normal mode
    if settings.show_tracks {
        if let Some(results) = track_results {
            let tracks = results.results_by_frame.get(&display_frame_idx).map(|v| v.as_slice()).unwrap_or(&[]);

            for track in tracks {
                draw_track(painter, video_rect, track, settings, scale_x, scale_y);
            }
        }
    }

    new_hovered
}

/// Draw ByteTrack mechanics step-by-step (each stage shown individually)
/// Returns hovered track_id if any
fn draw_bytetrack_mechanics(
    painter: &Painter,
    video_rect: Rect,
    debug_info: &FrameDebugInfo,
    stage: usize,
    scale_x: f32,
    scale_y: f32,
    mouse_pos: Option<Pos2>,
    current_hovered: Option<usize>,
) -> Option<usize> {
    let mut new_hovered: Option<usize> = None;

    // Build a mapping of detection index -> matched track_id for highlighting
    let mut high_det_to_track: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut low_det_to_track: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for &(track_idx, det_idx) in &debug_info.stage1.matches {
        if track_idx < debug_info.stage1.track_pool.len() {
            high_det_to_track.insert(det_idx, debug_info.stage1.track_pool[track_idx].track_id);
        }
    }
    for &(track_idx, det_idx) in &debug_info.stage2.matches {
        if track_idx < debug_info.stage2.track_pool.len() {
            low_det_to_track.insert(det_idx, debug_info.stage2.track_pool[track_idx].track_id);
        }
    }

    match stage {
        0 => {
            // Stage 0: Kalman predictions (dashed boxes)
            // Show all predictions with state labels
            for pred in &debug_info.predictions {
                let highlighted = current_hovered == Some(pred.track_id);
                let rect = to_screen_rect(&pred.predicted_rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = Some(pred.track_id);
                }

                let color = track_color(pred.track_id as u64);
                let alpha = if highlighted { 255 } else { 150 };
                let draw_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
                let stroke_width = if highlighted { 3.0 } else { 2.0 };

                draw_dashed_rect(painter, rect, Stroke::new(stroke_width, draw_color), 6.0, 4.0);

                // Label based on track state
                let label = match pred.state {
                    jamtrack_rs::debug_info::TrackState::Tracked => "PRED",
                    jamtrack_rs::debug_info::TrackState::Lost => "LOST",
                    jamtrack_rs::debug_info::TrackState::New => "NEW",
                    jamtrack_rs::debug_info::TrackState::Removed => "REM",
                };
                draw_track_label(painter, rect, pred.track_id, draw_color, label);
            }

            // If no predictions, show a message
            if debug_info.predictions.is_empty() {
                painter.text(
                    Pos2::new(video_rect.center().x, video_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "No active tracks",
                    FontId::proportional(16.0),
                    Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                );
            }
        }
        1 => {
            // Stage 1: Detections (blue=high-conf, orange=low-conf)
            for (i, det) in debug_info.high_conf_detections.iter().enumerate() {
                let track_id = high_det_to_track.get(&i).copied();
                let highlighted = track_id.is_some() && current_hovered == track_id;
                let rect = to_screen_rect(&det.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = track_id;
                }

                let color = if highlighted { COLOR_HIGHLIGHT } else { COLOR_HIGH_CONF };
                let stroke_width = if highlighted { 3.0 } else { 2.0 };
                painter.rect_stroke(rect, 0.0, Stroke::new(stroke_width, color));

                let label = format!("{:.0}%", det.confidence * 100.0);
                painter.text(
                    Pos2::new(rect.min.x + 2.0, rect.max.y + 2.0),
                    egui::Align2::LEFT_TOP, &label, FontId::proportional(10.0), color,
                );
            }
            for (i, det) in debug_info.low_conf_detections.iter().enumerate() {
                let track_id = low_det_to_track.get(&i).copied();
                let highlighted = track_id.is_some() && current_hovered == track_id;
                let rect = to_screen_rect(&det.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = track_id;
                }

                let color = if highlighted { COLOR_HIGHLIGHT } else { COLOR_LOW_CONF };
                let stroke_width = if highlighted { 3.0 } else { 2.0 };
                painter.rect_stroke(rect, 0.0, Stroke::new(stroke_width, color));

                let label = format!("{:.0}%", det.confidence * 100.0);
                painter.text(
                    Pos2::new(rect.min.x + 2.0, rect.max.y + 2.0),
                    egui::Align2::LEFT_TOP, &label, FontId::proportional(10.0), color,
                );
            }

            // Show message if no detections
            if debug_info.high_conf_detections.is_empty() && debug_info.low_conf_detections.is_empty() {
                painter.text(
                    Pos2::new(video_rect.center().x, video_rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    "No detections this frame",
                    FontId::proportional(16.0),
                    Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                );
            }
        }
        2 => {
            // Stage 2: High-conf matching
            // Draw track predictions
            for entry in &debug_info.stage1.track_pool {
                let highlighted = current_hovered == Some(entry.track_id);
                let rect = to_screen_rect(&entry.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = Some(entry.track_id);
                }

                let color = track_color(entry.track_id as u64);
                let alpha = if highlighted { 255 } else { 150 };
                let draw_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
                let stroke_width = if highlighted { 3.0 } else { 2.0 };

                draw_dashed_rect(painter, rect, Stroke::new(stroke_width, draw_color), 6.0, 4.0);
                draw_track_label(painter, rect, entry.track_id, draw_color, "");
            }
            // Draw high-conf detections
            for (i, det) in debug_info.high_conf_detections.iter().enumerate() {
                let track_id = high_det_to_track.get(&i).copied();
                let highlighted = track_id.is_some() && current_hovered == track_id;
                let rect = to_screen_rect(&det.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = track_id;
                }

                let color = if highlighted { COLOR_HIGHLIGHT } else { COLOR_HIGH_CONF };
                let stroke_width = if highlighted { 3.0 } else { 2.0 };
                painter.rect_stroke(rect, 0.0, Stroke::new(stroke_width, color));

                // Show matched track ID if any
                if let Some(tid) = track_id {
                    draw_track_label(painter, rect, tid, color, "");
                }
            }
            // Draw match lines
            draw_stage_matches_highlighted(
                painter, video_rect, &debug_info.stage1, &debug_info.high_conf_detections,
                COLOR_HIGH_CONF, scale_x, scale_y, current_hovered,
            );
        }
        3 => {
            // Stage 3: Low-conf matching
            for entry in &debug_info.stage2.track_pool {
                let highlighted = current_hovered == Some(entry.track_id);
                let rect = to_screen_rect(&entry.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = Some(entry.track_id);
                }

                let color = track_color(entry.track_id as u64);
                let alpha = if highlighted { 255 } else { 150 };
                let draw_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
                let stroke_width = if highlighted { 3.0 } else { 2.0 };

                draw_dashed_rect(painter, rect, Stroke::new(stroke_width, draw_color), 6.0, 4.0);
                draw_track_label(painter, rect, entry.track_id, draw_color, "");
            }
            for (i, det) in debug_info.low_conf_detections.iter().enumerate() {
                let track_id = low_det_to_track.get(&i).copied();
                let highlighted = track_id.is_some() && current_hovered == track_id;
                let rect = to_screen_rect(&det.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = track_id;
                }

                let color = if highlighted { COLOR_HIGHLIGHT } else { COLOR_LOW_CONF };
                let stroke_width = if highlighted { 3.0 } else { 2.0 };
                painter.rect_stroke(rect, 0.0, Stroke::new(stroke_width, color));

                if let Some(tid) = track_id {
                    draw_track_label(painter, rect, tid, color, "");
                }
            }
            draw_stage_matches_highlighted(
                painter, video_rect, &debug_info.stage2, &debug_info.low_conf_detections,
                COLOR_LOW_CONF, scale_x, scale_y, current_hovered,
            );
        }
        4 => {
            // Stage 4: Final tracks (red solid boxes)
            for output in &debug_info.track_outputs {
                if !output.is_activated {
                    continue;
                }
                let highlighted = current_hovered == Some(output.track_id);
                let rect = to_screen_rect(&output.rect, video_rect, scale_x, scale_y);

                if check_hover(mouse_pos, rect) {
                    new_hovered = Some(output.track_id);
                }

                let color = if highlighted { COLOR_HIGHLIGHT } else { COLOR_FINAL };
                let stroke_width = if highlighted { 4.0 } else { 3.0 };
                painter.rect_stroke(rect, 0.0, Stroke::new(stroke_width, color));
                draw_track_label(painter, rect, output.track_id, color, "");
            }
        }
        _ => {}
    }

    new_hovered
}

fn to_screen_rect(rect: &jamtrack_rs::rect::Rect<f32>, video_rect: Rect, scale_x: f32, scale_y: f32) -> Rect {
    Rect::from_min_size(
        Pos2::new(
            video_rect.min.x + rect.x() * scale_x,
            video_rect.min.y + rect.y() * scale_y,
        ),
        Vec2::new(rect.width() * scale_x, rect.height() * scale_y),
    )
}

fn check_hover(mouse_pos: Option<Pos2>, rect: Rect) -> bool {
    mouse_pos.map(|p| rect.contains(p)).unwrap_or(false)
}

fn draw_track_label(painter: &Painter, rect: Rect, track_id: usize, color: Color32, suffix: &str) {
    let label = if suffix.is_empty() {
        format!("#{}", track_id)
    } else {
        format!("#{} {}", track_id, suffix)
    };
    let label_pos = Pos2::new(rect.min.x + 2.0, rect.min.y - 14.0);
    let text_galley = painter.layout_no_wrap(label.clone(), FontId::proportional(11.0), Color32::WHITE);
    let text_rect = Rect::from_min_size(label_pos, text_galley.size() + Vec2::new(4.0, 2.0));
    painter.rect_filled(text_rect, 2.0, Color32::from_black_alpha(180));
    painter.text(
        Pos2::new(label_pos.x + 2.0, label_pos.y + 1.0),
        egui::Align2::LEFT_TOP, label, FontId::proportional(11.0), color,
    );
}

fn draw_stage_matches_highlighted(
    painter: &Painter,
    video_rect: Rect,
    stage: &AssociationStageInfo,
    detections: &[jamtrack_rs::debug_info::DetectionInfo],
    base_color: Color32,
    scale_x: f32,
    scale_y: f32,
    hovered: Option<usize>,
) {
    for &(track_idx, det_idx) in &stage.matches {
        if track_idx >= stage.track_pool.len() || det_idx >= stage.detection_pool.len() {
            continue;
        }

        let track = &stage.track_pool[track_idx];
        let det_pool_idx = stage.detection_pool[det_idx];
        if det_pool_idx >= detections.len() {
            continue;
        }
        let det = &detections[det_pool_idx];

        let highlighted = hovered == Some(track.track_id);
        let color = if highlighted { COLOR_HIGHLIGHT } else { base_color };
        let stroke_width = if highlighted { 3.0 } else { 2.0 };

        let track_rect = to_screen_rect(&track.rect, video_rect, scale_x, scale_y);
        let det_rect = to_screen_rect(&det.rect, video_rect, scale_x, scale_y);

        painter.line_segment([track_rect.center(), det_rect.center()], Stroke::new(stroke_width, color));
        painter.circle_filled(track_rect.center(), 4.0, color);
        painter.circle_stroke(det_rect.center(), 4.0, Stroke::new(2.0, color));
    }
}

/// Draw a dashed rectangle
fn draw_dashed_rect(painter: &Painter, rect: Rect, stroke: Stroke, dash_len: f32, gap_len: f32) {
    let corners = [
        (rect.left_top(), rect.right_top()),
        (rect.right_top(), rect.right_bottom()),
        (rect.right_bottom(), rect.left_bottom()),
        (rect.left_bottom(), rect.left_top()),
    ];

    for (start, end) in corners {
        draw_dashed_line(painter, start, end, stroke, dash_len, gap_len);
    }
}

fn draw_dashed_line(painter: &Painter, start: Pos2, end: Pos2, stroke: Stroke, dash_len: f32, gap_len: f32) {
    let dir = end - start;
    let len = dir.length();
    if len < 0.001 {
        return;
    }
    let dir = dir / len;
    let total_pattern = dash_len + gap_len;

    let mut pos = 0.0;
    while pos < len {
        let dash_start = start + dir * pos;
        let dash_end = start + dir * (pos + dash_len).min(len);
        painter.line_segment([dash_start, dash_end], stroke);
        pos += total_pattern;
    }
}

fn draw_track(
    painter: &Painter,
    video_rect: Rect,
    track: &TrackedObject,
    settings: &OverlaySettings,
    scale_x: f32,
    scale_y: f32,
) {
    let rect = &track.rect;
    let x = rect.x();
    let y = rect.y();
    let w = rect.width();
    let h = rect.height();

    let box_rect = Rect::from_min_size(
        Pos2::new(
            video_rect.min.x + x * scale_x,
            video_rect.min.y + y * scale_y,
        ),
        Vec2::new(w * scale_x, h * scale_y),
    );

    let color = track_color(track.track_id as u64);

    // Draw track box
    painter.rect_stroke(box_rect, 0.0, Stroke::new(2.5, color));

    // Draw label
    let mut labels: Vec<String> = Vec::new();

    if settings.show_track_ids {
        labels.push(format!("#{}", track.track_id));
    }

    if settings.show_confidence {
        labels.push(format!("{:.0}%", track.confidence * 100.0));
    }

    if !labels.is_empty() {
        let label_text = labels.join(" ");
        let label_pos = Pos2::new(box_rect.min.x + 2.0, box_rect.min.y - 14.0);

        // Background for label
        let text_galley = painter.layout_no_wrap(
            label_text.clone(),
            FontId::proportional(12.0),
            Color32::WHITE,
        );
        let text_rect = Rect::from_min_size(
            label_pos,
            text_galley.size() + Vec2::new(4.0, 2.0),
        );
        painter.rect_filled(text_rect, 2.0, Color32::from_black_alpha(180));

        // Label text
        painter.text(
            Pos2::new(label_pos.x + 2.0, label_pos.y + 1.0),
            egui::Align2::LEFT_TOP,
            label_text,
            FontId::proportional(12.0),
            color,
        );
    }

    // Draw velocity vector
    if settings.show_velocities {
        let (vx, vy) = track.velocity;
        if vx.abs() > 0.001 || vy.abs() > 0.001 {
            let center = box_rect.center();
            let vel_scale = 5.0; // Scale factor for visibility
            let end = Pos2::new(
                center.x + vx * vel_scale * scale_x,
                center.y + vy * vel_scale * scale_y,
            );

            painter.line_segment([center, end], Stroke::new(2.0, Color32::GREEN));
            draw_arrow_head(painter, center, end, Color32::GREEN);
        }
    }
}

fn draw_arrow_head(painter: &Painter, from: Pos2, to: Pos2, color: Color32) {
    let dir = (to - from).normalized();
    let perp = Vec2::new(-dir.y, dir.x);
    let arrow_size = 6.0;

    let p1 = to - dir * arrow_size + perp * (arrow_size * 0.5);
    let p2 = to - dir * arrow_size - perp * (arrow_size * 0.5);

    painter.line_segment([to, p1], Stroke::new(2.0, color));
    painter.line_segment([to, p2], Stroke::new(2.0, color));
}

/// Draw a legend explaining the current visualization stage
pub fn draw_mechanics_legend(painter: &Painter, video_rect: Rect, stage: usize) {
    let legend_height = 28.0;
    let legend_rect = Rect::from_min_max(
        Pos2::new(video_rect.min.x, video_rect.max.y - legend_height),
        video_rect.max,
    );

    // Semi-transparent background
    painter.rect_filled(legend_rect, 0.0, Color32::from_black_alpha(200));

    let y = legend_rect.min.y + 7.0;
    let mut x = legend_rect.min.x + 10.0;
    let spacing = 20.0;
    let box_size = 14.0;

    let items: Vec<(LegendItem, &str)> = match stage {
        0 => vec![
            (LegendItem::DashedBox(track_color(1)), "PRED=Active"),
            (LegendItem::DashedBox(track_color(2)), "LOST=Lost track"),
            (LegendItem::DashedBox(track_color(3)), "NEW=Pending"),
        ],
        1 => vec![
            (LegendItem::SolidBox(COLOR_HIGH_CONF), "High-conf detection"),
            (LegendItem::SolidBox(COLOR_LOW_CONF), "Low-conf detection"),
        ],
        2 => vec![
            (LegendItem::DashedBox(track_color(1)), "Track (predicted)"),
            (LegendItem::SolidBox(COLOR_HIGH_CONF), "High-conf detection"),
            (LegendItem::Line(COLOR_HIGH_CONF), "Match"),
        ],
        3 => vec![
            (LegendItem::DashedBox(track_color(1)), "Unmatched track"),
            (LegendItem::SolidBox(COLOR_LOW_CONF), "Low-conf detection"),
            (LegendItem::Line(COLOR_LOW_CONF), "Match"),
        ],
        4 => vec![
            (LegendItem::SolidBox(COLOR_FINAL), "Final track output"),
        ],
        _ => vec![],
    };

    for (item, label) in items {
        let item_rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(box_size, box_size));

        match item {
            LegendItem::SolidBox(color) => {
                painter.rect_stroke(item_rect, 0.0, Stroke::new(2.0, color));
            }
            LegendItem::DashedBox(color) => {
                draw_dashed_rect(painter, item_rect, Stroke::new(2.0, color), 4.0, 2.0);
            }
            LegendItem::Line(color) => {
                let line_y = y + box_size / 2.0;
                painter.line_segment(
                    [Pos2::new(x, line_y), Pos2::new(x + box_size, line_y)],
                    Stroke::new(2.0, color),
                );
                painter.circle_filled(Pos2::new(x, line_y), 3.0, color);
                painter.circle_stroke(Pos2::new(x + box_size, line_y), 3.0, Stroke::new(1.5, color));
            }
        }

        x += box_size + 5.0;

        painter.text(
            Pos2::new(x, y + 1.0),
            egui::Align2::LEFT_TOP,
            label,
            FontId::proportional(11.0),
            Color32::WHITE,
        );

        // Measure text width approximately
        let text_width = label.len() as f32 * 6.0;
        x += text_width + spacing;
    }
}

enum LegendItem {
    SolidBox(Color32),
    DashedBox(Color32),
    Line(Color32),
}
