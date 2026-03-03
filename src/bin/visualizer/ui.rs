//! UI rendering for the visualizer.

use crate::drawing;
use crate::VisualizerApp;
use eframe::egui::{self, Color32, ColorImage, TextureOptions};

pub fn render(app: &mut VisualizerApp, ctx: &egui::Context) {
    // Handle keyboard input
    handle_input(app, ctx);

    // Top panel with file info
    egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Separate buttons for Video, Detections, Timestamps
            if ui.button("Video...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Video", &["mp4", "avi", "mkv", "mov", "webm"])
                    .set_title("Select video file")
                    .pick_file()
                {
                    app.pending_video = Some(path);
                }
            }

            if ui.button("Detections...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .set_title("Select detections JSON file")
                    .pick_file()
                {
                    app.pending_detections = Some(path);
                }
            }

            if ui.button("Timestamps...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .set_title("Select timestamps JSON file")
                    .pick_file()
                {
                    app.pending_timestamps = Some(path);
                }
            }

            // Show pending files status
            let has_video = app.pending_video.is_some();
            let has_detections = app.pending_detections.is_some();
            let has_timestamps = app.pending_timestamps.is_some();

            if has_video || has_detections || has_timestamps {
                ui.separator();
                // Show checkmarks for selected files
                let video_mark = if has_video { "V" } else { "-" };
                let det_mark = if has_detections { "D" } else { "-" };
                let ts_mark = if has_timestamps { "T" } else { "-" };
                ui.label(format!("[{}{}{}]", video_mark, det_mark, ts_mark));
            }

            // Show "Load Dataset" button when all three are selected
            if has_video && has_detections && has_timestamps {
                if ui.button("Load Dataset").clicked() {
                    let video = app.pending_video.take().unwrap();
                    let det = app.pending_detections.take().unwrap();
                    let ts = app.pending_timestamps.take().unwrap();
                    app.load_with_video(&video, &det, &ts);
                }
            }

            ui.separator();

            if let Some(clip) = &app.clip {
                ui.label(format!(
                    "Video: {}x{} | {} frames",
                    clip.video_width, clip.video_height, clip.frame_count
                ));
                if clip.has_ground_truth {
                    ui.colored_label(Color32::GREEN, "(GT)");
                }
            } else if let Some(err) = &app.load_error {
                ui.colored_label(Color32::RED, err);
            } else {
                ui.label("Select Video, Detections, and Timestamps files to load");
            }
        });
    });

    // Left panel with controls (scrollable)
    egui::SidePanel::left("controls_panel")
        .default_width(250.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_controls(app, ui);
            });
        });

    // Bottom panel with timeline (larger when eval overlays are showing)
    let timeline_height = if app.overlay_settings.show_eval_overlays && app.get_gt_timeseries().len() > 0 {
        200.0
    } else {
        60.0
    };
    egui::TopBottomPanel::bottom("timeline_panel")
        .min_height(timeline_height)
        .show(ctx, |ui| {
            render_timeline(app, ui, ctx);
        });

    // Central panel with video
    egui::CentralPanel::default().show(ctx, |ui| {
        render_video(app, ui, ctx);
    });

    // Auto-advance if playing
    if app.is_playing {
        let now = std::time::Instant::now();

        // When mechanics is enabled, we step through 5 stages per frame
        // so each step is 1/5th of the frame duration
        let steps_per_frame = if app.overlay_settings.show_bytetrack_mechanics { 5.0 } else { 1.0 };
        let step_duration = std::time::Duration::from_secs_f32(1.0 / (app.playback_fps * steps_per_frame));

        if let Some(last_time) = app.last_frame_time {
            if now.duration_since(last_time) >= step_duration {
                if app.overlay_settings.show_bytetrack_mechanics {
                    // Step through mechanics stages, then advance frame
                    if app.mechanics_stage < 4 {
                        app.mechanics_stage += 1;
                    } else {
                        // All stages done, go to next frame
                        if app.current_frame + 1 < app.frame_count() {
                            app.current_frame += 1;
                            app.mechanics_stage = 0;
                        } else {
                            app.is_playing = false;
                        }
                    }
                } else {
                    // No mechanics, just advance frame
                    if app.current_frame + 1 < app.frame_count() {
                        app.current_frame += 1;
                    } else {
                        app.is_playing = false;
                    }
                }
                app.last_frame_time = Some(now);
            }
        } else {
            app.last_frame_time = Some(now);
        }

        ctx.request_repaint();
    }

    // Always request repaint when mechanics visualization is active to ensure smooth rendering
    if app.overlay_settings.show_bytetrack_mechanics {
        ctx.request_repaint();
    }
}

fn handle_input(app: &mut VisualizerApp, ctx: &egui::Context) {
    ctx.input(|i| {
        // Space: play/pause
        if i.key_pressed(egui::Key::Space) {
            app.is_playing = !app.is_playing;
            if app.is_playing {
                app.last_frame_time = Some(std::time::Instant::now());
            }
        }

        // Left/Right arrows: frame stepping
        if i.key_pressed(egui::Key::ArrowLeft) {
            app.is_playing = false;
            if app.current_frame > 0 {
                app.current_frame -= 1;
                app.mechanics_stage = 0;
            }
        }
        if i.key_pressed(egui::Key::ArrowRight) {
            app.is_playing = false;
            if app.current_frame + 1 < app.frame_count() {
                app.current_frame += 1;
                app.mechanics_stage = 0;
            }
        }

        // Home/End: jump to start/end
        if i.key_pressed(egui::Key::Home) {
            app.current_frame = 0;
            app.mechanics_stage = 0;
        }
        if i.key_pressed(egui::Key::End) {
            let count = app.frame_count();
            if count > 0 {
                app.current_frame = count - 1;
                app.mechanics_stage = 0;
            }
        }

        // [ and ]: step through ByteTrack mechanics stages
        if i.key_pressed(egui::Key::OpenBracket) && app.mechanics_stage > 0 {
            app.mechanics_stage -= 1;
        }
        if i.key_pressed(egui::Key::CloseBracket) && app.mechanics_stage < 4 {
            app.mechanics_stage += 1;
        }
    });
}

fn render_controls(app: &mut VisualizerApp, ui: &mut egui::Ui) {
    ui.heading("Overlays");
    ui.checkbox(&mut app.overlay_settings.show_detections, "Show Detections");
    ui.checkbox(&mut app.overlay_settings.show_tracks, "Show Tracks");
    ui.checkbox(&mut app.overlay_settings.show_track_ids, "Show Track IDs");
    ui.checkbox(&mut app.overlay_settings.show_confidence, "Show Confidence");
    ui.checkbox(&mut app.overlay_settings.show_velocities, "Show Velocities");
    ui.checkbox(&mut app.overlay_settings.show_class_labels, "Show Class Labels");

    ui.separator();

    ui.label("Min Confidence:");
    ui.add(egui::Slider::new(&mut app.overlay_settings.detection_min_confidence, 0.0..=1.0));

    ui.separator();

    // Class filter
    if !app.overlay_settings.all_classes.is_empty() {
        ui.heading("Class Filter");
        ui.horizontal(|ui| {
            if ui.button("All").clicked() {
                app.overlay_settings.enabled_classes = app.overlay_settings.all_classes.iter().cloned().collect();
            }
            if ui.button("None").clicked() {
                app.overlay_settings.enabled_classes.clear();
            }
        });
        for class in app.overlay_settings.all_classes.clone() {
            let mut enabled = app.overlay_settings.enabled_classes.contains(&class);
            if ui.checkbox(&mut enabled, &class).changed() {
                if enabled {
                    app.overlay_settings.enabled_classes.insert(class);
                } else {
                    app.overlay_settings.enabled_classes.remove(&class);
                }
            }
        }
        ui.separator();
    }

    ui.heading("ByteTrack Mechanics");
    ui.checkbox(&mut app.overlay_settings.show_bytetrack_mechanics, "Show Step-by-Step");

    if app.overlay_settings.show_bytetrack_mechanics {
        let stage_names = [
            "0: Kalman Predictions",
            "1: Detections Appear",
            "2: Stage 1 Matching",
            "3: Stage 2 Matching",
            "4: Final Updated State",
        ];

        ui.horizontal(|ui| {
            if ui.button("◀").clicked() && app.mechanics_stage > 0 {
                app.mechanics_stage -= 1;
            }
            ui.label(stage_names[app.mechanics_stage]);
            if ui.button("▶").clicked() && app.mechanics_stage < 4 {
                app.mechanics_stage += 1;
            }
        });

        ui.label("[ ] keys or play to step through stages");
    }

    ui.separator();

    // Evaluation section (only show when ground truth is available)
    if app.association_result.is_some() {
        ui.heading("Evaluation");

        // Evaluation overlays checkboxes
        ui.checkbox(&mut app.overlay_settings.show_eval_overlays, "Show Evaluation Overlays");
        if app.overlay_settings.show_eval_overlays {
            ui.indent("eval_overlay_indent", |ui| {
                ui.checkbox(&mut app.overlay_settings.show_gt_boxes, "Ground Truth Boxes");
            });
        }

        // Association Score display
        if let Some(result) = app.get_association_result() {
            let metrics = &result.metrics;
            egui::CollapsingHeader::new("Association Score")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Coverage:");
                        ui.strong(format!("{:.1}%", metrics.coverage() * 100.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Fragmentation:");
                        ui.label(format!("{:.2}", metrics.fragmentation));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Confusion:");
                        ui.label(format!("{:.2}", metrics.confusion));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Tracked No Obj:");
                        ui.label(format!("{}", metrics.tracked_no_object));
                    });
                });
        }

        // Current frame evaluation
        if let Some(frame_eval) = app.get_current_frame_eval() {
            egui::CollapsingHeader::new("Current Frame")
                .default_open(false)
                .show(ui, |ui| {
                    let correct = frame_eval.iter().filter(|(_, _, is_correct)| *is_correct).count();
                    let incorrect = frame_eval.iter().filter(|(_, gt_id, is_correct)| gt_id.is_some() && !is_correct).count();
                    let no_gt = frame_eval.iter().filter(|(_, gt_id, _)| gt_id.is_none()).count();

                    ui.colored_label(Color32::GREEN, format!("Correct: {}", correct));
                    ui.colored_label(Color32::RED, format!("Wrong ID: {}", incorrect));
                    ui.colored_label(Color32::GRAY, format!("No GT: {}", no_gt));
                });
        }

        ui.separator();
    }

    ui.heading("Playback");
    ui.label("FPS:");
    ui.add(egui::Slider::new(&mut app.playback_fps, 0.01..=60.0).logarithmic(true));

    ui.separator();

    // Show re-run button at top if params changed
    let params_changed = app.tracker_params != app.last_tracker_params;
    let classes_changed = app.overlay_settings.enabled_classes != app.last_enabled_classes;
    let min_conf_changed = app.overlay_settings.detection_min_confidence != app.last_detection_min_confidence;
    if (params_changed || classes_changed || min_conf_changed) && app.clip.is_some() {
        let label = if min_conf_changed && !params_changed && !classes_changed {
            "Re-run Tracker (min confidence changed)"
        } else if classes_changed && !params_changed {
            "Re-run Tracker (class filter changed)"
        } else {
            "Re-run Tracker (params changed)"
        };
        if ui.button(label).clicked() {
            app.run_tracker();
        }
        ui.separator();
    }

    egui::CollapsingHeader::new("Core Parameters")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Tracker FPS (0=native):");
            ui.add(egui::Slider::new(&mut app.tracker_params.tracker_fps, 0.0..=60.0));
            ui.separator();
            ui.label("Track Thresh (min conf):");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_thresh, 0.0..=1.0));
            ui.label("High Thresh (new track):");
            ui.add(egui::Slider::new(&mut app.tracker_params.high_thresh, 0.0..=1.0));
            ui.label("Track Buffer (seconds):");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_buffer_secs, 0.1..=10.0));
        });

    egui::CollapsingHeader::new("Matching IoU")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut app.tracker_params.use_ciou, "Use CIoU");
            ui.separator();
            ui.label("High-conf match min IoU:");
            ui.add(egui::Slider::new(&mut app.tracker_params.high_conf_match_min_iou, 0.0..=1.0));
            ui.label("High-conf IoU weight:");
            ui.add(egui::Slider::new(&mut app.tracker_params.high_conf_match_iou_weight, 0.0..=2.0));
            ui.separator();
            ui.label("Low-conf match min IoU:");
            ui.add(egui::Slider::new(&mut app.tracker_params.low_conf_match_min_iou, 0.0..=1.0));
            ui.label("Low-conf IoU weight:");
            ui.add(egui::Slider::new(&mut app.tracker_params.low_conf_match_iou_weight, 0.0..=2.0));
            ui.separator();
            ui.label("Track activation min IoU:");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_activation_min_iou, 0.0..=1.0));
            ui.label("Track activation IoU weight:");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_activation_iou_weight, 0.0..=2.0));
        });

    egui::CollapsingHeader::new("Kalman Filter")
        .default_open(false)
        .show(ui, |ui| {
            ui.label("Std weight pos:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_weight_pos, 0.001..=0.5).logarithmic(true));
            ui.label("Std weight vel:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_weight_vel, 0.001..=0.5).logarithmic(true));
            ui.separator();
            ui.label("Std pos measurement:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_weight_position_meas, 0.001..=0.5).logarithmic(true));
            ui.label("Std pos motion:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_weight_position_mot, 0.001..=0.5).logarithmic(true));
            ui.label("Std vel motion:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_weight_velocity_mot, 0.001..=0.5).logarithmic(true));
            ui.separator();
            ui.label("Std aspect init:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_aspect_ratio_init, 1e-6..=1.0).logarithmic(true));
            ui.label("Std d_aspect init:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_d_aspect_ratio_init, 1e-8..=1e-2).logarithmic(true));
            ui.label("Std aspect motion:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_aspect_ratio_mot, 1e-6..=1.0).logarithmic(true));
            ui.label("Std d_aspect motion:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_d_aspect_ratio_mot, 1e-8..=1e-2).logarithmic(true));
            ui.label("Std aspect meas:");
            ui.add(egui::Slider::new(&mut app.tracker_params.kalman_std_aspect_ratio_meas, 1e-4..=1.0).logarithmic(true));
        });

    ui.separator();

    ui.heading("Keyboard Shortcuts");
    ui.label("Space: Play/Pause");
    ui.label("Left/Right: Step frame");
    ui.label("Home/End: Jump to start/end");

    ui.separator();

    // Track Debug Panel
    egui::CollapsingHeader::new("Track Debug")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Track ID:");
                let response = ui.text_edit_singleline(&mut app.debug_track_id_input);
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(id) = app.debug_track_id_input.parse::<usize>() {
                        app.debug_track_id = Some(id);
                    }
                }
                if ui.button("Inspect").clicked() {
                    if let Ok(id) = app.debug_track_id_input.parse::<usize>() {
                        app.debug_track_id = Some(id);
                    }
                }
            });

            // If hovering a track, show option to inspect it
            if let Some(hovered_id) = app.hovered_track_id {
                if ui.button(format!("Inspect hovered track {}", hovered_id)).clicked() {
                    app.debug_track_id_input = hovered_id.to_string();
                    app.debug_track_id = Some(hovered_id);
                }
            }

            if let Some(track_id) = app.debug_track_id {
                ui.separator();

                let log = app.get_track_debug_log(track_id);

                // Copy to clipboard button
                if ui.button("Copy to Clipboard").clicked() {
                    ui.output_mut(|o| o.copied_text = log.clone());
                }

                // Save to file button
                if ui.button("Save to File").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name(&format!("track_{}_frame_{}.txt", track_id, app.current_frame))
                        .save_file()
                    {
                        let _ = std::fs::write(path, &log);
                    }
                }

                // Display the log in a scrollable area
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        ui.add(egui::TextEdit::multiline(&mut log.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY));
                    });
            }
        });
}

fn render_timeline(app: &mut VisualizerApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let frame_count = app.frame_count();
    if frame_count == 0 {
        ui.label("No video loaded");
        return;
    }

    // Top row: playback controls
    ui.horizontal(|ui| {
        // Play/Pause button
        let play_text = if app.is_playing { "⏸" } else { "▶" };
        if ui.button(play_text).clicked() {
            app.is_playing = !app.is_playing;
            if app.is_playing {
                app.last_frame_time = Some(std::time::Instant::now());
            }
        }

        // Frame counter
        ui.label(format!("Frame: {} / {}", app.current_frame + 1, frame_count));

        // Timestamp display (use original frame index)
        let original_frame_idx = app.get_original_frame_idx(app.current_frame);
        if let Some(clip) = &app.clip {
            if let Some(ts) = clip.get_timestamp(original_frame_idx) {
                let secs = ts / 1000;
                let mins = secs / 60;
                let secs = secs % 60;
                ui.label(format!("{}:{:02}", mins, secs));
            }
        }

        // Detection count for current frame (use original frame index)
        if let Some(clip) = &app.clip {
            let det_count = clip.get_detections(original_frame_idx).len();
            ui.label(format!("{} det", det_count));
        }
    });

    // Draw time series plot if eval overlays are enabled
    let pred_timeseries = app.get_gt_timeseries();
    if app.overlay_settings.show_eval_overlays && !pred_timeseries.is_empty() {
        let current_frame = app.get_current_frame();
        let total_frames = app.get_total_frames();

        // Allocate remaining space for the plot
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click());
        let rect = response.rect;

        // Draw the time series in the timeline panel
        draw_timeline_timeseries(
            &painter,
            rect,
            pred_timeseries,
            current_frame,
            total_frames,
        );

        // Handle click to seek
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let label_width = 50.0;
                let margin = 10.0;
                let plot_left = rect.min.x + label_width;
                let plot_width = rect.width() - label_width - margin;

                if pos.x >= plot_left && pos.x <= plot_left + plot_width {
                    let normalized = (pos.x - plot_left) / plot_width;
                    let target_frame = (normalized * total_frames as f32).round() as usize;
                    let target_frame = target_frame.min(total_frames.saturating_sub(1));
                    app.current_frame = target_frame;
                    app.mechanics_stage = 0;
                    app.is_playing = false;
                }
            }
        }
    } else {
        // Simple timeline scrubber when no eval data
        let mut frame = app.current_frame;
        if ui.add(egui::Slider::new(&mut frame, 0..=frame_count.saturating_sub(1))
            .show_value(false))
            .changed()
        {
            app.current_frame = frame;
            app.mechanics_stage = 0;
        }
    }

    // Request repaint if playing
    if app.is_playing {
        ctx.request_repaint();
    }
}

/// Draw time series plot in the timeline panel
fn draw_timeline_timeseries(
    painter: &egui::Painter,
    rect: egui::Rect,
    pred_timeseries: &std::collections::HashMap<u64, Vec<(usize, Option<u64>)>>,
    current_frame: usize,
    total_frames: usize,
) {
    if pred_timeseries.is_empty() || total_frames == 0 {
        return;
    }

    // Collect all unique GT IDs for Y-axis scale
    let mut all_gt_ids: Vec<u64> = pred_timeseries
        .values()
        .flat_map(|series| series.iter().filter_map(|(_, gt)| *gt))
        .collect();
    all_gt_ids.sort();
    all_gt_ids.dedup();

    if all_gt_ids.is_empty() {
        return;
    }

    let label_width = 50.0;
    let margin = 10.0;

    // Semi-transparent background
    painter.rect_filled(rect, 0.0, Color32::from_black_alpha(220));

    let plot_left = rect.min.x + label_width;
    let plot_width = rect.width() - label_width - margin;
    let plot_top = rect.min.y + margin;
    let plot_bottom = rect.max.y - margin;
    let plot_inner_height = plot_bottom - plot_top;

    // Map GT ID to Y position
    let gt_to_y = |gt_id: u64| -> f32 {
        let idx = all_gt_ids.iter().position(|&id| id == gt_id).unwrap_or(0);
        let normalized = if all_gt_ids.len() > 1 {
            idx as f32 / (all_gt_ids.len() - 1) as f32
        } else {
            0.5
        };
        plot_bottom - normalized * plot_inner_height
    };

    // Map frame to X position
    let frame_to_x = |frame: usize| -> f32 {
        plot_left + (frame as f32 / total_frames as f32) * plot_width
    };

    // Draw Y-axis labels (GT IDs)
    for &gt_id in &all_gt_ids {
        let y = gt_to_y(gt_id);
        painter.text(
            egui::Pos2::new(rect.min.x + 5.0, y - 6.0),
            egui::Align2::LEFT_TOP,
            format!("{}", gt_id),
            egui::FontId::monospace(10.0),
            Color32::GRAY,
        );
        // Horizontal grid line
        painter.line_segment(
            [egui::Pos2::new(plot_left, y), egui::Pos2::new(plot_left + plot_width, y)],
            egui::Stroke::new(0.5, Color32::from_gray(60)),
        );
    }

    // Generate colors for pred_ids
    let pred_id_color = |pred_id: u64| -> Color32 {
        drawing::track_color(pred_id)
    };

    // Draw each pred_id's line
    for (&pred_id, series) in pred_timeseries {
        let color = pred_id_color(pred_id);

        // Draw line segments between consecutive points where gt_id is Some
        let mut last_point: Option<egui::Pos2> = None;

        for &(frame_idx, gt_id_opt) in series {
            if let Some(gt_id) = gt_id_opt {
                let x = frame_to_x(frame_idx);
                let y = gt_to_y(gt_id);
                let point = egui::Pos2::new(x, y);

                // Draw line from last point
                if let Some(last) = last_point {
                    painter.line_segment([last, point], egui::Stroke::new(2.0, color));
                }

                // Draw point marker
                painter.circle_filled(point, 3.0, color);

                last_point = Some(point);
            } else {
                // GT is None - break the line
                last_point = None;
            }
        }
    }

    // Draw current frame indicator (vertical white line)
    let cursor_x = frame_to_x(current_frame);
    painter.line_segment(
        [
            egui::Pos2::new(cursor_x, plot_top),
            egui::Pos2::new(cursor_x, plot_bottom),
        ],
        egui::Stroke::new(2.0, Color32::WHITE),
    );

    // Label
    painter.text(
        egui::Pos2::new(rect.min.x + 5.0, rect.min.y + 2.0),
        egui::Align2::LEFT_TOP,
        "GT ID",
        egui::FontId::monospace(10.0),
        Color32::WHITE,
    );

    // X-axis label
    painter.text(
        egui::Pos2::new(plot_left + plot_width / 2.0, rect.max.y - 5.0),
        egui::Align2::CENTER_BOTTOM,
        "Time (frames) - click to seek",
        egui::FontId::monospace(9.0),
        Color32::GRAY,
    );
}

fn render_video(app: &mut VisualizerApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    if app.decoder.is_none() {
        ui.centered_and_justified(|ui| {
            ui.label("No video loaded");
        });
        return;
    }

    // Request frame from decoder (using original frame index for sampled playback)
    let requested_frame_idx = app.get_original_frame_idx(app.current_frame);
    let decoder = app.decoder.as_mut().unwrap();
    let frame_result = decoder.request_frame(requested_frame_idx);

    if let Some(frame) = frame_result {
        // Track which frame is actually in the texture
        app.displayed_original_frame = Some(frame.frame_idx);

        // Update texture
        let width = app.video_width as usize;
        let height = app.video_height as usize;

        // Convert u32 ARGB to RGBA bytes
        let pixels: Vec<Color32> = frame
            .frame
            .iter()
            .map(|&argb| {
                let b = (argb & 0xFF) as u8;
                let g = ((argb >> 8) & 0xFF) as u8;
                let r = ((argb >> 16) & 0xFF) as u8;
                let a = ((argb >> 24) & 0xFF) as u8;
                Color32::from_rgba_unmultiplied(r, g, b, a)
            })
            .collect();

        let image = ColorImage {
            size: [width, height],
            pixels,
        };

        app.video_texture = Some(ctx.load_texture(
            "video_frame",
            image,
            TextureOptions::LINEAR,
        ));
    }

    // Draw the video frame
    if let Some(texture) = &app.video_texture {
        let available_size = ui.available_size();

        // Calculate size maintaining aspect ratio
        let tex_size = texture.size_vec2();
        let scale = (available_size.x / tex_size.x).min(available_size.y / tex_size.y);
        let display_size = tex_size * scale;

        // Center the image
        let offset = (available_size - display_size) / 2.0;

        let (response, painter) = ui.allocate_painter(available_size, egui::Sense::hover());
        let rect = egui::Rect::from_min_size(response.rect.min + offset, display_size);

        // Get mouse position for hover detection
        let mouse_pos = response.hover_pos();

        // Draw video frame
        painter.image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        // Draw overlays for the frame that's actually displayed in the texture
        // This keeps overlays in sync with the video even when decoder returns a fallback frame
        if let Some(clip) = &app.clip {
            if let Some(displayed_original) = app.displayed_original_frame {
                // Find the display index for the frame that's actually shown
                // If this frame isn't in our sampled set, use current_frame as fallback
                let display_idx = app.get_display_idx_for_original(displayed_original)
                    .unwrap_or(app.current_frame);

                app.hovered_track_id = drawing::draw_overlays(
                    &painter,
                    rect,
                    clip,
                    display_idx,            // display index for track results
                    displayed_original,     // original index for detections
                    &app.overlay_settings,
                    app.track_results.as_ref(),
                    app.mechanics_stage,
                    mouse_pos,
                    app.hovered_track_id,
                );

                // Draw evaluation overlays (if ground truth is available)
                if app.overlay_settings.show_eval_overlays {
                    drawing::draw_eval_overlays(
                        &painter,
                        rect,
                        clip.video_width,
                        clip.video_height,
                        &app.overlay_settings,
                        app.get_current_frame_gt(),
                        app.get_current_frame_eval(),
                        app.track_results.as_ref(),
                        display_idx,
                        app.get_assignment(),
                    );
                    drawing::draw_eval_legend(&painter, rect, &app.overlay_settings);
                    // Time series is now drawn in the timeline panel below
                }

                // Draw legend when mechanics visualization is enabled
                if app.overlay_settings.show_bytetrack_mechanics {
                    drawing::draw_mechanics_legend(&painter, rect, app.mechanics_stage);
                }
            }
        }
    }
}
