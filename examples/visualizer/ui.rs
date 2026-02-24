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
            if ui.button("Open Folder...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    app.load_clip(&path);
                }
            }

            ui.separator();

            if let Some(clip) = &app.clip {
                ui.label(format!(
                    "Video: {}x{} | {} frames",
                    clip.video_width, clip.video_height, clip.frame_count
                ));
            } else if let Some(err) = &app.load_error {
                ui.colored_label(Color32::RED, err);
            } else {
                ui.label("No clip loaded. Pass a directory path or click 'Open Folder'");
            }
        });
    });

    // Left panel with controls
    egui::SidePanel::left("controls_panel")
        .default_width(250.0)
        .show(ctx, |ui| {
            render_controls(app, ui);
        });

    // Bottom panel with timeline
    egui::TopBottomPanel::bottom("timeline_panel")
        .min_height(60.0)
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

    ui.heading("Playback");
    ui.label("FPS:");
    ui.add(egui::Slider::new(&mut app.playback_fps, 0.01..=60.0).logarithmic(true));

    ui.separator();

    // Show re-run button at top if params changed
    let params_changed = app.tracker_params != app.last_tracker_params;
    let classes_changed = app.overlay_settings.enabled_classes != app.last_enabled_classes;
    if params_changed || classes_changed {
        let label = if classes_changed && !params_changed {
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
            ui.label("Track Thresh (min conf):");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_thresh, 0.0..=1.0));
            ui.label("High Thresh (new track):");
            ui.add(egui::Slider::new(&mut app.tracker_params.high_thresh, 0.0..=1.0));
            ui.label("Track Buffer (frames):");
            ui.add(egui::Slider::new(&mut app.tracker_params.track_buffer, 1..=120));
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
}

fn render_timeline(app: &mut VisualizerApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    let frame_count = app.frame_count();
    if frame_count == 0 {
        ui.label("No video loaded");
        return;
    }

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

        // Timeline slider
        let mut frame = app.current_frame;
        let response = ui.add(
            egui::Slider::new(&mut frame, 0..=(frame_count.saturating_sub(1)))
                .show_value(false)
                .trailing_fill(true),
        );
        if response.changed() {
            app.current_frame = frame;
            app.is_playing = false;
            app.mechanics_stage = 0; // Reset to first stage when jumping frames
        }

        // Timestamp display
        if let Some(clip) = &app.clip {
            if let Some(ts) = clip.get_timestamp(app.current_frame) {
                ui.label(format!("{}ms", ts));
            }
        }

        // Detection count for current frame
        if let Some(clip) = &app.clip {
            let det_count = clip.get_detections(app.current_frame).len();
            ui.label(format!("{} detections", det_count));
        }
    });

    // Request repaint if playing
    if app.is_playing {
        ctx.request_repaint();
    }
}

fn render_video(app: &mut VisualizerApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    if app.decoder.is_none() {
        ui.centered_and_justified(|ui| {
            ui.label("No video loaded");
        });
        return;
    }

    // Request frame from decoder
    let decoder = app.decoder.as_mut().unwrap();
    let frame_result = decoder.request_frame(app.current_frame);

    if let Some(frame) = frame_result {
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

        // Draw overlays and update hover state
        if let Some(clip) = &app.clip {
            app.hovered_track_id = drawing::draw_overlays(
                &painter,
                rect,
                clip,
                app.current_frame,
                &app.overlay_settings,
                app.track_results.as_ref(),
                app.mechanics_stage,
                mouse_pos,
                app.hovered_track_id,
            );

            // Draw legend when mechanics visualization is enabled
            if app.overlay_settings.show_bytetrack_mechanics {
                drawing::draw_mechanics_legend(&painter, rect, app.mechanics_stage);
            }
        }
    }
}
