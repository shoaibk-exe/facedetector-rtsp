import argparse
import os
import threading
import time
from datetime import datetime

import cv2

from utils.cameras import (
    Camera,
    discover_mp4_in_folder,
    load_cameras,
    select_cameras,
)
from utils.config import (
    CAMERAS_FILE,
    CLIPS_DIR,
    DATABASE_PATH,
    INPUT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    SAVE_RTSP_OUTPUT,
    SHOW_PREVIEW,
    SOURCE,
    STAFF_CLIP_FPS,
    STAFF_CLIP_POST_FRAMES,
    THRESHOLD,
)
from utils.detection_log import DetectionLogger
from utils.face_engine import FaceEngine
from utils.gallery import load_gallery
from utils.pipeline import DisplayFace, RecognitionPipeline
from utils.rtsp_stream import RtspFrameGrabber, configure_ffmpeg_for_rtsp
from utils.staff_clip import StaffClipRecorder


def draw_pipeline_faces(frame, faces: list[DisplayFace], header: str = ""):
    if header:
        cv2.putText(
            frame,
            header,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    for det in faces:
        x1, y1, x2, y2 = det.bbox
        if det.ghost:
            color = (180, 180, 180)
            thickness = 1
        elif det.confirmed:
            color = (0, 255, 0)
            thickness = 2
        elif det.label.startswith("track"):
            color = (255, 200, 0)
            thickness = 2
        elif det.label.endswith("?"):
            color = (0, 255, 255)
            thickness = 2
        else:
            color = (0, 165, 255)
            thickness = 2

        pct = det.score * 100.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            f"ID{det.track_id} {det.label} {pct:.1f}%",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
    return frame


def detections_for_log(faces: list[DisplayFace]) -> list[dict]:
    rows = []
    for det in faces:
        rows.append(
            {
                "label": det.label,
                "score": det.score,
                "scores": {det.label: det.score},
                "bbox": det.bbox,
                "track_id": det.track_id,
                "confirmed": det.confirmed,
                "samples": det.samples,
                "votes": det.votes,
            }
        )
    return rows


def open_capture(source: str):
    if source.lower().startswith("rtsp"):
        configure_ffmpeg_for_rtsp()
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def process_mp4(
    engine: FaceEngine,
    gallery,
    threshold: float,
    videos: list[Camera],
    output_dir: str,
    show_display: bool = False,
    detection_logger: DetectionLogger | None = None,
    save_staff_clips: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    if not videos:
        print("No MP4 entries selected. Enable items under mp4: in cameras.yaml")
        return

    for video in videos:
        video_path = video.path
        if not os.path.isfile(video_path):
            print(f"\n[{video.id}] SKIP — file not found: {video_path}")
            continue

        print(f"\n[{video.id}] Processing: {video.name} -> {video_path}")
        if show_display:
            print(f"[{video.id}] Display ON — press 'q' to skip to next video")

        pipeline = RecognitionPipeline(gallery=gallery, threshold=threshold)

        cap = open_capture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_name = f"{video.id}_{os.path.basename(video_path)}"
        if not out_name.lower().endswith(".mp4"):
            out_name += ".mp4"
        output_path = os.path.join(output_dir, out_name)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        clip_recorder = None
        if save_staff_clips:
            clip_dir = os.path.join(CLIPS_DIR, video.id)
            clip_recorder = StaffClipRecorder(
                camera_id=video.id,
                output_dir=clip_dir,
                fps=fps,
                post_pad_frames=STAFF_CLIP_POST_FRAMES,
            )

        frame_id = 0
        quit_video = False
        window = f"FR [{video.id}] {video.name}"
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                faces = engine.get_faces(frame)
                display = pipeline.process(frame, faces)
                draw_pipeline_faces(
                    frame, display, header=f"{video.name} ({video.id})"
                )

                if detection_logger is not None:
                    detection_logger.log_frame(
                        camera_id=video.id,
                        camera_name=video.name,
                        frame_id=frame_id,
                        detections=detections_for_log(display),
                    )

                if clip_recorder is not None:
                    clip_recorder.update(
                        frame, pipeline.staff_names_confirmed(display)
                    )

                writer.write(frame)

                if show_display:
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print(f"[{video.id}] Skipped by user (q)")
                        quit_video = True
                        break
        finally:
            if show_display:
                cv2.destroyWindow(window)
            if clip_recorder is not None:
                clip_recorder.close()
            cap.release()
            writer.release()
            if quit_video:
                print(f"[{video.id}] Partial save: {output_path}")
            else:
                print(f"[{video.id}] Saved: {output_path}")


class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera: Camera,
        engine: FaceEngine,
        engine_lock: threading.Lock,
        pipeline: RecognitionPipeline,
        output_dir: str,
        save_output: bool,
        stop_event: threading.Event,
        detection_logger: DetectionLogger | None = None,
        save_staff_clips: bool = False,
    ):
        super().__init__(daemon=True, name=f"cam-{camera.id}")
        self.camera = camera
        self.engine = engine
        self.engine_lock = engine_lock
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.save_output = save_output
        self.stop_event = stop_event
        self.detection_logger = detection_logger
        self.save_staff_clips = save_staff_clips

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.error: str | None = None
        self.frame_count = 0

    def run(self):
        cam = self.camera
        print(f"[{cam.id}] Connecting: {cam.masked_url()}")

        grabber = RtspFrameGrabber(cam.source, cam.id, self.stop_event)
        grabber.start()

        writer = None
        clip_recorder = None
        last_seq = -1
        deadline = time.time() + 15.0
        while not self.stop_event.is_set() and time.time() < deadline:
            frame, seq = grabber.get_latest()
            if frame is not None:
                break
            if grabber.error:
                self.error = grabber.error
                print(f"[{cam.id}] ERROR: {grabber.error}")
                return
            time.sleep(0.05)
        else:
            if not self.stop_event.is_set():
                print(f"[{cam.id}] Waiting for first frame...")

        sample, _ = grabber.get_latest()
        height, width = (720, 1280)
        if sample is not None:
            height, width = sample.shape[:2]

        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.output_dir, f"{cam.id}_{stamp}.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                STAFF_CLIP_FPS,
                (width, height),
            )
            print(f"[{cam.id}] Recording full stream -> {out_path}")

        if self.save_staff_clips:
            clip_recorder = StaffClipRecorder(
                camera_id=cam.id,
                output_dir=os.path.join(CLIPS_DIR, cam.id),
                fps=STAFF_CLIP_FPS,
                post_pad_frames=STAFF_CLIP_POST_FRAMES,
            )
            print(f"[{cam.id}] Staff-clip saver ON -> {CLIPS_DIR}/{cam.id}/")

        print(f"[{cam.id}] Pipeline started ({cam.name})")

        try:
            while not self.stop_event.is_set():
                if grabber.error:
                    frame, _ = grabber.get_latest(copy=False)
                    if frame is None:
                        self.error = grabber.error
                        time.sleep(0.5)
                        continue

                frame, seq = grabber.get_latest()
                if frame is None or seq == last_seq:
                    time.sleep(0.01)
                    continue

                last_seq = seq

                with self.engine_lock:
                    faces = self.engine.get_faces(frame)

                display = self.pipeline.process(frame, faces)
                draw_pipeline_faces(
                    frame, display, header=f"{cam.name} ({cam.id})"
                )

                self.frame_count += 1

                if self.detection_logger is not None:
                    self.detection_logger.log_frame(
                        camera_id=cam.id,
                        camera_name=cam.name,
                        frame_id=self.frame_count,
                        detections=detections_for_log(display),
                    )

                if clip_recorder is not None:
                    clip_recorder.update(
                        frame, self.pipeline.staff_names_confirmed(display)
                    )

                with self.frame_lock:
                    self.latest_frame = frame

                if writer is not None:
                    writer.write(frame)

                if self.frame_count % 90 == 0:
                    confirmed = self.pipeline.staff_names_confirmed(display)
                    print(
                        f"[{cam.id}] frames={self.frame_count} "
                        f"faces={len(faces)} confirmed={confirmed} "
                        f"skipped={grabber.skipped} "
                        f"reconnects={grabber.reconnects}"
                    )
        except Exception as exc:
            self.error = str(exc)
            print(f"[{cam.id}] Worker crashed: {exc}")
        finally:
            if clip_recorder is not None:
                clip_recorder.close()
            if writer is not None:
                writer.release()
            grabber.join(timeout=3.0)
            print(f"[{cam.id}] Stopped")

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


def process_rtsp_cameras(
    cameras: list[Camera],
    engine: FaceEngine,
    gallery,
    threshold: float,
    output_dir: str,
    show_display: bool,
    save_output: bool,
    detection_logger: DetectionLogger | None = None,
    save_staff_clips: bool = False,
):
    if not cameras:
        raise SystemExit("No cameras selected.")

    print("\n========== CAMERAS ==========")
    for cam in cameras:
        print(f"  {cam.id:<12} {cam.name:<20} {cam.masked_url()}")
    print("=============================\n")

    stop_event = threading.Event()
    engine_lock = threading.Lock()

    # One pipeline (tracker) per camera
    workers = []
    for cam in cameras:
        pipeline = RecognitionPipeline(gallery=gallery, threshold=threshold)
        workers.append(
            CameraWorker(
                camera=cam,
                engine=engine,
                engine_lock=engine_lock,
                pipeline=pipeline,
                output_dir=output_dir,
                save_output=save_output,
                stop_event=stop_event,
                detection_logger=detection_logger,
                save_staff_clips=save_staff_clips,
            )
        )

    for worker in workers:
        worker.start()

    if show_display:
        print("Display ON — press 'q' in any window to quit.")
    else:
        print("Display OFF — detection only. Press Ctrl+C to quit.")

    try:
        while any(worker.is_alive() for worker in workers):
            if show_display:
                for worker in workers:
                    frame = worker.get_frame()
                    if frame is None:
                        continue
                    window = f"FR [{worker.camera.id}] {worker.camera.name}"
                    cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit requested from display window.")
                    break
            else:
                time.sleep(0.5)

            if all(not w.is_alive() for w in workers):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_event.set()
        for worker in workers:
            try:
                worker.join(timeout=3.0)
            except KeyboardInterrupt:
                pass
        if show_display:
            cv2.destroyAllWindows()


def parse_camera_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Staff face recognition (quality + track + multi-embed + vote)."
        )
    )
    parser.add_argument(
        "--source",
        choices=["mp4", "rtsp"],
        default=None,
        help="mp4 = batch files, rtsp = cameras from cameras.yaml",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera ids (default: all enabled)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List cameras from cameras.yaml and exit",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="RTSP: detection + live preview",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="RTSP: detection only",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override match threshold from .env",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save full annotated RTSP stream per camera",
    )
    parser.add_argument(
        "--save-staff",
        action="store_true",
        help="Save clip when a staff identity is confirmed",
    )
    parser.add_argument(
        "--log-detections",
        action="store_true",
        help="Write per-frame detection log with confidence percent",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_display(args) -> bool:
    if args.display and (args.no_display or args.no_preview):
        raise SystemExit("Use either --display or --no-display, not both.")
    if args.display:
        return True
    if args.no_display or args.no_preview:
        return False
    return SHOW_PREVIEW


def main():
    args = parse_args()

    if args.list_cameras:
        print("=== RTSP ===")
        try:
            for cam in load_cameras("rtsp"):
                flag = "yes" if cam.enabled else "no"
                print(f"  {cam.id:<12} {flag:<8} {cam.name:<20} {cam.masked_url()}")
        except Exception as exc:
            print(f"  (none) {exc}")
        print("=== MP4 ===")
        try:
            for cam in load_cameras("mp4"):
                flag = "yes" if cam.enabled else "no"
                print(f"  {cam.id:<12} {flag:<8} {cam.name:<20} {cam.path}")
        except Exception as exc:
            print(f"  (none) {exc}")
        return

    source = (args.source or SOURCE).lower()
    threshold = args.threshold if args.threshold is not None else THRESHOLD
    show_display = resolve_display(args)
    save_output = True if args.save else SAVE_RTSP_OUTPUT
    save_staff_clips = bool(args.save_staff)
    log_detections = bool(args.log_detections)

    if source not in {"mp4", "rtsp"}:
        raise SystemExit(f"Invalid source '{source}'. Use mp4 or rtsp.")

    print(f"Mode: {source}")
    print(f"Threshold: {threshold}  (below → Unknown)")
    print(f"Save staff clips: {'ON' if save_staff_clips else 'OFF'}")
    print(f"Detection log: {'ON' if log_detections else 'OFF'}")

    gallery = load_gallery(DATABASE_PATH)
    print(
        "Staff gallery: "
        + ", ".join(f"{n}[{v.shape[0]}]" for n, v in gallery.items())
    )

    detection_logger = None
    if log_detections:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"detections_{stamp}.log")
        detection_logger = DetectionLogger(log_path)

    engine = FaceEngine()

    config_path = CAMERAS_FILE
    print(f"Config file: {config_path}  (edit this — NOT README.md)")

    all_sources = load_cameras(source)
    camera_ids = parse_camera_ids(args.cameras)

    try:
        selected = select_cameras(all_sources, camera_ids, source=source)
    except ValueError as exc:
        if source == "mp4" and not camera_ids:
            discovered = discover_mp4_in_folder(INPUT_DIR)
            if discovered:
                print(
                    f"No enabled mp4: in cameras.yaml — "
                    f"using {len(discovered)} file(s) from {INPUT_DIR}/"
                )
                selected = discovered
            else:
                raise SystemExit(str(exc)) from exc
        else:
            raise SystemExit(str(exc)) from exc

    print(f"Selected ({source}): {[c.id for c in selected]}")
    for item in selected:
        loc = item.path if item.kind == "mp4" else item.masked_url()
        print(f"  {item.id}: {item.name} -> {loc}")

    if source == "mp4":
        print(f"Display: {'ON' if show_display else 'OFF'}")
        process_mp4(
            engine,
            gallery,
            threshold,
            selected,
            OUTPUT_DIR,
            show_display=show_display,
            detection_logger=detection_logger,
            save_staff_clips=save_staff_clips,
        )
    else:
        print(f"Display: {'ON' if show_display else 'OFF'}")
        process_rtsp_cameras(
            selected,
            engine,
            gallery,
            threshold,
            OUTPUT_DIR,
            show_display,
            save_output,
            detection_logger=detection_logger,
            save_staff_clips=save_staff_clips,
        )

    print("\nDone")


if __name__ == "__main__":
    main()
