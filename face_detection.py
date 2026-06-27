"""Webcam face detection with a loading bar."""

from __future__ import annotations

import time

import cv2

LOADING_DURATION = 2.0
RESET_DELAY = 2.0
WINDOW_TITLE = "Face Detection with Loading Bar"


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return classifier


def draw_loading_bar(frame, x: int, y: int, w: int, h: int, progress: float) -> None:
    bar_height = 20
    bar_y = y + h + 5

    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_height), (0, 0, 0), -1)
    cv2.rectangle(
        frame,
        (x, bar_y),
        (x + int(w * progress), bar_y + bar_height),
        (0, 255, 0),
        -1,
    )


def draw_completed_face(frame, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Scan Complete",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )


def run(camera_index: int = 0) -> None:
    face_cascade = create_face_detector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check that a camera is connected and not in use."
        )

    loading_start_times: dict[tuple[int, int, int, int], float] = {}
    completed_faces: set[tuple[int, int, int, int]] = set()
    frozen_faces: dict[tuple[int, int, int, int], float] = {}
    reset_time: float | None = None
    reset_in_progress = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            current_time = time.time()

            if reset_in_progress:
                if reset_time is not None and current_time - reset_time >= RESET_DELAY:
                    loading_start_times.clear()
                    completed_faces.clear()
                    frozen_faces.clear()
                    reset_time = None
                    reset_in_progress = False
            else:
                all_faces_frozen = True

                for (x, y, w, h) in faces:
                    face_id = (x, y, w, h)

                    if face_id not in loading_start_times:
                        loading_start_times[face_id] = current_time

                    elapsed_time = current_time - loading_start_times[face_id]

                    if elapsed_time < LOADING_DURATION:
                        all_faces_frozen = False
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        progress = min(1.0, elapsed_time / LOADING_DURATION)
                        draw_loading_bar(frame, x, y, w, h, progress)
                    else:
                        if face_id not in completed_faces:
                            completed_faces.add(face_id)
                        draw_completed_face(frame, x, y, w, h)
                        frozen_faces[face_id] = current_time + RESET_DELAY

                faces_on_screen = {(x, y, w, h) for (x, y, w, h) in faces}
                for face_id in list(completed_faces):
                    if face_id not in faces_on_screen:
                        completed_faces.discard(face_id)
                        loading_start_times.pop(face_id, None)
                        frozen_faces.pop(face_id, None)

                if faces.size > 0 and all_faces_frozen:
                    cv2.putText(
                        frame,
                        "All scans complete",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    if reset_time is None:
                        reset_time = current_time
                        reset_in_progress = True

            for face_id, end_time in list(frozen_faces.items()):
                if current_time < end_time:
                    x, y, w, h = face_id
                    draw_completed_face(frame, x, y, w, h)

            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
