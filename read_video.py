import cv2
import sys

VIDEO_DIR = "Images"
COLOR_VIDEO = f"{VIDEO_DIR}/color_0.mkv"
DEPTH_VIDEO = f"{VIDEO_DIR}/depth_0.mkv"


def play_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps) if fps > 0 else 30

    print(f"Lecture de {path}  -  {fps:.1f} FPS  -  {total_frames} frames")
    print("Controles:")
    print("  ESPACE  : play / pause")
    print("  d       : frame suivante (en pause)")
    print("  a       : frame precedente (en pause)")
    print("  q       : quitter")

    paused = False
    ret, frame = cap.read()
    if not ret:
        print("Erreur: aucune frame a lire")
        return

    while True:
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        display = frame.copy()
        status = "PAUSE" if paused else "PLAY"
        cv2.putText(display, f"{status}  Frame {frame_num}/{total_frames}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(path, display)

        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d") and paused:
            ret, frame = cap.read()
            if not ret:
                break
        elif key == ord("a") and paused:
            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break
        elif not paused:
            ret, frame = cap.read()
            if not ret:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else COLOR_VIDEO
    play_video(video)
