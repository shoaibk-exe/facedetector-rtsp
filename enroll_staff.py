"""
Multi-shot staff enrollment.

Stores many L2-normalized embeddings per person (angles, lighting, CCTV frames).
Prefer images from the SAME CCTV cameras used at runtime.
"""

import os

import cv2
import numpy as np

from utils.config import DATABASE_PATH, MAX_EMBEDS_PER_PERSON, STAFF_DIR
from utils.face_engine import FaceEngine
from utils.gallery import save_gallery
from utils.quality import get_normed_embedding, score_face


def main():
    engine = FaceEngine()
    gallery = {}

    if not os.path.isdir(STAFF_DIR):
        raise SystemExit(f"Staff directory not found: {STAFF_DIR}")

    print(
        "Enrollment tips: use many CCTV frames per person "
        "(angles, lighting, glasses). Avoid HD selfies only.\n"
    )

    for person_name in sorted(os.listdir(STAFF_DIR)):
        person_path = os.path.join(STAFF_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        kept = 0
        rejected = 0

        for image_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = engine.get_faces(img)
            if not faces:
                rejected += 1
                continue

            # Largest face in the image
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            quality = score_face(img, face)
            if not quality.ok:
                rejected += 1
                continue

            emb = get_normed_embedding(face)
            if emb is None:
                rejected += 1
                continue

            embeddings.append((quality.score, emb))
            kept += 1

        if not embeddings:
            print(f"Skipped {person_name}: no quality faces (rejected={rejected})")
            continue

        # Keep highest-quality embeddings (cap gallery size)
        embeddings.sort(key=lambda item: item[0], reverse=True)
        selected = [emb for _, emb in embeddings[:MAX_EMBEDS_PER_PERSON]]
        gallery[person_name] = np.stack(selected, axis=0)
        print(
            f"Enrolled {person_name}: {len(selected)} embedding(s) "
            f"(kept={kept}, rejected={rejected})"
        )

    if not gallery:
        raise SystemExit("No staff embeddings created. Check staff/ images.")

    save_gallery(DATABASE_PATH, gallery)
    print(f"\nEnrollment complete -> {DATABASE_PATH}")
    print(f"Staff: {list(gallery.keys())}")
    print("Format: multiple normed embeddings per person (max-similarity match).")


if __name__ == "__main__":
    main()
