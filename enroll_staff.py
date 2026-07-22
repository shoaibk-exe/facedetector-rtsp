"""
Multi-shot staff enrollment.

Stores many L2-normalized embeddings per person.
No quality filtering — every detected face is enrolled.
"""

import os

import cv2
import numpy as np

from utils.config import DATABASE_PATH, MAX_EMBEDS_PER_PERSON, STAFF_DIR
from utils.face_engine import FaceEngine
from utils.gallery import save_gallery
from utils.quality import get_normed_embedding


def main():
    engine = FaceEngine()
    gallery = {}

    if not os.path.isdir(STAFF_DIR):
        raise SystemExit(f"Staff directory not found: {STAFF_DIR}")

    print("Enrollment: no quality filter — all detected faces are kept.\n")

    for person_name in sorted(os.listdir(STAFF_DIR)):
        person_path = os.path.join(STAFF_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        kept = 0
        skipped = 0

        for image_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            faces = engine.get_faces(img)
            if not faces:
                skipped += 1
                continue

            # Largest face in the image
            face = max(
                faces,
                key=lambda f: float(
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                ),
            )
            emb = get_normed_embedding(face)
            if emb is None:
                skipped += 1
                continue

            embeddings.append(emb)
            kept += 1

        if not embeddings:
            print(f"Skipped {person_name}: no faces found (skipped={skipped})")
            continue

        # Cap gallery size (keep first N if over limit)
        selected = embeddings[:MAX_EMBEDS_PER_PERSON]
        gallery[person_name] = np.stack(selected, axis=0)
        print(
            f"Enrolled {person_name}: {len(selected)} embedding(s) "
            f"(kept={kept}, no_face_or_unreadable={skipped})"
        )

    if not gallery:
        raise SystemExit("No staff embeddings created. Check staff/ images.")

    save_gallery(DATABASE_PATH, gallery)
    print(f"\nEnrollment complete -> {DATABASE_PATH}")
    print(f"Staff: {list(gallery.keys())}")
    print("Format: multiple normed embeddings per person (max-similarity match).")


if __name__ == "__main__":
    main()
