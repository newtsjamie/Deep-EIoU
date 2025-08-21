import csv
import numpy as np


class ExternalMOTDetections:
    """
    Load MOT-style detection file (one per sequence) and serve per-frame detections.

    Expected input row format per line:
      frame,-1,x,y,w,h,score,*,*

    get(frame_id) returns an array of shape (N, 5): [x1, y1, x2, y2, score]
    with pixel coordinates.
    """

    def __init__(self, mot_path: str) -> None:
        self.frame_to_dets = {}
        with open(mot_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                if len(row) < 7:
                    continue
                try:
                    fid = int(float(row[0]))
                    x = float(row[2]); y = float(row[3])
                    w = float(row[4]); h = float(row[5])
                    s = float(row[6])
                except Exception:
                    continue
                x1, y1, x2, y2 = x, y, x + w, y + h
                self.frame_to_dets.setdefault(fid, []).append([x1, y1, x2, y2, s])
        # DEBUG: show a few frames loaded
        if 1 in self.frame_to_dets:
            try:
                print(f"[ExternalMOT] frame 1 dets: {len(self.frame_to_dets[1])}")
                for k in (30, 50, 100):
                    if k in self.frame_to_dets:
                        print(f"[ExternalMOT] frame {k} dets: {len(self.frame_to_dets[k])}")
            except Exception:
                pass

    def get(self, frame_id: int) -> np.ndarray:
        arr = self.frame_to_dets.get(frame_id, [])
        if not arr:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)


