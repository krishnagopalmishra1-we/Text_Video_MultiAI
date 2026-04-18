import sys
sys.path.insert(0, "/app")
from scene_splitter import SceneSplitter
s = SceneSplitter()
scenes = s.split(open("/app/smoke_test_1min.txt").read())
print(len(scenes), "scenes")
for sc in scenes:
    print(f"  scene {sc.scene_id}: {sc.duration:.1f}s — {sc.text[:80]}")
