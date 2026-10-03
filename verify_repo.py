"""Verification script for Assignment-10 repo integrity and README compliance."""
import re
import os
import subprocess
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("ASSIGNMENT-10 FINAL VERIFICATION SUITE")
    print("=" * 60)
    all_passed = True

    # 1. Parse markdown links in README.md
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    link_pattern = re.compile(r'!?\[.*?\]\((?!https?://|mailto:)(.*?)\)')
    local_links = link_pattern.findall(content)
    # Strip any anchors or quotes
    cleaned_links = set()
    for l in local_links:
        clean = l.split("#")[0].strip()
        if clean:
            cleaned_links.add(clean)

    print(f"\n[Check 1] Relative Links in README.md ({len(cleaned_links)} found):")
    missing_links = []
    for l in sorted(cleaned_links):
        exists = os.path.exists(l)
        print(f"  - {l:<40} -> Exists: {exists}")
        if not exists:
            missing_links.append(l)

    if missing_links:
        print(f"  FAIL: Missing files: {missing_links}")
        all_passed = False
    else:
        print("  PASS: All relative links and images exist on disk.")

    # 2. Check git ls-files for forbidden items
    print("\n[Check 2] Git Tracked Files (Forbidden Patterns Check):")
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    tracked_files = res.stdout.splitlines()

    forbidden_patterns = [
        ("PDF files", lambda f: f.lower().endswith(".pdf")),
        ("Dataset images", lambda f: "dataset" in f.lower() and f.lower().endswith((".jpg", ".png"))),
        ("Result masks", lambda f: f.startswith("result/") or f.startswith("result_temporal/")),
        ("TensorBoard events", lambda f: "events.out.tfevents" in f)
    ]

    for label, checker in forbidden_patterns:
        matched = [f for f in tracked_files if checker(f)]
        if matched:
            print(f"  FAIL: Found tracked {label}: {matched[:5]}")
            all_passed = False
        else:
            print(f"  PASS: No {label} tracked in git.")

    # 3. Mermaid & LaTeX syntax check
    print("\n[Check 3] Mermaid & Formula Verification:")
    has_mermaid = "```mermaid" in content
    print(f"  - ```mermaid fence present: {has_mermaid}")
    if not has_mermaid:
        all_passed = False

    # Check for KaTeX broken tokens like \mathcal{L}*
    broken_latex = re.findall(r'\\mathcal\{L\}\*', content)
    if broken_latex:
        print(f"  FAIL: Found broken LaTeX tokens: {broken_latex}")
        all_passed = False
    else:
        print("  PASS: No broken LaTeX tokens found.")

    # 4. Numbers diff check
    print("\n[Check 4] Performance Metrics Accuracy Check:")
    with open("results/metrics.json") as f:
        m_rand = json.load(f)
    with open("results/metrics_temporal.json") as f:
        m_temp = json.load(f)

    s_rand = m_rand["summary"]
    s_temp = m_temp["summary"]

    expected_numbers = [
        ("Random Positive Detected", f"{s_rand['num_detected_positive']} / {s_rand['num_test_images']}", f"{s_rand['num_detected_positive']} / {s_rand['num_test_images']}" in content),
        ("Random Detection Rate", f"{s_rand['detection_rate_pct']:.2f}%", f"{s_rand['detection_rate_pct']:.2f}%" in content),
        ("Random Detected IoU", f"{s_rand['avg_iou_detected_images']:.4f}", f"{s_rand['avg_iou_detected_images']:.4f}" in content),
        ("Random All IoU", f"{s_rand['avg_iou_all_images']:.4f}", f"{s_rand['avg_iou_all_images']:.4f}" in content),
        ("Temporal Positive Detected", f"{s_temp['num_detected_positive']} / {s_temp['num_test_images']}", f"{s_temp['num_detected_positive']} / {s_temp['num_test_images']}" in content),
        ("Temporal Detection Rate", f"{s_temp['detection_rate_pct']:.2f}%", f"{s_temp['detection_rate_pct']:.2f}%" in content),
        ("Temporal Detected IoU", f"{s_temp['avg_iou_detected_images']:.4f}", f"{s_temp['avg_iou_detected_images']:.4f}" in content),
        ("Temporal All IoU", f"{s_temp['avg_iou_all_images']:.4f}", f"{s_temp['avg_iou_all_images']:.4f}" in content),
    ]

    for label, val, matched in expected_numbers:
        print(f"  - {label:<28}: {val:<12} in README -> {matched}")
        if not matched:
            all_passed = False

    # 5. Assignment Brief Requirements Check (5 Items)
    print("\n[Check 5] 5 Required Items of the Brief:")
    items = [
        ("1. Network Architecture + Rationale", any(k in content for k in ["โครงสร้าง Neural Network", "สถาปัตยกรรม", "UNet", "Architectural Rationale"])),
        ("2. Loss Convergence Graph", any(k in content for k in ["assets/loss_curve.png", "Loss Curve"])),
        ("3. Performance Metrics", any(k in content for k in ["ค่าการวัดผลประสิทธิภาพ", "Performance Metrics", "99.71%", "0.9339"])),
        ("4. Before/After Inference Snapshots", any(k in content for k in ["assets/snapshot_comparison.jpg", "snapshot"])),
        ("5. Inference Memory Footprint", any(k in content for k in ["Memory Footprint", "VRAM", "4.71 MB", "RSS"]))
    ]
    for label, found in items:
        print(f"  - {label:<40} -> Found: {found}")
        if not found:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (100% compliant)!")
    else:
        print("SOME VERIFICATION CHECKS FAILED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
