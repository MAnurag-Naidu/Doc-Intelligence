import json
import shutil
from pathlib import Path
from paddleocr import PPStructureV3

out_dir = Path("output_structure")
shutil.rmtree(out_dir, ignore_errors=True)   # clear old results

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,   # FIX: stop it flipping lines upside down
)

output = pipeline.predict("sample.jpeg", layout_threshold=0.2)

for res in output:
    res.save_to_markdown(save_path=str(out_dir))
    res.save_to_json(save_path=str(out_dir))
    res.save_to_img(save_path=str(out_dir))   # pictures showing detected layout boxes

# 1) What blocks did the layout model find?
json_file = next(out_dir.glob("*.json"))
data = json.loads(json_file.read_text(encoding="utf-8"))
data = data.get("res", data)

print("\n===== Layout blocks detected =====")
for block in data.get("parsing_res_list", []):
    label = block.get("block_label")
    content = (block.get("block_content") or "").replace("\n", " ")
    print(f"[{label}]  {content[:80]}")

# 2) The final Markdown
for md_file in out_dir.glob("*.md"):
    print(f"\n===== {md_file.name} =====\n")
    print(md_file.read_text(encoding="utf-8"))