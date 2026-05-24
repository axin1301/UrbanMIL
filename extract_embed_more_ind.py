import json
from pathlib import Path
from typing import Dict

import torch
import timm
from PIL import Image
from tqdm import tqdm


# =========================
# 1. 模型 & 预处理
# =========================
def build_model(model_name="vit_base_patch16_224", device="cuda"):
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    return model, transform


# =========================
# 2. 提取 CLS 特征
# =========================
@torch.no_grad()
def extract_cls(model, image, transform, device="cuda"):
    x = transform(image).unsqueeze(0).to(device)

    feats = model.forward_features(x)

    if feats.dim() == 3:
        cls = feats[:, 0, :]
    else:
        cls = feats

    return cls.squeeze(0).cpu()  # [D]


# =========================
# 3. 主函数
# =========================
def extract_region_features(
    json_path: str,
    out_path: str,
    *,
    indicator: str = "gdp",
    label_field: str = "reference",
    image_root: str = "../dataset/Citylens_images",
    model_name="vit_base_patch16_224",
    device="cuda",
):
    with open(json_path, "r") as f:
        data = json.load(f)

    model, transform = build_model(model_name, device)

    region_features: Dict = {}

    for idx, item in enumerate(tqdm(data, desc="Extracting")):
        # -------- region id --------

        # try:
        if 1:
            region_id = item.get("region_id", f"sample_{idx:06d}")

            image_paths = item["images"]
            assert len(image_paths) == 11, f"{region_id} not 11 images"

            sat_path = image_paths[0]
            street_paths = image_paths[1:]

            # -------- sat --------
            sat_path = str(Path(image_root) / sat_path.split("/")[-1])
            sat_img = Image.open(sat_path).convert("RGB")
            sat_feat = extract_cls(model, sat_img, transform, device)

            # -------- street --------
            street_feats = []
            for p in street_paths:
                p = str(Path(image_root) / p.split("/")[-1])
                img = Image.open(p).convert("RGB")
                feat = extract_cls(model, img, transform, device)
                street_feats.append(feat)

            street_feats = torch.stack(street_feats, dim=0)  # [10, D]

            # -------- label --------
            gdp = float(item["reference"]) # 绝对值

            # -------- 存 --------
            if label_field == "reference":
                y = float(gdp)
            else:
                if label_field not in item:
                    raise KeyError(f"Missing label_field={label_field!r} in item keys: {list(item.keys())}")
                y = float(item[label_field])

            payload = {
                "sat": sat_feat,          # [D]
                "street": street_feats,   # [10, D]
                "y": y,                   # raw label
                "indicator": str(indicator),
                str(indicator): y,
            }
            if str(indicator) == "gdp":
                payload["gdp"] = float(gdp)
            region_features[region_id] = payload
        # except:
        #     continue

    torch.save(region_features, out_path)

    print(f"\nSaved to {out_path}")
    print(f"Total regions: {len(region_features)}")

    # 打印一个例子
    k = list(region_features.keys())[0]
    print("\nExample:")
    print(k)
    print(region_features[k]["sat"].shape)
    print(region_features[k]["street"].shape)


# =========================
# 4. CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--indicator", type=str, default="gdp", help="Indicator name used for output key and default filename.")
    parser.add_argument("--label_field", type=str, default="reference", help="Field in input json used as raw label.")
    parser.add_argument("--image_root", type=str, default="../dataset/Citylens_images", help="Resolve image paths as image_root/<basename>.")
    parser.add_argument("--out_path", type=str, default=None, help="Output .pt path. If omitted, uses ../outputs/region_features_{indicator}.pt")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = args.out_path
    if not out_path:
        out_path = str(Path("../outputs") / f"region_features_{args.indicator}.pt")

    extract_region_features(
        json_path=args.json_path,
        out_path=out_path,
        indicator=args.indicator,
        label_field=args.label_field,
        image_root=args.image_root,
        model_name=args.model_name,
        device=args.device,
    )

    # --json_path  ../dataset/UrbanSensing_data/all_global_gdp_task.json   --out_path  ../outputs/region_features_gdp.pt
