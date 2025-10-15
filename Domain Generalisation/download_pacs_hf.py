import os
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def setup_pacs_from_hf(data_dir="./data"):
    """
    Downloads flwrlabs/PACS from Hugging Face and saves it
    in DomainBed-style layout: PACS/<domain>/<class>/*.jpg
    """
    hf_dataset_name = "flwrlabs/PACS"
    pacs_base_dir = os.path.join(data_dir, "PACS")
    os.makedirs(pacs_base_dir, exist_ok=True)

    print(f"Loading '{hf_dataset_name}' from Hugging Face Hub...")
    try:
        ds = load_dataset(hf_dataset_name)  # returns {"train": Dataset}
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    dset = ds["train"]
    # Try to grab label names if present
    label_names = None
    try:
        label_names = dset.features["label"].names
    except Exception:
        pass

    counters = defaultdict(int)
    total = 0

    print("Dataset loaded. Now saving images to the required folder structure...")

    for item in tqdm(dset, desc="Saving PACS images"):
        # Domain (expected values: art_painting, cartoon, photo, sketch)
        domain_name = item.get("domain")
        if not domain_name:
            # If domain is missing, skip or default
            continue

        # Class name resolution
        if "label_name" in item and item["label_name"] is not None:
            class_name = str(item["label_name"])
        elif "category" in item and item["category"] is not None:
            class_name = str(item["category"])
        elif "class" in item and item["class"] is not None:
            class_name = str(item["class"])
        elif "label" in item and item["label"] is not None and label_names:
            class_name = str(label_names[item["label"]])
        else:
            # Fallback: raw label id as string
            class_name = str(item.get("label", "unknown"))

        # Image id / filename
        image_id = item.get("image_id")
        if image_id is None:
            # Make a deterministic-ish fallback
            image_id = f"{total:07d}"

        # PIL image
        image = item["image"]
        if isinstance(image, Image.Image) and image.mode == "RGBA":
            image = image.convert("RGB")

        # Paths
        class_path = os.path.join(pacs_base_dir, domain_name, class_name)
        os.makedirs(class_path, exist_ok=True)
        save_path = os.path.join(class_path, f"{image_id}.jpg")

        try:
            image.save(save_path)
            counters[domain_name] += 1
            total += 1
        except Exception as e:
            print(f"⚠️ Failed to save {save_path}: {e}")

    # Summary
    if total == 0:
        print("\n⚠️ No files were saved. This should not happen with the corrected script.")
    else:
        print("\n✅ Finished.")
        for d in sorted(counters):
            print(f"- {d}: {counters[d]} images")
        print(f"The PACS dataset is now at: '{pacs_base_dir}'")
        
if __name__ == "__main__":
    setup_pacs_from_hf()