import io
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch

from PIL import Image
from datasets import load_dataset
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import norm
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


def split_text(text):
    parts = text.split("\n", 1)
    prompt = parts[0].strip()
    ground_truth = parts[1].strip() if len(parts) > 1 else ""
    return prompt, ground_truth


def load_image(img):
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    return img.convert("RGB")


def clean(text):
    return text.split("Assistant:")[-1].strip()


def z_test(low, high):
    low = np.array(low, dtype=float)
    high = np.array(high, dtype=float)

    mean1, mean2 = np.mean(low), np.mean(high)
    std1 = np.std(low, ddof=1) if len(low) > 1 else 0.0
    std2 = np.std(high, ddof=1) if len(high) > 1 else 0.0

    n1, n2 = len(low), len(high)
    pooled_se = np.sqrt((std1 ** 2) / max(n1, 1) + (std2 ** 2) / max(n2, 1))

    if pooled_se < 1e-12:
        return 1.0

    z = (mean1 - mean2) / pooled_se
    return float(np.clip(norm.sf(z), 1e-300, 1.0))


def compute_tpr_at_fpr(y_true, y_score, target_fpr=0.10):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def sample_sets(indices, set_size, n_sets, rng):
    return [rng.sample(indices, set_size) for _ in range(n_sets)]


def partition_sets(indices, set_size, rng):
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[i:i + set_size] for i in range(0, len(shuffled), set_size)]


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data = load_dataset(args.dataset_id)[args.split]
    print("Split:", args.split)
    print("Total samples:", len(data))

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    mpn = SentenceTransformer("all-mpnet-base-v2")
    rouge = Rouge()

    def generate(img, prompt, temperature):
        img = load_image(img)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(text=text, images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        return clean(processor.batch_decode(out, skip_special_tokens=True)[0])

    def rouge_sim(a, b):
        try:
            return float(rouge.get_scores(a, b)[0]["rouge-2"]["f"])
        except Exception:
            return 0.0

    ids = []
    labels = []
    ground_truths = []
    low_texts = []
    high_texts = []
    low_rouges = []
    high_rouges = []

    for i, sample in enumerate(data):
        if args.limit is not None and i >= args.limit:
            break

        prompt, ground_truth = split_text(sample["text"])
        if not ground_truth:
            continue

        low_text = generate(sample["image"], prompt, args.t_low)
        high_text = generate(sample["image"], prompt, args.t_high)

        ids.append(sample["id"])
        labels.append(sample.get("is_member", None))
        ground_truths.append(ground_truth)
        low_texts.append(low_text)
        high_texts.append(high_text)

        if args.metric == "rouge":
            low_rouges.append(rouge_sim(low_text, ground_truth))
            high_rouges.append(rouge_sim(high_text, ground_truth))

        if args.print_every is not None and len(ids) % args.print_every == 0:
            print(f"Generated {len(ids)} samples")

    print("Finished generation for", len(ids), "samples")

    if args.metric == "mpn":
        print("Computing MPNet similarities...")
        low_emb = mpn.encode(
            low_texts,
            batch_size=args.embed_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        high_emb = mpn.encode(
            high_texts,
            batch_size=args.embed_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        gt_emb = mpn.encode(
            ground_truths,
            batch_size=args.embed_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        low_scores = np.sum(low_emb * gt_emb, axis=1) / (
            np.linalg.norm(low_emb, axis=1) * np.linalg.norm(gt_emb, axis=1) + 1e-12
        )
        high_scores = np.sum(high_emb * gt_emb, axis=1) / (
            np.linalg.norm(high_emb, axis=1) * np.linalg.norm(gt_emb, axis=1) + 1e-12
        )
    else:
        low_scores = np.array(low_rouges, dtype=float)
        high_scores = np.array(high_rouges, dtype=float)

    df = pd.DataFrame({
        "id": ids,
        "label": labels,
        "low": low_scores,
        "high": high_scores
    })

    # VALIDATION:
    if args.split == "validation":
        if df["label"].isna().any():
            raise ValueError("Validation split must have labels.")

        member_idx = df.index[df["label"] == 1].tolist()
        nonmember_idx = df.index[df["label"] == 0].tolist()

        if len(member_idx) < args.set_size or len(nonmember_idx) < args.set_size:
            raise ValueError("Not enough member/non-member samples for set evaluation.")

        rng = random.Random(args.seed)

        member_sets = sample_sets(member_idx, args.set_size, args.n_eval_sets, rng)
        nonmember_sets = sample_sets(nonmember_idx, args.set_size, args.n_eval_sets, rng)

        scores = []
        labels = []

        for group in member_sets:
            low = df.iloc[group]["low"].tolist()
            high = df.iloc[group]["high"].tolist()
            p = z_test(low, high)
            scores.append(-p)
            labels.append(1)

        for group in nonmember_sets:
            low = df.iloc[group]["low"].tolist()
            high = df.iloc[group]["high"].tolist()
            p = z_test(low, high)
            scores.append(-p)
            labels.append(0)

        auc = roc_auc_score(labels, scores)
        tpr10 = compute_tpr_at_fpr(labels, scores, target_fpr=0.10)

        print(f"AUC: {auc:.4f}")
        print(f"TPR@FPR=10%: {tpr10:.4f}")

        pd.DataFrame({
            "score": scores,
            "label": labels
        }).to_csv(os.path.join(args.save_dir, "validation_set_scores.csv"), index=False)

    # TEST: Kaggle submission
    elif args.split == "test":
        idx = list(range(len(df)))
        score_sum = np.zeros(len(df), dtype=float)
        count = np.zeros(len(df), dtype=float)

        for r in range(args.num_rounds):
            rng = random.Random(args.seed + r)
            sets = partition_sets(idx, args.set_size, rng)

            for group in sets:
                if len(group) < 2:
                    continue

                low = df.iloc[group]["low"].tolist()
                high = df.iloc[group]["high"].tolist()

                p = z_test(low, high)
                score = -p

                for i in group:
                    score_sum[i] += score
                    count[i] += 1

        df["score"] = score_sum / np.maximum(count, 1.0)

        submission = df[["id", "score"]].copy()
        submission.to_csv(os.path.join(args.save_dir, "submission.csv"), index=False)
        print("Saved submission.csv")

    else:
        raise ValueError("split must be either 'validation' or 'test'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_id", default="UBC-SLIME/colx585_group_project_data")
    parser.add_argument("--model_id", default="UBC-SLIME/colx_585_vlm")

    parser.add_argument("--split", default="validation")

    parser.add_argument("--t_low", type=float, default=0.1)
    parser.add_argument("--t_high", type=float, default=1.6)

    parser.add_argument("--set_size", type=int, default=5)
    parser.add_argument("--n_eval_sets", type=int, default=1000)
    parser.add_argument("--num_rounds", type=int, default=3)

    parser.add_argument("--metric", choices=["mpn", "rouge"], default="mpn")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--embed_batch_size", type=int, default=64)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_dir", default="./outputs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args(args=[])

    # Examples:
    # args.split = "validation"
    # args.limit = None

    args.split = "test"
    args.limit = None
    run(args)