from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_model(
    model_name_or_path: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    local_files_only: bool = False,
):
    """Build a pretrained BERT-style sequence classification model."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=local_files_only,
        use_safetensors=True,
    )
