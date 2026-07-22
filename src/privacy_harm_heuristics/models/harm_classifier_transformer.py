"""Transformer-based multi-label privacy harm classifier.

This module implements a high-fidelity transformer-based classifier for categorizing
privacy harms using state-of-the-art NLP models. Supports multi-label classification
with configurable base models and comprehensive harm taxonomies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    from typing import TYPE_CHECKING

    # Provide lightweight runtime fallbacks while avoiding type-check redefinition warnings.
    if TYPE_CHECKING:
        from torch.utils.data import Dataset, DataLoader  # type: ignore
    else:  # pragma: no cover - minimal dummies

        class nn:  # type: ignore
            class Module:
                pass

            class Linear:
                def __init__(self, *_, **__):
                    pass

            class Dropout:
                def __init__(self, *_, **__):
                    pass

            class BCEWithLogitsLoss:
                def __call__(self, *_, **__):  # type: ignore[no-untyped-call]
                    return 0.0

        class Dataset:  # type: ignore[misc]
            pass

        class DataLoader:  # type: ignore[misc]
            def __init__(self, *_, **__):
                self._items = []

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return 0

    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer-based harm classifier."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    hidden_dropout_prob: float = 0.1
    threshold: float = 0.5
    device: str = "auto"  # auto, cpu, cuda
    gradient_accumulation_steps: int = 1  # High-performance: >1 when scaling batch
    use_mixed_precision: bool = True  # Enable autocast when device supports it


class HarmDataset(Dataset):
    """Dataset class for transformer-based harm classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[List[str]]],
        label_to_idx: Dict[str, int],
    ):
        self.texts = texts
        self.labels = labels if labels is not None else [[] for _ in texts]
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        # Create multi-label tensor
        label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
        for label in labels:
            if label in self.label_to_idx:
                label_tensor[self.label_to_idx[label]] = 1.0

        return text, label_tensor


class HarmCollator:
    """Collator for dynamic padding and batch tokenization."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)

        # Tokenize batch of texts efficiently
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,  # Dynamic padding
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.stack(labels),
        }


class MultiLabelTransformerClassifier(nn.Module):
    """Multi-label transformer classifier for privacy harms."""

    def __init__(self, model_name: str, num_labels: int, hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class TransformerHarmClassifier:
    """High-level interface for transformer-based harm classification."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers and/or torch not available. Install with: "
                "pip install transformers torch"
            )

        self.config = config or TransformerConfig()
        self.device = self._get_device()
        self.tokenizer: Any = None
        self.model: Optional[MultiLabelTransformerClassifier] = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.label_vocab: List[str] = []

        # Comprehensive harm taxonomy (100+ categories as requested)
        self.default_taxonomy = self._get_comprehensive_taxonomy()

    def _get_device(self) -> str:
        """Determine the appropriate device for training/inference."""
        if self.config.device == "auto":
            try:
                if torch.cuda.is_available():
                    return "cuda"
                mps = getattr(torch.backends, "mps", None)
                if mps is not None and mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        return self.config.device

    def _get_comprehensive_taxonomy(self) -> List[str]:
        """Get comprehensive privacy harm taxonomy with 100+ categories."""
        return [
            # Core Privacy Violations
            "unauthorized_access",
            "data_breach",
            "identity_theft",
            "financial_fraud",
            "healthcare_privacy_violation",
            "biometric_misuse",
            "genetic_privacy_violation",
            "location_tracking",
            "surveillance_overreach",
            "government_surveillance",
            # Consent and Collection Issues
            "lack_of_consent",
            "deceptive_consent",
            "forced_consent",
            "unclear_consent",
            "excessive_data_collection",
            "unnecessary_data_collection",
            "secret_collection",
            "collection_without_notice",
            "collection_scope_creep",
            "data_minimization_violation",
            # Processing and Use Violations
            "purpose_limitation_violation",
            "function_creep",
            "secondary_use_violation",
            "automated_decision_making",
            "algorithmic_discrimination",
            "profiling_violation",
            "inference_violation",
            "behavioral_manipulation",
            "dark_patterns",
            "predictive_analytics_misuse",
            # Sharing and Disclosure
            "unauthorized_sharing",
            "third_party_sharing",
            "data_selling",
            "cross_border_transfer",
            "inadequate_sharing_controls",
            "vendor_sharing_violation",
            "partner_sharing_violation",
            "government_disclosure",
            "law_enforcement_sharing",
            "commercial_sharing",
            # Security Failures
            "inadequate_security",
            "encryption_failure",
            "access_control_failure",
            "authentication_weakness",
            "insider_threat",
            "vendor_security_failure",
            "cloud_security_violation",
            "database_exposure",
            "backup_exposure",
            "transmission_security_failure",
            # Rights Violations
            "access_right_denial",
            "correction_right_denial",
            "deletion_right_denial",
            "portability_right_denial",
            "objection_right_denial",
            "subject_rights_delay",
            "subject_rights_fees",
            "right_to_explanation_denial",
            "rectification_denial",
            "erasure_violation",
            # Special Categories
            "child_privacy_violation",
            "student_privacy_violation",
            "employee_privacy_violation",
            "medical_privacy_violation",
            "financial_privacy_violation",
            "communication_privacy",
            "social_media_privacy",
            "iot_privacy_violation",
            "smart_home_violation",
            "vehicle_privacy_violation",
            # Technology-Specific
            "facial_recognition_misuse",
            "voice_recognition_misuse",
            "fingerprint_misuse",
            "ai_privacy_violation",
            "machine_learning_bias",
            "deep_learning_violation",
            "nlp_privacy_violation",
            "computer_vision_violation",
            "recommendation_violation",
            "search_privacy_violation",
            # Contextual and Social
            "workplace_surveillance",
            "academic_surveillance",
            "retail_surveillance",
            "public_space_surveillance",
            "online_tracking",
            "cross_device_tracking",
            "behavioral_tracking",
            "purchasing_behavior_tracking",
            "health_tracking",
            "fitness_tracking",
            # Jurisdictional and Regulatory
            "gdpr_violation",
            "ccpa_violation",
            "coppa_violation",
            "hipaa_violation",
            "ferpa_violation",
            "glba_violation",
            "regulatory_non_compliance",
            "cross_border_violation",
            "safe_harbor_violation",
            "adequacy_violation",
            # Positive Privacy Features (as requested)
            "positive_privacy_feature",
            "privacy_by_design",
            "privacy_enhancing_technology",
            "strong_encryption",
            "anonymization_success",
            "pseudonymization_success",
            "differential_privacy",
            "zero_knowledge_proof",
            "homomorphic_encryption",
            "secure_multiparty_computation",
            # Additional Categories
            "notification_failure",
            "transparency_failure",
            "accountability_failure",
            "privacy_impact_assessment_failure",
            "data_protection_officer_failure",
            "record_keeping_failure",
            "audit_failure",
            "incident_response_failure",
            "breach_notification_delay",
            "regulatory_reporting_failure",
        ]

    def bootstrap_labels(self, texts: List[str]) -> List[List[str]]:
        """Generate initial labels using keyword-based heuristics."""
        keyword_mapping = {
            "data_breach": [
                "breach",
                "hack",
                "cyber attack",
                "unauthorized access",
                "exposed",
            ],
            "identity_theft": [
                "identity theft",
                "ssn",
                "social security",
                "stolen identity",
            ],
            "financial_fraud": [
                "financial fraud",
                "credit card",
                "bank account",
                "payment",
            ],
            "healthcare_privacy_violation": ["medical", "health", "hipaa", "patient"],
            "biometric_misuse": [
                "biometric",
                "fingerprint",
                "facial recognition",
                "iris",
            ],
            "location_tracking": ["location", "gps", "tracking", "geolocation"],
            "surveillance_overreach": ["surveillance", "monitoring", "spying"],
            "child_privacy_violation": ["child", "children", "minor", "coppa"],
            "gdpr_violation": ["gdpr", "general data protection"],
            "positive_privacy_feature": [
                "privacy by design",
                "encryption",
                "anonymization",
                "security enhancement",
            ],
        }

        labels = []
        for text in texts:
            text_lower = text.lower()
            text_labels = []

            for category, keywords in keyword_mapping.items():
                if any(keyword in text_lower for keyword in keywords):
                    text_labels.append(category)

            # If no labels found, assign a default
            if not text_labels:
                text_labels = ["general_privacy_concern"]

            labels.append(text_labels)

        return labels

    def train(self, texts: List[str], labels: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """Train the transformer-based classifier."""
        if labels is None:
            logger.info("No labels provided, using bootstrap heuristics")
            labels = self.bootstrap_labels(texts)

        # Build label vocabulary
        all_labels = set()
        for label_list in labels:
            all_labels.update(label_list)

        # Use default taxonomy if labels are sparse
        if len(all_labels) < 20:
            all_labels.update(self.default_taxonomy)

        self.label_vocab = sorted(list(all_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.label_vocab)}

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = MultiLabelTransformerClassifier(
            self.config.model_name,
            len(self.label_vocab),
            self.config.hidden_dropout_prob,
        ).to(self.device)

        # Create dataset and collator
        dataset = HarmDataset(texts, labels, self.label_to_idx)
        collator = HarmCollator(self.tokenizer, self.config.max_length)

        # Adjust performance settings based on device
        effective_batch_size = self.config.batch_size
        if self.device in {"cuda", "mps"}:
            # Scale batch size conservatively; allow override if already large
            if effective_batch_size < 24:
                effective_batch_size = min(32, effective_batch_size * 2)
            # Auto set gradient accumulation to smooth memory usage for very large batches
            if effective_batch_size >= 32 and self.config.gradient_accumulation_steps == 1:
                self.config.gradient_accumulation_steps = 2

        dataloader = DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True, collate_fn=collator
        )

        # Training setup
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        total_steps = len(dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.model.train()
        total_loss = 0.0

        # Mixed precision setup (simplified)
        amp_enabled = False
        amp_device_type = None
        scaler = None
        if self.config.use_mixed_precision and self.device in {"cuda", "mps"}:
            amp_enabled = True
            amp_device_type = self.device
            if self.device == "cuda":
                try:  # GradScaler only for CUDA
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    scaler = None

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            step_count = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                def forward_pass():
                    logits_fp = self.model(input_ids, attention_mask)
                    loss_fp = criterion(logits_fp, labels_batch)
                    return loss_fp, logits_fp

                if amp_enabled and amp_device_type is not None:
                    autocast_ctx = torch.autocast(device_type=amp_device_type, dtype=torch.float16)  # type: ignore[arg-type]
                    with autocast_ctx:
                        loss, _ = forward_pass()
                else:
                    loss, _ = forward_pass()

                # Gradient accumulation
                grad_acc = self.config.gradient_accumulation_steps
                if scaler:
                    scaler.scale(loss / grad_acc).backward()
                else:
                    (loss / grad_acc).backward()

                step_count += 1
                if step_count % grad_acc == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {epoch_loss:.4f}, "
                f"BatchSize={effective_batch_size}, GradAcc={self.config.gradient_accumulation_steps}, Device={self.device}"
            )

        return {
            "total_loss": total_loss,
            "num_epochs": self.config.num_epochs,
            "num_labels": len(self.label_vocab),
            "num_samples": len(texts),
            "batch_size": effective_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "device": self.device,
        }

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict harm categories for given texts."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        results = []

        # Use DataLoader for batched prediction
        dataset = HarmDataset(texts, None, self.label_to_idx)
        collator = HarmCollator(self.tokenizer, self.config.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collator
        )

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                batch_probs = torch.sigmoid(logits).cpu().numpy()

                for probabilities in batch_probs:
                    # Create category scores
                    category_scores = {}
                    for idx, prob in enumerate(probabilities):
                        if prob >= self.config.threshold:
                            category_scores[self.idx_to_label[idx]] = float(prob)
                    results.append(category_scores)

        return results

    def annotate_records(
        self, records: List[Dict[str, Any]], text_field: str = "description"
    ) -> List[Dict[str, Any]]:
        """Annotate records with harm classifications."""
        texts = []
        valid_indices = []

        for i, record in enumerate(records):
            text = record.get(text_field, "") or record.get("raw", {}).get("text", "")
            if text:
                texts.append(str(text))
                valid_indices.append(i)

        if not texts:
            return records

        predictions = self.predict(texts)

        # Update records
        for valid_idx, pred in zip(valid_indices, predictions):
            record = records[valid_idx]
            record["harm_category_scores"] = pred
            record["harm_categories"] = list(pred.keys())
            record["harm_score"] = max(pred.values()) if pred else 0.0
            record["classification_version"] = f"transformer-{self.config.model_name}"
            record["categories_extracted_from"] = text_field

        return records

    def save(self, save_dir: Path) -> None:
        """Save the trained model and configuration."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            raise ValueError("No trained model to save")

        # Save model and tokenizer
        #
        # Note: On Apple Silicon (MPS) some tensors can be non-contiguous which
        # causes safetensors to raise when saving. We avoid this by moving the
        # transformer temporarily to CPU and disabling safe serialization so
        # weights are stored using PyTorch's format (pytorch_model.bin), which
        # loads fine via AutoModel.from_pretrained().
        orig_device = next(self.model.parameters()).device  # remember device
        try:
            self.model.transformer.to("cpu")
            # Use safe_serialization=False to bypass safetensors contiguity checks
            self.model.transformer.save_pretrained(
                save_dir / "transformer", safe_serialization=False
            )
        finally:
            # Restore model device
            try:
                self.model.to(orig_device)
            except Exception:
                # If moving back fails, continue; loading always moves to target device
                pass
        self.tokenizer.save_pretrained(save_dir / "tokenizer")

        # Save classifier head (ensure CPU tensors for portability)
        head_state = {k: v.detach().cpu() for k, v in self.model.classifier.state_dict().items()}
        torch.save(head_state, save_dir / "classifier_head.pt")

        # Save metadata
        metadata = {
            "config": {
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "threshold": self.config.threshold,
                "hidden_dropout_prob": self.config.hidden_dropout_prob,
            },
            "label_vocab": self.label_vocab,
            "label_to_idx": self.label_to_idx,
            "num_labels": len(self.label_vocab),
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, save_dir: Path) -> None:
        """Load a trained model from disk."""
        save_dir = Path(save_dir)

        # Load metadata
        with open(save_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Restore configuration
        config_dict = metadata["config"]
        self.config.model_name = config_dict["model_name"]
        self.config.max_length = config_dict["max_length"]
        self.config.threshold = config_dict["threshold"]
        self.config.hidden_dropout_prob = config_dict["hidden_dropout_prob"]

        self.label_vocab = metadata["label_vocab"]
        self.label_to_idx = metadata["label_to_idx"]
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir / "tokenizer")

        self.model = MultiLabelTransformerClassifier(
            self.config.model_name,
            len(self.label_vocab),
            self.config.hidden_dropout_prob,
        ).to(self.device)

        # Load transformer weights
        transformer = AutoModel.from_pretrained(save_dir / "transformer")
        # Ensure transformer is on the target device
        self.model.transformer = transformer.to(self.device)

        # Load classifier head
        self.model.classifier.load_state_dict(
            torch.load(save_dir / "classifier_head.pt", map_location=self.device)
        )

        self.model.eval()


def check_transformers_available() -> bool:
    """Check if transformers and torch are available."""
    return TRANSFORMERS_AVAILABLE
