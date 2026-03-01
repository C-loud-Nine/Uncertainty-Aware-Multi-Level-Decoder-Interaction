"""Training script for multi-task breast ultrasound model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import config
from model import enhanced_hct_model
from loss import enhanced_lesion_focus_loss, enhanced_multi_modal_focal_loss
from data_loader import prepare_datasets


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================
class CompositeMetric(Callback):
    """Combined metric: 70% IoU + 30% Accuracy"""
    def on_epoch_end(self, epoch, logs=None):
        seg_iou = logs.get("val_segmentation_output_mean_io_u", 0)
        clf_acc = logs.get("val_classification_output_accuracy", 0)
        val_combined = 0.7 * seg_iou + 0.3 * clf_acc
        logs["val_combined"] = val_combined
        print(f"\nEpoch {epoch+1}: Combined Metric = {val_combined:.4f}")


class AdaptiveLossWeights(Callback):
    """Dynamically adjust loss weights during training"""
    def __init__(self):
        super().__init__()
        self.seg_weight = config.SEG_WEIGHT_START
        self.clf_weight = config.CLF_WEIGHT_START
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 15:
            self.seg_weight = min(0.92, self.seg_weight + 0.007)
            self.clf_weight = max(0.08, self.clf_weight - 0.007)
        else:
            self.seg_weight = config.SEG_WEIGHT_FINAL
            self.clf_weight = config.CLF_WEIGHT_FINAL
        
        self.model.loss_weights = {
            "segmentation_output": self.seg_weight,
            "classification_output": self.clf_weight
        }
        print(f"Loss weights: Seg={self.seg_weight:.3f}, Clf={self.clf_weight:.3f}")


class CosineDecayScheduler(Callback):
    """Cosine annealing learning rate schedule"""
    def __init__(self, initial_lr, total_epochs, alpha=0.03):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.alpha = alpha
    
    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / self.total_epochs
        lr = self.alpha + 0.5 * (1 - self.alpha) * (1 + np.cos(np.pi * progress))
        current_lr = self.initial_lr * lr
        tf.keras.backend.set_value(self.model.optimizer.lr, current_lr)
        print(f"Learning rate: {current_lr:.7f}")


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print("=" * 80)
    print("Multi-Task Breast Ultrasound Training")
    print("=" * 80)
    
    # Load data
    data = prepare_datasets(config.BUSI_PATH)
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_train'],
        {
            "segmentation_output": data['mask_train'],
            "classification_output": data['y_train']
        }
    )).shuffle(3000).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_val'],
        {
            "segmentation_output": data['mask_val'],
            "classification_output": data['y_val']
        }
    )).batch(config.VAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print("\nBuilding model...")
    model = enhanced_hct_model(
        input_size=config.INPUT_SIZE,
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_clf_classes=config.NUM_CLF_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LR,
            beta_1=config.ADAM_BETA_1,
            beta_2=config.ADAM_BETA_2,
            epsilon=config.ADAM_EPSILON,
            global_clipnorm=config.GLOBAL_CLIPNORM
        ),
        loss={
            "segmentation_output": enhanced_lesion_focus_loss,
            "classification_output": enhanced_multi_modal_focal_loss
        },
        metrics={
            "segmentation_output": [
                tf.keras.metrics.MeanIoU(num_classes=2, name="mean_io_u"),
                tf.keras.metrics.BinaryAccuracy(name="bin_acc")
            ],
            "classification_output": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        },
        loss_weights={
            "segmentation_output": config.SEG_WEIGHT_START,
            "classification_output": config.CLF_WEIGHT_START
        }
    )
    
    # Callbacks
    composite_metric = CompositeMetric()
    adaptive_weights = AdaptiveLossWeights()
    lr_scheduler = CosineDecayScheduler(
        initial_lr=config.INITIAL_LR,
        total_epochs=config.EPOCHS,
        alpha=config.COSINE_ALPHA
    )
    early_stop = EarlyStopping(
        monitor="val_combined",
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_combined",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        mode="max",
        cooldown=config.REDUCE_LR_COOLDOWN,
        verbose=1
    )
    
    # Warmup
    print("\n" + "=" * 80)
    print("WARMUP PHASE")
    print("=" * 80)
    model.fit(
        train_dataset.take(100),
        validation_data=val_dataset,
        epochs=config.WARMUP_EPOCHS,
        callbacks=[adaptive_weights, lr_scheduler],
        verbose=1
    )
    
    # Main training
    print("\n" + "=" * 80)
    print("MAIN TRAINING")
    print("=" * 80)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        initial_epoch=config.WARMUP_EPOCHS,
        callbacks=[
            composite_metric,
            early_stop,
            reduce_lr,
            lr_scheduler,
            adaptive_weights
        ],
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.h5')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_test'],
        {
            "segmentation_output": data['mask_test'],
            "classification_output": data['y_test']
        }
    )).batch(config.VAL_BATCH_SIZE)
    
    test_results = model.evaluate(test_dataset, verbose=1)
    print("\nTest Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
