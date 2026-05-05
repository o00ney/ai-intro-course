import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


IMG_SHAPE = (28, 28, 1)
LATENT_DIM = 100


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    sample_interval: int
    output_dir: str
    results_dir: str
    seed: int | None
    d_learning_rate: float
    g_learning_rate: float
    beta_1: float
    real_label: float


def build_generator(latent_dim: int = LATENT_DIM) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=1, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(
                64, kernel_size=5, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(
                1, kernel_size=5, strides=2, padding="same", activation="tanh"
            ),
        ],
        name="generator",
    )


def build_discriminator() -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=IMG_SHAPE),
            layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="discriminator",
    )


def save_image_grid(generator: tf.keras.Model, noise: tf.Tensor, epoch: int, output_dir: str) -> None:
    generated_images = generator(noise, training=False).numpy()
    generated_images = (generated_images + 1.0) / 2.0

    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    index = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(generated_images[index, :, :, 0], cmap="gray")
            axes[i, j].axis("off")
            index += 1
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)


def save_training_curves(history: list[dict], output_path: str) -> None:
    epochs = [item["epoch"] for item in history]
    d_loss = [item["d_loss"] for item in history]
    g_loss = [item["g_loss"] for item in history]
    real_score = [item["real_score"] for item in history]
    fake_score = [item["fake_score"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, d_loss, label="D loss")
    axes[0].plot(epochs, g_loss, label="G loss")
    axes[0].set_title("GAN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, real_score, label="Real score")
    axes[1].plot(epochs, fake_score, label="Fake score")
    axes[1].set_title("Discriminator Scores")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sigmoid(logit)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_training_summary(history: list[dict], config: TrainConfig, output_path: str) -> None:
    summary = {
        "config": asdict(config),
        "epochs_trained": len(history),
        "final_metrics": history[-1] if history else {},
        "best_g_loss_epoch": min(history, key=lambda item: item["g_loss"])["epoch"]
        if history
        else None,
        "best_d_loss_epoch": min(history, key=lambda item: item["d_loss"])["epoch"]
        if history
        else None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def prepare_dataset(batch_size: int, seed: int | None) -> tuple[tf.data.Dataset, int]:
    (x_train, _), _ = mnist.load_data()
    x_train = x_train.astype("float32")
    x_train = (x_train - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=x_train.shape[0], seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = x_train.shape[0] // batch_size
    return dataset, steps_per_epoch


def train(config: TrainConfig) -> None:
    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    dataset, steps_per_epoch = prepare_dataset(config.batch_size, config.seed)
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.g_learning_rate, beta_1=config.beta_1
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.d_learning_rate, beta_1=config.beta_1
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    fixed_noise = tf.random.normal((25, LATENT_DIM))

    metrics_csv_path = os.path.join(config.results_dir, "training_metrics.csv")
    curves_png_path = os.path.join(config.results_dir, "training_curves.png")
    summary_json_path = os.path.join(config.results_dir, "training_summary.json")
    generator_model_path = os.path.join(config.results_dir, "generator.keras")
    discriminator_model_path = os.path.join(config.results_dir, "discriminator.keras")

    @tf.function
    def train_step(real_images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        noise = tf.random.normal((config.batch_size, LATENT_DIM))
        real_labels = tf.ones((config.batch_size, 1)) * config.real_label
        fake_labels = tf.zeros((config.batch_size, 1))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = generator(noise, training=True)

            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)

            g_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            d_real_loss = loss_fn(real_labels, real_logits)
            d_fake_loss = loss_fn(fake_labels, fake_logits)
            d_loss = d_real_loss + d_fake_loss

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(d_gradients, discriminator.trainable_variables)
        )

        real_score = tf.reduce_mean(tf.sigmoid(real_logits))
        fake_score = tf.reduce_mean(tf.sigmoid(fake_logits))
        return d_loss, g_loss, real_score, fake_score

    history: list[dict] = []
    progress_interval = max(1, steps_per_epoch // 10)

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "d_loss", "g_loss", "real_score", "fake_score"],
        )
        writer.writeheader()

        save_image_grid(generator, fixed_noise, 0, config.output_dir)

        for epoch in range(1, config.epochs + 1):
            d_loss_metric = tf.keras.metrics.Mean()
            g_loss_metric = tf.keras.metrics.Mean()
            real_score_metric = tf.keras.metrics.Mean()
            fake_score_metric = tf.keras.metrics.Mean()

            print(f"Epoch {epoch}/{config.epochs} started ({steps_per_epoch} steps)")

            for step, real_images in enumerate(dataset, start=1):
                d_loss, g_loss, real_score, fake_score = train_step(real_images)

                d_loss_metric.update_state(d_loss)
                g_loss_metric.update_state(g_loss)
                real_score_metric.update_state(real_score)
                fake_score_metric.update_state(fake_score)

                if step == 1 or step % progress_interval == 0 or step == steps_per_epoch:
                    print(
                        f"  Step {step}/{steps_per_epoch} "
                        f"[D loss: {float(d_loss):.4f}] "
                        f"[G loss: {float(g_loss):.4f}] "
                        f"[real score: {float(real_score):.3f}] "
                        f"[fake score: {float(fake_score):.3f}]"
                    )

            row = {
                "epoch": epoch,
                "d_loss": float(d_loss_metric.result().numpy()),
                "g_loss": float(g_loss_metric.result().numpy()),
                "real_score": float(real_score_metric.result().numpy()),
                "fake_score": float(fake_score_metric.result().numpy()),
            }
            history.append(row)
            writer.writerow(row)

            print(
                f"Epoch {epoch}/{config.epochs} "
                f"[D loss: {row['d_loss']:.4f}] "
                f"[G loss: {row['g_loss']:.4f}] "
                f"[real score: {row['real_score']:.3f}] "
                f"[fake score: {row['fake_score']:.3f}]"
            )

            if epoch % config.sample_interval == 0 or epoch == config.epochs:
                save_image_grid(generator, fixed_noise, epoch, config.output_dir)
                generator.save(generator_model_path)
                discriminator.save(discriminator_model_path)

    save_training_curves(history, curves_png_path)
    save_training_summary(history, config, summary_json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stable GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=2,
        help="Save generated image grid every N epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gan_mnist",
        help="Directory for generated sample images.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="gan_results",
        help="Directory for metrics, plots, and saved models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--d-learning-rate",
        type=float,
        default=2e-4,
        help="Discriminator learning rate.",
    )
    parser.add_argument(
        "--g-learning-rate",
        type=float,
        default=2e-4,
        help="Generator learning rate.",
    )
    parser.add_argument(
        "--beta-1",
        type=float,
        default=0.5,
        help="Adam beta_1 for both optimizers.",
    )
    parser.add_argument(
        "--real-label",
        type=float,
        default=0.9,
        help="Label smoothing value for real images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_interval=args.sample_interval,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            d_learning_rate=args.d_learning_rate,
            g_learning_rate=args.g_learning_rate,
            beta_1=args.beta_1,
            real_label=args.real_label,
        )
    )
