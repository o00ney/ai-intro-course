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
LATENT_DIM = 128


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    sample_interval: int
    output_dir: str
    results_dir: str
    seed: int | None
    g_learning_rate: float
    c_learning_rate: float
    beta_1: float
    beta_2: float
    n_critic: int
    gp_weight: float

def build_generator(latent_dim: int = LATENT_DIM) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(
                128, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(
                64, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(1, kernel_size=3, padding="same", activation="tanh"),
        ],
        name="generator",
    )


def build_critic() -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=IMG_SHAPE),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.2),
            layers.Conv2D(256, kernel_size=3, strides=1, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="critic",
    )


def save_image_grid(generator: tf.keras.Model, noise: tf.Tensor, epoch: int, output_dir: str) -> None:
    generated_images = generator(noise, training=False).numpy()
    generated_images = (generated_images + 1.0) / 2.0

    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    index = 0
    for i in range(5):
        for j in range(5):
            axes[i, j].imshow(generated_images[index, :, :, 0], cmap="gray")
            axes[i, j].axis("off")
            index += 1
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)


def save_training_curves(history: list[dict], output_path: str) -> None:
    epochs = [item["epoch"] for item in history]
    critic_loss = [item["critic_loss"] for item in history]
    generator_loss = [item["generator_loss"] for item in history]
    real_score = [item["real_score"] for item in history]
    fake_score = [item["fake_score"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, critic_loss, label="Critic loss")
    axes[0].plot(epochs, generator_loss, label="Generator loss")
    axes[0].set_title("WGAN-GP Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, real_score, label="Real score")
    axes[1].plot(epochs, fake_score, label="Fake score")
    axes[1].set_title("Critic Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean logit")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_training_summary(history: list[dict], config: TrainConfig, output_path: str) -> None:
    summary = {
        "config": asdict(config),
        "epochs_trained": len(history),
        "final_metrics": history[-1] if history else {},
        "best_generator_loss_epoch": min(
            history, key=lambda item: item["generator_loss"]
        )["epoch"]
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
    dataset = dataset.shuffle(x_train.shape[0], seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, x_train.shape[0] // batch_size


def train(config: TrainConfig) -> None:
    if config.seed is not None:
        tf.keras.utils.set_random_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    dataset, steps_per_epoch = prepare_dataset(config.batch_size, config.seed)
    generator = build_generator()
    critic = build_critic()

    g_optimizer = tf.keras.optimizers.Adam(
        config.g_learning_rate, beta_1=config.beta_1, beta_2=config.beta_2
    )
    c_optimizer = tf.keras.optimizers.Adam(
        config.c_learning_rate, beta_1=config.beta_1, beta_2=config.beta_2
    )
    fixed_noise = tf.random.normal((25, LATENT_DIM))

    metrics_csv_path = os.path.join(config.results_dir, "wgan_gp_metrics.csv")
    curves_png_path = os.path.join(config.results_dir, "wgan_gp_curves.png")
    summary_json_path = os.path.join(config.results_dir, "wgan_gp_summary.json")
    generator_model_path = os.path.join(config.results_dir, "wgan_gp_generator.keras")
    critic_model_path = os.path.join(config.results_dir, "wgan_gp_critic.keras")

    @tf.function
    def gradient_penalty(real_images: tf.Tensor, fake_images: tf.Tensor) -> tf.Tensor:
        alpha = tf.random.uniform([config.batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1.0 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = critic(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean(tf.square(norm - 1.0))

    @tf.function
    def critic_step(real_images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        noise = tf.random.normal((config.batch_size, LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_logits = critic(real_images, training=True)
            fake_logits = critic(fake_images, training=True)
            gp = gradient_penalty(real_images, fake_images)
            c_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            c_loss = c_loss + config.gp_weight * gp
        grads = tape.gradient(c_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        return c_loss, tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

    @tf.function
    def generator_step() -> tf.Tensor:
        noise = tf.random.normal((config.batch_size, LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            fake_logits = critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_logits)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    history: list[dict] = []
    progress_interval = max(1, steps_per_epoch // 10)

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "critic_loss", "generator_loss", "real_score", "fake_score"],
        )
        writer.writeheader()

        save_image_grid(generator, fixed_noise, 0, config.output_dir)

        for epoch in range(1, config.epochs + 1):
            critic_loss_metric = tf.keras.metrics.Mean()
            generator_loss_metric = tf.keras.metrics.Mean()
            real_score_metric = tf.keras.metrics.Mean()
            fake_score_metric = tf.keras.metrics.Mean()

            print(f"Epoch {epoch}/{config.epochs} started ({steps_per_epoch} steps)")

            for step, real_images in enumerate(dataset, start=1):
                c_loss = 0.0
                real_score = 0.0
                fake_score = 0.0
                for _ in range(config.n_critic):
                    c_loss, real_score, fake_score = critic_step(real_images)

                g_loss = generator_step()

                critic_loss_metric.update_state(c_loss)
                generator_loss_metric.update_state(g_loss)
                real_score_metric.update_state(real_score)
                fake_score_metric.update_state(fake_score)

                if step == 1 or step % progress_interval == 0 or step == steps_per_epoch:
                    print(
                        f"  Step {step}/{steps_per_epoch} "
                        f"[C loss: {float(c_loss):.4f}] "
                        f"[G loss: {float(g_loss):.4f}] "
                        f"[real score: {float(real_score):.3f}] "
                        f"[fake score: {float(fake_score):.3f}]"
                    )

            row = {
                "epoch": epoch,
                "critic_loss": float(critic_loss_metric.result().numpy()),
                "generator_loss": float(generator_loss_metric.result().numpy()),
                "real_score": float(real_score_metric.result().numpy()),
                "fake_score": float(fake_score_metric.result().numpy()),
            }
            history.append(row)
            writer.writerow(row)

            print(
                f"Epoch {epoch}/{config.epochs} "
                f"[C loss: {row['critic_loss']:.4f}] "
                f"[G loss: {row['generator_loss']:.4f}] "
                f"[real score: {row['real_score']:.3f}] "
                f"[fake score: {row['fake_score']:.3f}]"
            )

            if epoch % config.sample_interval == 0 or epoch == config.epochs:
                save_image_grid(generator, fixed_noise, epoch, config.output_dir)
                generator.save(generator_model_path)
                critic.save(critic_model_path)

    save_training_curves(history, curves_png_path)
    save_training_summary(history, config, summary_json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WGAN-GP on MNIST.")
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
        default="gan_mnist_wgan_gp",
        help="Directory for generated sample images.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="gan_results_wgan_gp",
        help="Directory for metrics, plots, and saved models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--g-learning-rate",
        type=float,
        default=1e-4,
        help="Generator learning rate.",
    )
    parser.add_argument(
        "--c-learning-rate",
        type=float,
        default=1e-4,
        help="Critic learning rate.",
    )
    parser.add_argument("--beta-1", type=float, default=0.0, help="Adam beta_1.")
    parser.add_argument("--beta-2", type=float, default=0.9, help="Adam beta_2.")
    parser.add_argument(
        "--n-critic",
        type=int,
        default=5,
        help="Number of critic updates per generator update.",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=10.0,
        help="Gradient penalty weight.",
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
            g_learning_rate=args.g_learning_rate,
            c_learning_rate=args.c_learning_rate,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            n_critic=args.n_critic,
            gp_weight=args.gp_weight,
        )
    )
