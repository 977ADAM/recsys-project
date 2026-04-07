import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs
import numpy as np

# --------------------------------------------------
# 1) Пример входных данных:
# user_id, item_id, weight
# weight можно использовать для силы сигнала:
# view=1.0, click=2.0, cart=3.0, purchase=5.0
# --------------------------------------------------

user_ids = np.array(["u1", "u1", "u2", "u3", "u3", "u4"])
item_ids = np.array(["i2", "i7", "i3", "i2", "i9", "i1"])
weights  = np.array([1.0, 3.0, 1.0, 2.0, 5.0, 1.0], dtype=np.float32)

unique_user_ids = np.unique(user_ids)
unique_item_ids = np.unique(item_ids)

interactions = tf.data.Dataset.from_tensor_slices({
    "user_id": user_ids,
    "item_id": item_ids,
    "weight": weights,
})

items_ds = tf.data.Dataset.from_tensor_slices(unique_item_ids)

# --------------------------------------------------
# 2) Башня пользователя
# --------------------------------------------------

embedding_dim = 64

class UserModel(tf.keras.Model):
    def __init__(self, user_vocab):
        super().__init__()
        self.lookup = keras.layers.StringLookup(
            vocabulary=user_vocab, mask_token=None
        )
        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(user_vocab) + 1,
            output_dim=embedding_dim
        )
        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, user_id):
        x = self.lookup(user_id)
        x = self.embedding(x)
        return self.dnn(x)

# --------------------------------------------------
# 3) Башня объекта
# --------------------------------------------------

class ItemModel(tf.keras.Model):
    def __init__(self, item_vocab):
        super().__init__()
        self.lookup = tf.keras.layers.StringLookup(
            vocabulary=item_vocab, mask_token=None
        )
        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(item_vocab) + 1,
            output_dim=embedding_dim
        )
        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, item_id):
        x = self.lookup(item_id)
        x = self.embedding(x)
        return self.dnn(x)

# --------------------------------------------------
# 4) Retrieval model
# --------------------------------------------------

class CandidateGenerationModel(tfrs.models.Model):
    def __init__(self, user_vocab, item_vocab, items_dataset):
        super().__init__()
        self.user_model = UserModel(user_vocab)
        self.item_model = ItemModel(item_vocab)

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items_dataset.batch(1024).map(
                    lambda x: (x, self.item_model(x))
                )
            ),
            num_hard_negatives=50,
            remove_accidental_hits=True,
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["item_id"])

        return self.task(
            query_embeddings=user_embeddings,
            candidate_embeddings=item_embeddings,
            candidate_ids=features["item_id"],
            sample_weight=features["weight"],
            compute_metrics=not training,
        )

model = CandidateGenerationModel(
    user_vocab=unique_user_ids,
    item_vocab=unique_item_ids,
    items_dataset=items_ds,
)

train = (
    interactions
    .shuffle(10000)
    .batch(1024)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(train, epochs=5)