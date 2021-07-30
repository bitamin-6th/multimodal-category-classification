from ast import literal_eval
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Komoran, Kkma, Hannanum, Okt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import argparse


def encode_label(dataframe, column_name):
    encoder = preprocessing.LabelEncoder()
    dataframe[column_name] = encoder.fit_transform(dataframe[column_name])
    return dataframe


def load_dataframe(path="./preprocessed_dataset.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

def get_model():

    ## Text Brach
    input_text = tf.keras.layers.Input(shape=(100,))
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100)(input_text)
    x = layers.Conv1D(100, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.LSTM(100)(x)
    x = tf.keras.layers.Activation("relu")(x)
    text_feature = layers.BatchNormalization()(x)
    text_embed = tf.keras.Model(inputs=input_text, outputs=text_feature)

    ## Image Branch
    input_image = tf.keras.layers.Input(shape=(512, ))
    image_feature = layers.Dense(256)(input_image)
    image_feature = layers.BatchNormalization()(image_feature)
    image_feature= layers.Activation("relu")(image_feature)
    image_embed = tf.keras.Model(inputs=input_image, outputs=image_feature)

    ## Price Branch
    input_price = tf.keras.layers.Input(shape=(1, ))

    concatenated = layers.Concatenate()([text_embed.output, image_embed.output, input_price])
    concatenated = layers.Dense(520)(concatenated)
    concatenated = layers.BatchNormalization()(concatenated)
    concatenated = layers.Activation("relu")(concatenated)
    outputs = layers.Dense(520, activation="softmax")(concatenated)

    model = tf.keras.Model(inputs=[input_text,input_image, input_price], outputs=outputs)
    return model

parser = argparse.ArgumentParser(description="Audio Text Clip Implementation")

parser.add_argument("--epochs", default=200, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=256, type=int,
                help="batch size of training")
parser.add_argument("--split", default=0.2, type=float, 
                help="train test split ratio")
parser.add_argument("--seed", default=42, type=int, 
                help="train test split ratio")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                help='initial learning rate')

args = parser.parse_args()

if __name__ == "__main__":

    df = load_dataframe()
    okt = Okt()
    df["label"] = df["label_big"] + ">" + df["label_medium"] + ">" + df["label_small"]

    df = encode_label(df, "label")

    df["image_feature"] = df["image_feature"].apply(literal_eval)
    df["text_feature"] = df["name"].apply(okt.nouns)

    df_train, df_test = train_test_split(df, test_size=args.split, random_state=args.seed)
    y_train = df_train["label"]
    y_test = df_test["label"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train["text_feature"])
    rare_cnt = 0

    for k, v in tokenizer.word_counts.items():
        if v < 2:
            rare_cnt += 1

    vocab_size = len(tokenizer.word_index) - rare_cnt + 2
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(df_train["text_feature"])

    x_train_text = tokenizer.texts_to_sequences(df_train["text_feature"])
    x_test_text = tokenizer.texts_to_sequences(df_test["text_feature"])

    x_train_text = pad_sequences(x_train_text, maxlen=100)
    x_test_text = pad_sequences(x_test_text, maxlen=100)

    x_train_image = df_train["image_feature"].values.tolist()
    x_test_image = df_test["image_feature"].values.tolist()

    x_train_price = df_train["price"].str.replace(",","").astype(float).values.reshape(-1, 1)
    x_test_price = df_test["price"].str.replace(",","").astype(float).values.reshape(-1, 1)

    scaler = StandardScaler()
    x_train_price = scaler.fit_transform(x_train_price)
    x_test_price = scaler.transform(x_test_price)


    model = get_model()

    cb_checkpoint = ModelCheckpoint(filepath="model.h5", monitor='val_acc',
                                    verbose=1, save_best_only=True)
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(args.lr),
                    metrics=["acc"])

    history = model.fit([np.asarray(x_train_text), np.asarray(x_train_image), x_train_price], np.asarray(y_train), 
                epochs=args.epochs,
                batch_size=args.batch_size, validation_data=([np.asarray(x_test_text), np.asarray(x_test_image), x_test_price], np.asarray(y_test)),
                callbacks=[cb_checkpoint])

    model = load_model("model.h5")

    predictions = np.argmax(model.predict([np.asarray(x_test_text), np.asarray(x_test_image), x_test_price]), axis=-1)
    acc = np.mean((y_test==predictions).astype(np.float))
    print(acc)