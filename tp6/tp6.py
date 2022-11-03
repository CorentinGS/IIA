from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns


def setup():
    matplotlib.use('TkAgg')  # or whatever other backend that you want to use. Required for my OS & Python version


def main():
    setup()
    ex3()

    return 0


def ex_scatter():
    # Open World_hapiness_dataset.csv file
    df = pd.read_csv('World_hapiness_dataset_2019.csv', sep=',', header=0)

    plt.scatter(df['GDP per capita'], df['Score'], s=50)
    plt.show()

    sns.scatterplot(x='GDP per capita', y='Score', data=df)
    plt.show()

    sns.catplot(x='GDP per capita', y='Score', data=df)
    plt.show()


def ex3():
    # Open wine.csv file
    df = pd.read_csv('wine.csv', sep=',', header=0)

    # Convert to numpy
    X = df.to_numpy()

    pca: PCA = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)

    # 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(reduced_data)
    classe = kmeans.predict(reduced_data)

    labels_unique = np.unique(classe)
    classe = df["Class"]
    for label in labels_unique:
        to_plot = reduced_data[np.where(classe == label)[0]]
        plt.scatter(to_plot[:, 0], to_plot[:, 1], s=20)

    plt.show()

def ex2():
    # Open wine.csv file
    df = pd.read_csv('wine.csv', sep=',', header=0)
    # Print the first 10  rows
    print(df.head(10))
    # Print the last 10 rows
    print(df.tail(10))

    # Print info about the dataset
    print(df.info())

    # Check for null values
    print(df.isnull().sum())

    # Display labels of the dataset using np.unique()
    print(np.unique(df['Class']))


def ex1():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50)

    # Print the coordinates of the points
    for i in range(len(X)):
        print(X[i])

    plt.show()

    kmeans = KMeans(n_clusters=4)  # 4 clusters
    kmeans.fit(X)  # Fit the model to the data

    y_kmeans = kmeans.predict(X)  # Predict the clusters of the data

    # Display the clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_  # Get the coordinates of the centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    # Display X classes and y_kmeans classes
    print("X classes: ", y_true)
    print("y_kmeans classes: ", y_kmeans)


if __name__ == '__main__':
    main()
