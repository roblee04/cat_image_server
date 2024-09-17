import clip
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, cast
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def populate_db():
    image_urls = [
        'http://images.cocodataset.org/val2014/COCO_val2014_000000159977.jpg',
        'http://images.cocodataset.org/val2014/COCO_val2014_000000311295.jpg',
        'http://images.cocodataset.org/val2014/COCO_val2014_000000457834.jpg',
        'http://images.cocodataset.org/val2014/COCO_val2014_000000555472.jpg',
        'http://images.cocodataset.org/val2014/COCO_val2014_000000174070.jpg',
        'http://images.cocodataset.org/val2014/COCO_val2014_000000460929.jpg'
        ]

    image_database = []
    for url in image_urls:
        image_database.append(Image.open(requests.get(url, stream=True).raw))


    image_database_processed = [
        preprocess(im) for im in image_database
    ]  # preprocess each Image
    with torch.no_grad():
        database_embeddings = model.encode_image(
            torch.stack(image_database_processed)
        )  # Torch.Stack will help us to levragebatch processing to speed up the calculation

    return database_embeddings

#how to save tensors?
# print(database_embeddings)

def text_image_search(query_text: str, database_embeddings: torch.Tensor):
    query_embeddings = model.encode_text(clip.tokenize([query_text]).to(device))

    similariries = query_embeddings @ database_embeddings.T
    return similariries


if __name__ == "__main__":

    database_embeddings = populate_db()
    query = "hot dog"
    sim = text_image_search(query, database_embeddings)
    sim_dict = dict(
        zip(range(len(sim[0])), sim[0])
    )  # Use Dictionary to Sort the Results
    sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    top_sim = sorted_sim[:6]  # Get top 6 results

    fig, axs = plt.subplots(2, 3, figsize=(15, 6), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=0.5, wspace=0.001)

    axs = axs.ravel()
    fig.suptitle(f"Text - Image Search: \nQuery: {query}")
    for num, i in enumerate(top_sim):
        axs[num].imshow(image_database[i[0]])
        axs[num].set_title(f"Similarity: {i[1]:.2f}")

    plt.show()
