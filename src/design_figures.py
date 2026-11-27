import matplotlib.pyplot as plt
from prepare_fer2013 import FIGURES


def diagram(name,title,labels):
    fig,ax = plt.subplots(figsize=(10,3.2))
    ax.set(xlim=(0,1),ylim=(0,1),title=title)
    ax.axis("off")
    for i,label in enumerate(labels):
        x = (i+.5)/len(labels)
        ax.text(x,.5,label,ha="center",va="center",fontsize=10,
                bbox=dict(boxstyle="round,pad=.6",fc="#e3f3f1",ec="#23766e"))
        if i:
            ax.annotate("",xy=(x-.071,.5),xytext=(x-1/len(labels)+.071,.5),
                        arrowprops=dict(arrowstyle="->",color="#334455"))
    fig.tight_layout()
    fig.savefig(FIGURES/f"{name}.png",dpi=220)
    plt.close(fig)


if __name__=="__main__":
    diagram("system_architecture","Offline observable movement descriptions",
            ["Camera","Landmarks\n+ blendshapes","Face selection\n+ visibility","Thresholds\n+ smoothing","Sentence\n+ offline TTS"])
    diagram("cnn_architecture","FER2013 compact CNN",
            ["48 x 48 gray","Conv 32\nBN/ReLU/pool","Conv 64\nBN/ReLU/pool","Conv 96\nBN/ReLU/pool","Global pool\nDropout / 7 logits"])
