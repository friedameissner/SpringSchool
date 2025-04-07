import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from copy import copy
from tqdm.auto import tqdm

def setup(n=30):
    r"""
    Generiert drei Punktewolken (jeweils N(mu,1)) mit Mittelwerten mu=vv.

    Eingabe:
        n: Anzahl Punkte pro Wolke.

    Ausgabe:
        data: 3nx2-Array mit Punkten.
        vv: Mittelwerte in 3x2-Array.
    """
    vv = np.array([[5,5], [-5,0], [5,-6]])
    data = np.random.randn(n,2)+vv[0]
    data = np.row_stack([data, np.random.randn(n,2)+vv[1]])
    data = np.row_stack([data, np.random.randn(n,2)+vv[2]])
    return data, vv

def build_g(punkt1, punkt2):
    r"""
    Generiert Mittelsenkrechte zwischen zwei gegebenen Punkten.

    Eingabe:
        punkt1, punkt2: Numpy-Arrays der beiden Punkte, durch die die Mittelsenkrechte definiert wird.

    Ausgabe:
        w: Gewichtsvektor für linearen Klassifizierer
        theta: Bias für linearen Klassifizierer
        g: Linearer Klassifizierer als Lambda-Funktion
            Eingabe:
                x: Eingabepunkt
            Ausgabe:
                Klassifikation (entweder 0 oder 1)
    """

    m = 0.5*(punkt1+punkt2)
    w = punkt1-punkt2
    theta = w@m

    g = lambda x: 1*((x@w-theta)>=0)
    return w, theta, g


def plotline(axes, w, theta):
    r"""
    Plottet Klassifikationsgerade in gegebenes Koordinatensystem

    Eingabe:
        axes: Achsen einer plt.figure.
        w, theta: Gewichtsvektor und Bias des Klassifizierers

    Ausgabe:
        None
    """

    xbound = axes.get_xbound()
    ybound = axes.get_ybound()
    
    von = np.zeros(2)
    bis = np.zeros(2)
    if w[1]==0:
        # 0 = w1x1+w2x2-theta
        von[1] = ybound[0]
        bis[1] = ybound[1]
        von[0] = -theta-w[1]*von[1]
        bis[0] = von[0]
    else:
        von[0] = xbound[0]
        bis[0] = xbound[1]
        von[1] = (theta-w[0]*von[0])/w[1]
        bis[1] = (theta-w[0]*bis[0])/w[1]
    xxyy = np.stack((von, bis)).T
    plt.plot(xxyy[0], xxyy[1], "r")
    plt.xlim(xbound)
    plt.ylim(ybound)


def hex2rgb(c):
    r"""
    Übersetzt Farbe in Hexadezimaldarstellung in RGB-Werte.

    Eingabe:
        c: Farbe in Hexadezimaldarstellung.

    Ausgabe:
        RGB-Vektor.
    """
    return np.array(list(int(c[i:i+2], 16)/256 for i in (1, 3, 5)))

def rgb2hex(c):
    r"""
    Übersetzt Farbe in RGB-Werte in Hexadezimaldarstellung.

    Eingabe:
        c: RGB-Vektor.

    Ausgabe:
        Farbe in Hexadezimaldarstellung.
    """
    return '#%02x%02x%02x' % tuple((c*256).astype(int))

def load_img(path, maxlen=1000):
    r"""
    Lädt lokales Bild

    Eingabe:
        path: Speicherort des Bildes.
        maxlen: Maximale Länge der längeren Seite.

    Ausgabe:
        normalisiertes Bild als Numpy-Array.
    """
    img = plt.imread(path+".jpg")
    #img = io.imread(Link).astype(float)
    while np.max(img.shape)>maxlen:
        #print(np.max(img.shape))
        img = img[::2, ::2]
    return normalize(img)

def normalize(img):
    r"""
    Normalisiert Bild, sodass min=0 und max=1 ist.

    Eingabe:
        img: Bild als Numpy-Array, das normalisiert werden soll.

    Ausgabe:
        normalisiertes Bild als Numpy-Array.
    """
    i = img.copy()
    i -= np.min(i)
    i = i/np.max(i)
    return i

def imshow(img):
    r"""
    Blottet Bild

    Eingabe:
        img: Bild als Numpy-Array.

    Ausgabe:
        None
    """
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def plot_img_w_points(img, colors=None):
    r"""
    Blottet Bild neben RGB-Vektoren als Punktewolke.

    Eingabe:
        img: Bild als Numpy-Array.
        colors: Farbzuordnung der einzelnen Punkte

    Ausgabe:
        None
    """
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax.axis("off")
    if colors is None:
        colors = np.unique(np.reshape(img, (-1, 3)), axis=0)[::20]
        colors = 2*[colors]
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.scatter(colors[0][:,0], colors[0][:,1], colors[0][:,2], c=colors[1], s=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    plt.xlabel("x");plt.ylabel("y")
    fig.show()

def pointwise_sq(d1, d2):
    """
    Punktweiser L2-Abstand zweier Listen mit Punkten
    Eingabe:
        d1: Liste mit Punkten R^{nxm}
        d2: Liste mit Punkten R^{pxm}
    Ausgabe:
        out: Matrix mit Abständen zwischen jeweiligen Punkten R^{nxp}
    """
    out = np.sqrt(np.sum((d1[:,None]-d2)**2, axis=-1))
    return out


def zugehoerigkeit(codebook, data):
    r"""
    Berechnet für gegebene Daten die Indizes im Codebook (k Vektoren).

    Eingabe:
        codebook: Codebook als kxd-Numpy-Array.
        data: Daten als nxd-Numpy-Array

    Ausgabe:
        argmin: Vektor mit Indizes der nächsten Codebooks.
        error: Mittlerer Fehler über alle gegebenen Daten.
    """
    dists = np.sqrt(np.sum((codebook[:,None]-data)**2, axis=-1)).T
    argmin = np.argmin(dists, axis=-1)
    error = 0
    for i in range(len(argmin)):
        error += dists[i, argmin[i]]
    return argmin, error/len(data)

def draw(codebook, data, title=None):
    r"""
    Produziert Abbildung mit eingefärbten Punkten, passend zu gegebenen Codebookvektor.

    Eingabe:
        codebook: Codebook als kxd-Numpy-Array.
        data: Daten als nxd-Numpy-Array.
        title: Titel für Abbildung.

    Ausgabe:
        None
    """
    colors = list(mcolors.TABLEAU_COLORS.keys())
    label, _ = zugehoerigkeit(codebook, data)
    plt.figure()
    for i in range(len(codebook)):
        plt.scatter(data[label==i,0], data[label==i,1], c=colors[i%len(colors)], marker=".", alpha=0.5)
        plt.scatter(codebook[i,0], codebook[i,1], c=colors[i%len(colors)], marker="x", s=50, linewidths=4)
    plt.axis("equal")
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.xlabel("x");plt.ylabel("y")
    plt.show()


def run_experiments(data, n_runs=5):
    r"""
    Testet kMeans-Algorithmus für unterschiedliche Werte für k, jeweils n_runs mal.

    Eingabe:
        data: Daten als nxd-Numpy-Array.
        n_runs: Anzahl Runs pro Experiment für bessere Mittelwertschätzung.

    Ausgabe:
        None
    """
    idc = sorted([1,2,3]+[2*i for i in range(2,10)]+[5*i for i in range(5,20)]+[len(data)])
    errors = []
    for i in tqdm(idc):
        errorlist = []
        for _ in range(n_runs):
            codebook = np.random.permutation(data)[:i]
            #print(codebook.shape)
            d_error = np.inf
            e_neu = np.inf
            while d_error>1e-2:
                e_alt = copy(e_neu)
                codebook, e_neu = kmeans_update(codebook, data)
                d_error = abs(e_alt-e_neu)
                #print("Error",e_alt, e_neu, d_error)
            errorlist.append(e_neu)
        #print(errorlist)
        errors.append(np.mean(np.array(errorlist)))
        
    plt.figure()
    plt.plot(idc+[2*len(data)], errors+[errors[-1]])
    plt.xlabel("k");plt.ylabel("Fehler")
    plt.show()

def kmeans_update(codebook, data):
    r"""
    Updateschritt vom kMeans-Algorithmus mit Batchlearning. Optimiert durch Broadcasting

    Eingabe:
        codebook: Codebook als kxd-Numpy-Array.
        data: Daten als nxd-Numpy-Array.

    Ausgabe:
        cb: Aktualisierter Codebook.
        error: Aktueller MSE für gegebene Daten.
    """
    cb = codebook.copy()
    z, error = zugehoerigkeit(codebook, data)
    for i in range(len(cb)):
        if len(data[z==i])>0:
            cb[i] = np.mean(data[z==i], axis=0)
    return cb, error


def build_vq_img(colorlist, codebook, shape):
    r"""
    Updateschritt vom kMeans-Algorithmus mit Batchlearning. Optimiert durch Broadcasting

    Eingabe:
        codebook: Codebook als kxd-Numpy-Array.
        data: Daten als nxd-Numpy-Array.

    Ausgabe:
        cb: Aktualisierter Codebook.
        error: Aktueller MSE für gegebene Daten.
    """
    dists = pointwise_sq(colorlist, codebook)
    naechster = np.argmin(dists, axis=-1)
    img_neu_list = codebook[naechster]
    img_neu = np.reshape(img_neu_list, shape)
    return img_neu