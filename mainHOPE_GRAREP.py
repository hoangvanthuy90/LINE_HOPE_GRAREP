from __future__ import print_function

import random
from graph import *
from grarep import GraRep
from hope import HOPE

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    g = Graph()
    print("Reading...")
    g.read_adjlist("karate_club.adjlist")


    model = HOPE(graph=g, d=8)
    model.save_embeddings("embeddindgoutput_hope")


    #model = GraRep(graph=g, Kstep=3, dim=12)
    #model.save_embeddings("embeddindgoutput_line")

    print("Done.")