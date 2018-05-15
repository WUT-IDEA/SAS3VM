CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

all: nb_learn nb_classify nbem_ssl nbem_usl

nb_learn: nb_learn.cpp NB.o
	$(CXX) $(CFLAGS) nb_learn.cpp NB.o -o nb_learn -lm
nb_classify: nb_classify.cpp NB.o
	$(CXX) $(CFLAGS) nb_classify.cpp NB.o -o nb_classify -lm
nbem_ssl: nbem_ssl.cpp NBEM.o NB.o
	$(CXX) $(CFLAGS) nbem_ssl.cpp NBEM.o NB.o -o nbem_ssl -lm
nbem_usl: nbem_usl.cpp NBEM.o NB.o
	$(CXX) $(CFLAGS) nbem_usl.cpp NBEM.o NB.o -o nbem_usl -lm
clean:
	rm -f NB.o NBEM.o nb_learn nb_classify nbem_ssl nbem_usl
