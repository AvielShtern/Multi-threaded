CC=g++
CXX=g++
RANLIB=ranlib

LIBSRC=Barrier.cpp MapReduceFramework.cpp
LIBOBJ=$(LIBSRC:.cpp=.o)
# LIBOBJ=MapReduceFrameWork.o

INCS=-I.
CFLAGS = -Wall -std=c++11 -g  $(INCS)
CXXFLAGS =  -Wall -std=c++11 -g $(INCS)

OSMLIB = libMapReduceFramework.a
TARGETS = $(OSMLIB)

TAR=tar
TARFLAGS=-cvf
TARNAME=ex3.tar
TARSRCS=$(LIBSRC) Makefile README Contexts.h Barrier.h

all: $(TARGETS)

$(TARGETS): $(LIBOBJ)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

clean:
	$(RM) $(TARGETS) $(OSMLIB) $(OBJ) $(LIBOBJ) *~ *core

depend:
	makedepend -- $(CFLAGS) -- $(SRC) $(LIBSRC)

tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TARSRCS)
