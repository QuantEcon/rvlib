# -*- mode: makefile-gmake -*-

OS = $(shell uname)
ARCH = $(shell uname -m)

FC = gfortran
FFLAGS += -O3

USEGCC = 1
USECLANG = 0

CFLAGS += -std=c99 -Wall -O3 -I$(OPENLIBM_HOME) -I$(OPENLIBM_HOME)/include -I$(OPENLIBM_HOME)/ld80 -I$(OPENLIBM_HOME)/$(ARCH) -I$(OPENLIBM_HOME)/src -DASSEMBLER -D__BSD_VISIBLE -Wno-implicit-function-declaration

ifeq ($(USECLANG),1)
USEGCC = 0
CC = clang
CFLAGS += -fno-builtin
endif

ifeq ($(USEGCC),1)
CC = gcc
CFLAGS += -fno-gnu89-inline
endif
AR = ar

ifeq ($(OS), Linux)
SHLIB_EXT = so
CFLAGS+=-fPIC
FFLAGS+=-fPIC
endif

ifeq ($(OS), FreeBSD)
SHLIB_EXT = so
CFLAGS+=-fPIC
FFLAGS+=-fPIC
endif

ifeq ($(OS), Darwin)
SHLIB_EXT = dylib
CFLAGS+=-fPIC
FFLAGS+=-fPIC
endif

ifeq ($(OS), WINNT)
SHLIB_EXT = dll
endif
