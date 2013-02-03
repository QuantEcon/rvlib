# -*- mode: makefile-gmake -*-

OS = $(shell uname)
ARCH = $(shell uname -m)

FC = gfortran
FFLAGS += -O3

USEGCC = 1
USECLANG = 0

CFLAGS += -Wall -O3 

ifeq ($(USECLANG),1)
USEGCC = 0
CC = clang
endif

ifeq ($(USEGCC),1)
CC = gcc
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
