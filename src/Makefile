include ../Make.inc

SRCS  = $(wildcard *.c)
OBJS  = $(SRCS:%.c=%.o)
DOBJS = $(SRCS:%.c=%.do)

ifneq ($(MAKECMDGOALS),debug)
XOBJS = $(OBJS)
else
XOBJS = $(DOBJS)
endif

CFLAGS-add += -DMATHLIB_STANDALONE


default: release

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS-add) $(CFLAGS) $(fPIC) -std=gnu99 -I../include -DNDEBUG -c $< -o $@
%.do: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS-add) $(CFLAGS) $(fPIC) -std=gnu99 -I../include -g  -c $< -o $@

release debug: libRmath-julia.$(SHLIB_EXT)

libRmath-julia.$(SHLIB_EXT): $(XOBJS)
	-rm -f $@
	$(CC) $(LDFLAGS-add) $(LDFLAGS) -shared -o $@ $^

clean:
	-rm -f *.o *.do *.a *.$(SHLIB_EXT) core* *~ *#
