C_SOURCES = $(wildcard tinydot/*.c)
CFLAGS = -fPIC -shared

CLANG = /usr/bin/clang

all: c_lib.so

c_lib.so: $(C_SOURCES)
	$(CLANG) $(CFLAGS) $(C_SOURCES) -o build/c_lib.so

