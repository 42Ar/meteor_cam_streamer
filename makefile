CXX=g++
CPPFLAGS=-g3 -O0 -std=c++11 $(shell pkg-config --cflags opencv4)
LDFLAGS=
LIBS=$(shell pkg-config --libs opencv4)

SRCS=streamer.cpp rpoly.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: streamer

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $<

streamer: $(OBJS)
	$(CXX) $(LDFLAGS) -o streamer $(OBJS) $(LIBS)

clean:
	rm -f streamer $(OBJS)
