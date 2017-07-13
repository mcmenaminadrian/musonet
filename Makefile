default: all

all: musonet

debug: debugmusonet

clean:
	rm -f *.o

musonet: main.o nnetwork.o
	g++ -std=c++11 -O2 -o musonet -Wall main.o nnetwork.o -ljpeg

main.o: main.cpp 
	g++ -std=c++11 -O2 -o main.o -c -Wall main.cpp

nnetwork.o: nnetwork.cpp
	g++ -std=c++11 -O2 -o nnetwork.o -c -Wall nnetwork.cpp


debugmusonet: dmain.o dnnetwork.o
	g++ -g -std=c++11 -o musonet -Wall dmain.o dnnetwork.o -ljpeg

dmain.o: main.cpp
	g++ -g -std=c++11 -o dmain.o -c -Wall main.cpp

dnnetwork.o: nnetwork.cpp
	g++ -g -std=c++11 -o dnnetwork.o -c -Wall nnetwork.cpp
